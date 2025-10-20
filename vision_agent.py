r"""
Vision Agent: Observe → Reason → Act loop
- Takes a screenshot, detects UI elements and overlays numbered green boxes
- Sends the annotated image(s) + a compact catalog and your GOAL to a VLM
- The model returns the NEXT ACTION in strict JSON
- Executes via MCP mouse/keyboard tools with PyAutoGUI fallback
- Loops until done or --max-steps is reached

Usage (PowerShell):
  # Ensure OPENAI_API_KEY is in env or .env in repo root
    python .\vision_agent.py --goal "Open the folder C:\\Users\\user\\Desktop\\dev and open training.html" --max-steps 6 --verbose

Notes:
- Allowed actions: click/move on a box number, type_text, key, hotkey, scroll, wait, done
- The agent prefers interacting with provided box numbers to remain grounded in the UI
"""

from __future__ import annotations
import os
import sys
import re
import json
import time
import argparse
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any

import pyautogui as pygui
from PIL import Image, ImageDraw, ImageFont

# VLM client
try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: openai. Install with 'pip install openai'.", file=sys.stderr)
    raise SystemExit(1)

# Local helpers
from bbox_detection import (
    detect_ui_elements,
    create_bbox_overlay,
)


# ------------- Minimal .env loader -------------

def _load_env_if_needed():
    if os.environ.get("OPENAI_API_KEY"):
        return
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip(); v = v.strip().strip('"').strip("'")
                    if k and (k not in os.environ):
                        os.environ[k] = v
    except Exception:
        pass


# ------------- MCP wrappers (fallback to PyAutoGUI) -------------

def _mcp_import():
    try:
        import importlib.util
        module_path = os.path.join(os.path.dirname(__file__), "mcp-mouse-keyboard-server", "server.py")
        spec = importlib.util.spec_from_file_location("mcp_mouse_server", module_path)
        if not spec or not spec.loader:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod
    except Exception:
        return None


def mcp_move(x: int, y: int, duration: float = 0.2) -> bool:
    mod = _mcp_import()
    try:
        if mod and hasattr(mod, "move_mouse"):
            mod.move_mouse(int(x), int(y), duration=float(duration))
            return True
    except Exception:
        pass
    try:
        pygui.moveTo(int(x), int(y), duration=duration)
        return True
    except Exception:
        return False


def mcp_click(button: str = "left", clicks: int = 1) -> bool:
    mod = _mcp_import()
    try:
        if mod and hasattr(mod, "click_mouse"):
            mod.click_mouse(button=button, clicks=int(clicks))
            return True
    except Exception:
        pass
    try:
        pygui.click(clicks=int(clicks), button=button)
        return True
    except Exception:
        return False


def mcp_type(text: str) -> bool:
    mod = _mcp_import()
    try:
        if mod and hasattr(mod, "type_text"):
            mod.type_text(text)
            return True
    except Exception:
        pass
    try:
        pygui.typewrite(text, interval=0.02)
        return True
    except Exception:
        return False


def mcp_hotkey(*keys: str) -> bool:
    mod = _mcp_import()
    try:
        if mod and hasattr(mod, "press_hotkey"):
            mod.press_hotkey(*keys)
            return True
    except Exception:
        pass
    try:
        pygui.hotkey(*keys)
        return True
    except Exception:
        return False


def mcp_key(key: str) -> bool:
    mod = _mcp_import()
    try:
        if mod and hasattr(mod, "press_key"):
            mod.press_key(key)
            return True
    except Exception:
        pass
    try:
        pygui.press(key)
        return True
    except Exception:
        return False


def mcp_scroll(amount: int) -> bool:
    try:
        pygui.scroll(int(amount))
        return True
    except Exception:
        return False


# ------------- Image helpers -------------

def _encode_image(path: str, fmt: str = "jpeg", quality: int = 80, resize_width: Optional[int] = 1280) -> str:
    fmt_l = (fmt or "jpeg").lower()
    if fmt_l not in ("png", "jpeg", "jpg", "webp"):
        fmt_l = "jpeg"
    img = Image.open(path).convert("RGB")
    if resize_width and img.width > resize_width:
        ratio = resize_width / float(img.width)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    save_kwargs = {}
    if fmt_l in ("jpeg", "jpg", "webp"):
        save_kwargs["quality"] = max(1, min(100, int(quality)))
        if fmt_l in ("jpeg", "jpg"):
            fmt_l = "JPEG"
        elif fmt_l == "webp":
            fmt_l = "WEBP"
    else:
        fmt_l = "PNG"
    img.save(buf, format=fmt_l, **save_kwargs)
    import base64
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_annotated_crop2x(screenshot_path: str, element_map: Dict[int, Dict]) -> Tuple[Optional[str], Optional[str]]:
    try:
        if not element_map:
            return None, None
        xs, ys, x2s, y2s = [], [], [], []
        for meta in element_map.values():
            x, y, w, h = meta["bbox"]
            xs.append(x); ys.append(y); x2s.append(x + w); y2s.append(y + h)
        x1 = max(0, min(xs) - 20); y1 = max(0, min(ys) - 20)
        x2 = max(x2s) + 20;        y2 = max(y2s) + 20
        full = Image.open(screenshot_path)
        W, H = full.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        crop = full.crop((x1, y1, x2, y2))
        crop_up = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        crop_path = screenshot_path.replace('.png', '_crop2x.png')
        crop_up.save(crop_path)
        annot = crop_up.copy()
        draw = ImageDraw.Draw(annot)
        for idx in sorted(element_map.keys()):
            bx, by, bw, bh = element_map[idx]["bbox"]
            cx, cy = element_map[idx]["center"]
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            tx = int((bx - x1) * 2); ty = int((by - y1) * 2)
            tw = int(bw * 2);       th = int(bh * 2)
            box_color = (0, 255, 0)
            thickness = max(2, min(tw, th) // 30)
            draw.rectangle([tx, ty, tx + tw, ty + th], outline=box_color, width=thickness)
            label = str(idx)
            min_dim = max(1, min(tw, th))
            font_size = max(16, min(64, int(min_dim * 0.45)))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
            try:
                bb = draw.textbbox((0, 0), label, font=font)
                twd = bb[2] - bb[0]
                thd = bb[3] - bb[1]
            except Exception:
                twd = len(label) * (font_size // 2)
                thd = font_size
            padding = max(3, font_size // 8)
            lx = tx + thickness + 2; ly = ty + thickness + 2
            draw.rectangle([lx, ly, lx + twd + padding * 2, ly + thd + padding * 2],
                           fill=(255, 255, 255), outline=(0, 0, 0), width=max(1, thickness // 2))
            draw.text((lx + padding, ly + padding), label, fill=(0, 0, 0), font=font)
        crop_bbox_path = screenshot_path.replace('.png', '_crop2x_bbox.png')
        annot.save(crop_bbox_path)
        return crop_path, crop_bbox_path
    except Exception:
        return None, None


def _limit_for_prompt(element_map: Dict[int, Dict], max_items: int = 100) -> Dict[int, Dict]:
    """Limit elements for the prompt/catalog only to reduce noise.
    Rank by area, presence of OCR text, and a simple centrality tie-breaker.
    Execution continues to use the full element_map.
    """
    if not element_map:
        return element_map
    ranked: List[Tuple[int, int, int, int]] = []
    # area, has_text, -centrality
    for idx, meta in element_map.items():
        x, y, w, h = meta.get("bbox", (0, 0, 0, 0))
        area = int(w) * int(h)
        txt = (meta.get("text") or "").strip()
        cx, cy = meta.get("center", (x + w // 2, y + h // 2))
        # approximate centrality (smaller better). We negate for sorting desc.
        centrality = -(abs(int(cx)) + abs(int(cy)))
        ranked.append((idx, area, 1 if txt else 0, centrality))
    ranked.sort(key=lambda t: (-t[1], -t[2], -t[3]))
    keep = {idx: element_map[idx] for idx, *_ in ranked[:max_items]}
    return keep


def _build_catalog_lines(element_map: Dict[int, Dict]) -> str:
    # Limit catalog for the prompt only; execution still uses full map
    em = _limit_for_prompt(element_map, max_items=80)
    if not em:
        return "(no catalog)"
    lines: List[str] = []
    for idx in sorted(em.keys()):
        meta = em[idx]
        x, y, w, h = meta.get("bbox", (0, 0, 0, 0))
        region = meta.get("region", "?")
        color = meta.get("color", "?")
        txt = (meta.get("text") or "").replace("\n", " ").strip()
        if len(txt) > 80:
            txt = txt[:77] + "..."
        lines.append(f"- Box {idx}: region={region}, color={color}, size={w}x{h}, text=\"{txt}\"")
    return "\n".join(lines)


# ------------- Prompt builder -------------

def _derive_state(element_map: Dict[int, Dict]) -> Dict[str, Any]:
    """Derive generic UI state from OCR: counts and sample texts; avoid task-specific overfitting."""
    texts: List[str] = []
    for meta in element_map.values():
        t = (meta.get("text") or "").strip()
        if t:
            texts.append(t[:60])
    return {
        "element_count": len(element_map),
        "sample_texts": texts[:8],  # include a few short snippets to ground context
    }


def build_agent_prompt(goal: str, element_map: Dict[int, Dict], history: List[Dict[str, Any]],
                       observations: Optional[Dict[str, Any]] = None,
                       memory: Optional[Dict[str, Any]] = None) -> str:
    catalog = _build_catalog_lines(element_map)
    # Build a compact continuity context: last 3 steps with action, ok, params, result, and reason
    recent_items = []
    for h in history[-3:]:
        s = h.get('step', '?'); a = h.get('action'); ok = h.get('ok'); p = h.get('params'); r = h.get('result'); rsn = h.get('reason')
        recent_items.append(f"[step {s}] action={a} ok={ok} params={p} result={r} reason={rsn}")
    recent = "\n".join(recent_items) or "(none)"
    small_hint = (
        "Tip: If you cannot find the target in the current window, you may minimize the active window and look elsewhere.\n"
    )
    obs_lines: List[str] = []
    if observations:
        if observations.get("element_count") is not None:
            obs_lines.append(f"Observed element count: {observations['element_count']}")
        if observations.get("sample_texts"):
            st = "; ".join([str(s) for s in observations["sample_texts"]])
            obs_lines.append(f"Sample texts: {st}")
        if observations.get("last_clicked_box") is not None:
            obs_lines.append(f"Last clicked box: {observations['last_clicked_box']}")
        if observations.get("last_action_ok") is not None:
            obs_lines.append(f"Last action ok: {observations['last_action_ok']}")
        if observations.get("last_action") is not None:
            obs_lines.append(f"Last action: {observations['last_action']}")
        if observations.get("last_params") is not None:
            obs_lines.append(f"Last params: {observations['last_params']}")
        if observations.get("last_result") is not None:
            obs_lines.append(f"Last result: {observations['last_result']}")
        if observations.get("last_reason") is not None:
            obs_lines.append(f"Last reason: {observations['last_reason']}")
    mem_lines: List[str] = []
    if memory:
        # include a compact KV list
        for k, v in list(memory.items())[:10]:
            mem_lines.append(f"{k}={v}")
    observations_block = ("Observed state:\n" + "\n".join(obs_lines) + "\n\n") if obs_lines else ""
    memory_block = ("Working memory:\n" + "\n".join(mem_lines) + "\n\n") if mem_lines else ""
    schema = (
        "Return STRICT JSON with fields: action, params, and optional reason.\n"
        "Allowed actions and params:\n"
        "- click: {\"box_number\": int, \"button\": \"left|right\", \"clicks\": 1|2}\n"
        "- move: {\"box_number\": int}\n"
        "- type_text: {\"text\": string}\n"
        "- key: {\"key\": string}  # e.g., 'enter','esc','tab','up','down','left','right','backspace','delete','home','end'\n"
        "- hotkey: {\"keys\": [string,...]}  # e.g., ['ctrl','l']\n"
        "- scroll: {\"amount\": int}  # positive=up, negative=down\n"
        "- wait: {\"seconds\": float}\n"
        "- minimize_active: {}  # Minimize current window\n"
    "- remember: {\"key\": string, \"value\": string}  # Persist a small fact for future steps\n"
    "- forget: {\"key\": string}  # Remove a key from memory\n"
    "- done: {\"message\": string}\n"
        "Rules:\n"
        "1) CONTINUE FROM PREVIOUS STEP: Assume the last action has just occurred; choose the next logical action.\n"
        "2) Prefer clicking/moving on provided box_number(s) from the catalog; pick the exact number.\n"
        "3) Avoid repeating the exact same click on the same box_number in consecutive steps unless explicitly necessary.\n"
        "4) If text input is needed, first click the input box, then use type_text on a subsequent step.\n"
        "5) Only one action per step. Keep actions small and purposeful.\n"
        "6) If the goal is fully satisfied, output action='done' with a concise message.\n"
        "7) If last_action_ok=false, adjust the plan: choose a different element or approach rather than repeating the same failed action.\n"
        "8) Before type_text, ensure focus is on the intended input (click it first if needed, based on the last action).\n"
        "9) Use only box_number values that appear in the catalog above; do not invent numbers.\n"
    )
    instructions = (
        f"Goal: {goal}\n\n"
        f"Catalog of visible elements (use these box numbers):\n{catalog}\n\n"
        f"Previous steps (for continuity):\n{recent}\n\n"
        f"{observations_block}"
        f"{small_hint}"
        f"{memory_block}"
        f"{schema}\n"
        f"Respond with JSON only.\n"
    )
    return instructions


# ------------- Perception -------------

def take_screenshot(path: str) -> str:
    pygui.screenshot(path)
    return path


def perceive() -> Tuple[str, str, Dict[int, Dict], Optional[str]]:
    shot = "agent_shot.png"
    take_screenshot(shot)
    elements = detect_ui_elements(shot, min_area=200, min_width=20, min_height=20)
    bbox_path, element_map = create_bbox_overlay(shot, elements)
    crop_path, crop_bbox_path = _build_annotated_crop2x(shot, element_map)
    return bbox_path, shot, element_map, crop_bbox_path or crop_path


# ------------- Action execution -------------

def execute_action(action: str, params: Dict[str, Any], element_map: Dict[int, Dict], move_duration: float = 0.2) -> Tuple[bool, str]:
    try:
        a = (action or "").lower().strip()
        if a in ("click", "double_click", "right_click", "move"):
            if "box_number" not in params:
                return False, "Missing box_number"
            bn = params.get("box_number")
            if isinstance(bn, str) and bn.isdigit():
                bn = int(bn)
            if not isinstance(bn, int) or bn not in element_map:
                return False, f"Invalid box_number: {bn}"
            cx, cy = element_map[bn]["center"]
            if a == "move":
                ok = mcp_move(cx, cy, duration=move_duration)
                return ok, f"move to {bn} -> {cx},{cy}"
            clicks = 1
            button = "left"
            if a == "double_click":
                clicks = 2
            elif a == "right_click":
                button = "right"
            ok1 = mcp_move(cx, cy, duration=move_duration)
            ok2 = mcp_click(button=button, clicks=clicks)
            return ok1 and ok2, f"{a} on {bn} -> {cx},{cy}"

        if a == "type_text":
            text = str(params.get("text", ""))
            ok = mcp_type(text)
            return ok, f"type_text: {text!r}"

        if a == "key":
            key = str(params.get("key", ""))
            ok = mcp_key(key)
            return ok, f"key: {key}"

        if a == "hotkey":
            keys = params.get("keys")
            if not isinstance(keys, list) or not keys:
                return False, "hotkey.keys must be a non-empty list"
            ok = mcp_hotkey(*[str(k) for k in keys])
            return ok, f"hotkey: {keys}"

        if a == "scroll":
            amount = int(params.get("amount", 0))
            ok = mcp_scroll(amount)
            return ok, f"scroll: {amount}"

        if a == "wait":
            sec = float(params.get("seconds", 0.3))
            time.sleep(max(0.0, sec))
            return True, f"waited {sec}s"

        if a == "remember":
            # Memory is applied by caller; here just acknowledge
            k = params.get("key"); v = params.get("value")
            return True, f"remember: {k}={v}"

        if a == "forget":
            k = params.get("key")
            return True, f"forget: {k}"

        if a == "minimize_active":
            # Try Win+Down (twice often minimizes), fallback Alt+Space then 'n'
            ok = mcp_hotkey("win", "down")
            time.sleep(0.05)
            ok2 = mcp_hotkey("win", "down")
            if not (ok and ok2):
                ok = mcp_hotkey("alt", "space") and mcp_key("n")
            return ok, "minimize_active"

        if a == "done":
            msg = str(params.get("message", "done"))
            return True, f"done: {msg}"

        return False, f"Unknown action: {action}"
    except Exception as e:
        return False, f"Exception: {e}"


# ------------- Reasoning call -------------

def call_model(client: OpenAI, model: str, prompt: str, images: List[Tuple[str, str]], max_tokens: int = 128) -> Dict[str, Any]:
    # images: list of (mime, base64)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for mime, b64 in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    # Attempt 1: enforce JSON with response_format
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful UI agent. Respond with STRICT JSON only."},
                {"role": "user", "content": content},
            ],
            temperature=0,
            top_p=0.1,
            max_tokens=max(64, min(256, int(max_tokens))),
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            raise ValueError("empty_content")
        return json.loads(raw)
    except Exception as e1:
        # Fallback attempt 2: no response_format, stronger instruction
        try:
            resp2 = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful UI agent. Output ONLY valid JSON. No prose."},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                top_p=0.1,
                max_tokens=max(64, min(256, int(max_tokens))),
            )
            raw2 = (resp2.choices[0].message.content or "").strip()
            # Try direct JSON parse
            try:
                return json.loads(raw2)
            except Exception:
                # Weak regex extraction of first JSON object
                m = re.search(r"\{[\s\S]*\}", raw2)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass
            return {"_error": f"model_or_parse_error: {e1}", "_raw": (raw2 or "")[:800]}
        except Exception as e2:
            # Return structured error with partial raw if available from first attempt
            try:
                raw = raw  # type: ignore[name-defined]
            except Exception:
                raw = ""
            return {"_error": f"model_or_parse_error: {e1}; fallback_error: {e2}", "_raw": raw[:800]}


# ------------- Agent loop -------------

def run_agent(goal: str, api_key: Optional[str], model: str, max_steps: int, move_duration: float, verbose: bool,
              image_format: str = "jpeg", quality: int = 80, resize_width: int = 1280, send_crop: bool = True) -> int:
    _load_env_if_needed()
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set.", file=sys.stderr)
        return 2
    client = OpenAI(api_key=key)

    history: List[Dict[str, Any]] = []
    last_clicked_box: Optional[int] = None
    working_memory: Dict[str, Any] = {}
    for step in range(1, max(1, max_steps) + 1):
        bbox_path, shot_path, element_map, crop_path = perceive()
        images: List[Tuple[str, str]] = [(f"image/{image_format}", _encode_image(bbox_path, fmt=image_format, quality=quality, resize_width=resize_width))]
        if send_crop and crop_path and os.path.exists(crop_path):
            images.append((f"image/{image_format}", _encode_image(crop_path, fmt=image_format, quality=quality, resize_width=resize_width)))
        # Observations for continuity (score + start_clicked)
        derived = _derive_state(element_map)
        derived["last_clicked_box"] = last_clicked_box
        if history:
            derived["last_action_ok"] = bool(history[-1].get("ok"))
            derived["last_action"] = history[-1].get("action")
            derived["last_params"] = history[-1].get("params")
            derived["last_result"] = history[-1].get("result")
            derived["last_reason"] = history[-1].get("reason")

        # Ask model; retry within the step on parse/invalid actions
        data: Dict[str, Any] = {}
        allowed_actions = {"click","move","type_text","key","hotkey","scroll","wait","minimize_active","remember","forget","done"}
        for retry in range(3):
            prompt = build_agent_prompt(goal, element_map, history, observations=derived, memory=working_memory)
            data = call_model(client, model, prompt, images)
            if verbose:
                print(f"\nStep {step} retry {retry} model JSON:\n{json.dumps(data, indent=2)[:1200]}")
            if "_error" in data:
                history.append({"action": "error", "result": data.get("_error"), "ok": False})
                time.sleep(0.3)
                continue
            a = (data.get("action") or "").lower().strip()
            p = data.get("params") if isinstance(data.get("params"), dict) else {}
            # Duplicate click guard
            if a == "click" and isinstance(p.get("box_number"), (int, str)):
                bn = int(p["box_number"]) if isinstance(p["box_number"], str) and p["box_number"].isdigit() else p["box_number"]
                if isinstance(bn, int) and last_clicked_box is not None and bn == last_clicked_box:
                    history.append({"action": "info", "result": f"duplicate click on same box {bn} prevented; re-planning", "ok": False})
                    time.sleep(0.2)
                    continue
            # If invalid or empty action, re-ask this step
            if not a or a not in allowed_actions:
                history.append({"action": "error", "result": f"invalid action '{a}'", "ok": False})
                time.sleep(0.2)
                continue
            # Good to proceed
            break
        else:
            # All retries failed → do not execute None; wait and re-observe next step
            if verbose:
                print("No valid action after retries; waiting 1.0s and re-observing.", file=sys.stderr)
            time.sleep(1.0)
            continue

        # Execute the chosen action
        action = data.get("action") if isinstance(data, dict) else None
        params = data.get("params") if isinstance(data, dict) and isinstance(data.get("params"), dict) else {}
        reason = data.get("reason") if isinstance(data, dict) else None
        ok, result = execute_action(action, params, element_map, move_duration=move_duration)
        # Update last_clicked_box state
        if (action or "").lower() == "click" and isinstance(params.get("box_number"), (int, str)):
            bn = int(params["box_number"]) if isinstance(params["box_number"], str) and params["box_number"].isdigit() else params["box_number"]
            if isinstance(bn, int):
                last_clicked_box = bn
        # Update working memory on remember/forget actions
        if (action or "").lower() == "remember" and isinstance(params.get("key"), str):
            k = params.get("key"); v = str(params.get("value", ""))
            if k:
                working_memory[k] = v
        if (action or "").lower() == "forget" and isinstance(params.get("key"), str):
            k = params.get("key")
            if k in working_memory:
                working_memory.pop(k, None)
        history.append({"step": step, "action": action, "params": params, "ok": ok, "result": result, "reason": reason})
        if verbose:
            print(f"Executed: {action} -> {result} (ok={ok})")
        # If model says done or action was 'done', stop
        if (action or "").lower() == "done":
            if verbose:
                print("Agent reports done.")
            return 0
        # brief wait to let UI update before next perception
        time.sleep(0.6)
    if verbose:
        print("Reached max steps without 'done'.")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--goal", required=True, help="Natural language goal for the agent")
    p.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    p.add_argument("--api-key", dest="api_key", default=None)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--move-duration", type=float, default=0.2)
    p.add_argument("--verbose", action="store_true", default=False)
    # perf flags similar to cloud script
    p.add_argument("--image-format", default="jpeg")
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--resize-width", type=int, default=1280)
    p.add_argument("--no-crop", dest="no_crop", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    code = run_agent(
        goal=args.goal,
        api_key=args.api_key,
        model=args.model,
        max_steps=args.max_steps,
        move_duration=args.move_duration,
        verbose=args.verbose,
        image_format=args.image_format,
        quality=args.quality,
        resize_width=args.resize_width,
        send_crop=not args.no_crop,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
