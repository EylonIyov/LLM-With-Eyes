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
- Allowed actions: click_box/double_click_box/right_click_box/hover_box, type_text, key, hotkey, scroll, wait, remember, forget, done
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


def _shorten_text(value: str, limit: int = 64) -> str:
    try:
        cleaned = re.sub(r"\s+", " ", value or "").strip()
    except Exception:
        cleaned = (value or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _format_memory(memory: Optional[Dict[str, Any]]) -> str:
    if not memory:
        return "(none)"
    lines: List[str] = []
    for k in sorted(memory.keys()):
        try:
            v = memory[k]
            if isinstance(v, (dict, list)):
                v_repr = json.dumps(v, ensure_ascii=True)
            else:
                v_repr = str(v)
        except Exception:
            v_repr = "<unserializable>"
        lines.append(f"{k} = {v_repr}")
        if len(lines) >= 12:
            lines.append("...")
            break
    return "\n".join(lines)


def _format_history(history: List[Dict[str, Any]], limit: int = 6) -> str:
    if not history:
        return "(no prior steps)"
    recent = history[-limit:]
    lines: List[str] = []
    for item in recent:
        step = item.get("step", "?")
        action = item.get("action") or "?"
        ok = item.get("ok")
        params = item.get("params")
        result = item.get("result")
        try:
            params_repr = json.dumps(params, ensure_ascii=True)
        except Exception:
            params_repr = str(params)
        line = f"[{step}] action={action} ok={ok} params={params_repr} result={_shorten_text(str(result), 72)}"
        if item.get("reason"):
            line += f" reason={_shorten_text(str(item.get('reason')), 48)}"
        if item.get("plan"):
            plan_val = item.get("plan")
            plan_repr = plan_val
            if isinstance(plan_val, (dict, list)):
                try:
                    plan_repr = json.dumps(plan_val, ensure_ascii=True)
                except Exception:
                    plan_repr = str(plan_val)
            line += f" plan={_shorten_text(str(plan_repr), 48)}"
        lines.append(line)
    return "\n".join(lines)


def _build_structured_observation(element_map: Dict[int, Dict], max_items: int = 60) -> str:
    if not element_map:
        return "(no elements detected)"
    limited = _limit_for_prompt(element_map, max_items=max_items)
    lines: List[str] = []
    for idx in sorted(limited.keys()):
        meta = element_map[idx]
        x, y, w, h = meta.get("bbox", (0, 0, 0, 0))
        region = meta.get("region", "?")
        color = meta.get("color", "?")
        text = _shorten_text(meta.get("text") or "")
        conf = meta.get("text_confidence")
        conf_str = f"{int(conf)}" if isinstance(conf, (int, float)) else "?"
        cx, cy = meta.get("center", (x + w // 2, y + h // 2))
        lines.append(
            f"Box {idx}: region={region} size={w}x{h} color={color} text=\"{text}\" conf={conf_str} center=({cx}, {cy})"
        )
    return "\n".join(lines)


def _format_observations(observations: Optional[Dict[str, Any]]) -> str:
    if not observations:
        return "(none)"
    lines: List[str] = []
    for key in sorted(observations.keys()):
        value = observations[key]
        if isinstance(value, (dict, list)):
            try:
                value_repr = json.dumps(value, ensure_ascii=True)
            except Exception:
                value_repr = str(value)
        else:
            value_repr = str(value)
        lines.append(f"{key}: {value_repr}")
        if len(lines) >= 12:
            lines.append("...")
            break
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


def _normalize_quotes(s: str) -> str:
    """Normalize curly quotes and backticks to standard quotes for easier parsing."""
    try:
        return (
            s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'")
             .replace("`", "'")
        )
    except Exception:
        return s


def _extract_typing_intent(goal: str) -> Tuple[Optional[str], bool]:
    """Extract a literal string to type from the goal, plus whether to press enter.

    Heuristics:
    - Look for quoted text after verbs like: type, search, enter, write, input, query
    - Fallback to the first quoted segment if verbs not found
    - Treat backticks and curly quotes as quotes
    - requires_enter: goal mentions 'press enter', 'hit enter', 'submit', or 'search'
    """
    if not goal:
        return None, False
    g = _normalize_quotes(goal)
    requires_enter = bool(re.search(r"\b(press|hit)\s+enter\b|\bsubmit\b|\bsearch\b", g, re.IGNORECASE))
    # Prefer quoted strings following typing verbs
    verb_pattern = r"(?:type|search|enter|write|input|query)\s*[\"']([^\"']+)[\"']"
    m = re.search(verb_pattern, g, flags=re.IGNORECASE)
    if m:
        text = m.group(1).strip()
        return (text if text else None), requires_enter
    # Fallback: first quoted block anywhere
    m2 = re.search(r"[\"']([^\"']+)[\"']", g)
    if m2:
        text = m2.group(1).strip()
        return (text if text else None), requires_enter
    return None, requires_enter


def build_agent_prompt(goal: str, element_map: Dict[int, Dict], history: List[Dict[str, Any]],
                       observations: Optional[Dict[str, Any]] = None,
                       memory: Optional[Dict[str, Any]] = None) -> str:
    structured_obs = _build_structured_observation(element_map, max_items=70)
    history_block = _format_history(history, limit=6)
    memory_block = _format_memory(memory)
    observation_block = _format_observations(observations)

    schema = (
        "Return STRICT JSON with keys: action, params, optional reason, optional plan.\n"
        "Allowed actions:\n"
        "- click_box: {\"box_id\": int}\n"
        "- double_click_box: {\"box_id\": int}\n"
        "- right_click_box: {\"box_id\": int}\n"
        "- hover_box: {\"box_id\": int}\n"
        "- type_text: {\"text\": string, \"submit\": bool (optional)}\n"
        "- key: {\"key\": string}\n"
        "- hotkey: {\"keys\": [string,...]}\n"
        "- scroll: {\"amount\": int}\n"
        "- wait: {\"seconds\": float}\n"
        "- remember: {\"key\": string, \"value\": string}\n"
        "- forget: {\"key\": string}\n"
        "- done: {\"message\": string}\n"
        "Rules:\n"
        "1) Use the numbered boxes from the observation when referring to UI elements.\n"
        "2) Click, double-click, right-click, or hover actions must specify a valid box_id.\n"
        "3) Focus an input before type_text; keep click and typing as separate steps.\n"
        "4) Choose a single, purposeful action per step; adapt if the previous action failed.\n"
        "5) Mark the task complete with action='done' once the goal is achieved.\n"
        "6) If text entry needs confirmation, set submit=true or plan a follow-up key press.\n"
    )

    instructions = (
        "You are a grounded computer interaction agent. Observe the annotated screenshot and decide the next GUI action.\n"
        f"GOAL: {goal}\n\n"
        "Observation (numbered boxes):\n"
        f"{structured_obs}\n\n"
        "State clues:\n"
        f"{observation_block}\n\n"
        "Recent trajectory:\n"
        f"{history_block}\n\n"
        "Working memory:\n"
        f"{memory_block}\n\n"
        f"{schema}\n"
        "Respond with JSON only."
    )
    return instructions


# Small heuristic: is a UI element likely an input field?
def _is_input_candidate(meta: Dict[str, Any]) -> bool:
    try:
        x, y, w, h = meta.get("bbox", (0, 0, 0, 0))
        text = (meta.get("text") or "").lower()
        # Heuristics: fairly wide and not too tall; or text hints
        wide_flat = (w >= 200 and 18 <= h <= 90) or (w >= 320 and h <= 120)
        text_hint = any(k in text for k in ("search", "type", "enter", "query", "find"))
        return bool(wide_flat or text_hint)
    except Exception:
        return False


# ------------- Perception -------------

def take_screenshot(path: str) -> str:
    pygui.screenshot(path)
    return path


def perceive() -> Tuple[str, Optional[str], Dict[int, Dict]]:
    shot_path = "agent_shot.png"
    take_screenshot(shot_path)

    try:
        elements = detect_ui_elements(shot_path, min_area=200, min_width=20, min_height=20)
    except Exception:
        elements = []

    try:
        annotated_path, element_map = create_bbox_overlay(shot_path, elements)
    except Exception:
        annotated_path = None
        # Fallback to minimal metadata if overlay creation fails
        element_list = []
        for idx, bbox in enumerate(elements, start=1):
            x, y, w, h = bbox
            element_list.append(
                (
                    idx,
                    {
                        "bbox": (x, y, w, h),
                        "center": (x + w // 2, y + h // 2),
                        "area": w * h,
                        "text": "",
                        "color": "unknown",
                        "region": "unknown",
                    },
                )
            )
        element_map = {idx: meta for idx, meta in element_list}

    return shot_path, annotated_path, element_map


def find_and_click(client: OpenAI, model: str, description: str, screenshot_path: str, action_type: str, move_duration: float) -> Tuple[bool, str]:
    """
    Use the VLM to find an element by description and perform a click action.
    """
    try:
        # Encode the screenshot
        b64_image = _encode_image(screenshot_path, fmt="jpeg", quality=90, resize_width=1400)
        
        # Build the prompt for the VLM to find the coordinates
        prompt = (
            f"You are a coordinate finder. Based on the provided screenshot, identify the center coordinates (x, y) "
            f"of the element best described as: \"{description}\".\n"
            "Respond with STRICT JSON containing only 'x' and 'y' keys, like {\"x\": 123, \"y\": 456}.\n"
            "If the element is not visible, respond with {\"error\": \"not found\"}."
        )
        
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
        ]
        
        # Call the model to get coordinates
        coord_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        
        coord_data_raw = coord_resp.choices[0].message.content or "{}"
        coord_data = json.loads(coord_data_raw)
        
        if "error" in coord_data or "x" not in coord_data or "y" not in coord_data:
            return False, f"Element '{description}' not found by VLM."
            
        x, y = int(coord_data["x"]), int(coord_data["y"])

        # Execute the click action at the found coordinates
        if action_type == "move":
            ok = mcp_move(x, y, duration=move_duration)
            return ok, f"move to '{description}' -> {x},{y}"
        
        clicks = 1
        button = "left"
        if action_type == "double_click":
            clicks = 2
        elif action_type == "right_click":
            button = "right"
            
        ok1 = mcp_move(x, y, duration=move_duration)
        ok2 = mcp_click(button=button, clicks=clicks)
        
        return ok1 and ok2, f"{action_type} on '{description}' -> {x},{y}"

    except Exception as e:
        return False, f"Exception in find_and_click: {e}"


# ------------- Action execution -------------

def execute_action(
    client: OpenAI,
    model: str,
    action: str,
    params: Dict[str, Any],
    screenshot_path: str,
    element_map: Dict[int, Dict],
    move_duration: float = 0.2,
    working_memory: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    try:
        a = (action or "").lower().strip()

        if working_memory is None:
            working_memory = {}

        if a in ("click_box", "double_click_box", "right_click_box", "hover_box"):
            try:
                box_id = int(params.get("box_id"))
            except Exception:
                return False, "Missing or invalid 'box_id' for box-based action"

            meta = element_map.get(box_id)
            if not meta:
                return False, f"Unknown box_id {box_id}"
            x, y, w, h = meta.get("bbox", (0, 0, 0, 0))
            cx, cy = meta.get("center", (x + w // 2, y + h // 2))

            ok_move = mcp_move(int(cx), int(cy), duration=move_duration)
            if a == "hover_box":
                return ok_move, f"hover_box #{box_id} -> ({cx}, {cy})"

            button = "left"
            clicks = 1
            if a == "double_click_box":
                clicks = 2
            elif a == "right_click_box":
                button = "right"

            ok_click = mcp_click(button=button, clicks=clicks)
            return ok_move and ok_click, f"{a} #{box_id} -> ({cx}, {cy})"
        
        if a in ("click", "double_click", "right_click", "move"):
            description = params.get("description")
            if not description:
                return False, "Missing 'description' for click action"
            return find_and_click(client, model, description, screenshot_path, a, move_duration)

        if a == "type_text":
            raw_text = params.get("text")
            text = str(raw_text) if raw_text is not None else ""
            if not text:
                fallback = working_memory.get("goal_literal_text")
                if fallback:
                    text = str(fallback)
            if not text:
                return False, "type_text requires 'text'"

            ok = mcp_type(text)
            should_submit: Optional[bool]
            submit_param = params.get("submit")
            if isinstance(submit_param, bool):
                should_submit = submit_param
            elif isinstance(submit_param, str):
                should_submit = submit_param.lower() in {"true", "1", "yes"}
            else:
                should_submit = None

            extra = ""
            if ok:
                needs_enter = False
                if should_submit is True:
                    needs_enter = True
                elif should_submit is False:
                    needs_enter = False
                elif working_memory.get("goal_requires_enter"):
                    needs_enter = True
                if needs_enter:
                    enter_ok = mcp_key("enter")
                    ok = ok and enter_ok
                    extra = " + enter"
                    working_memory["goal_requires_enter"] = False
            return ok, f"type_text: {text!r}{extra}"

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
            k = params.get("key"); v = params.get("value")
            return True, f"remember: {k}={v}"

        if a == "forget":
            k = params.get("key")
            return True, f"forget: {k}"

        if a == "minimize_active":
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
    working_memory: Dict[str, Any] = {}

    literal_text, literal_submit = _extract_typing_intent(goal)
    if literal_text:
        working_memory["goal_literal_text"] = literal_text
    if literal_submit:
        working_memory["goal_requires_enter"] = True
    working_memory["goal_summary"] = _shorten_text(goal, 96)

    for step in range(1, max(1, max_steps) + 1):
        shot_path, annotated_path, element_map = perceive()

        images: List[Tuple[str, str]] = [
            (f"image/{image_format}", _encode_image(shot_path, fmt=image_format, quality=quality, resize_width=resize_width))
        ]
        if annotated_path:
            images.append(("image/png", _encode_image(annotated_path, fmt="png", quality=quality, resize_width=resize_width)))

        if send_crop and element_map:
            try:
                top_subset = _limit_for_prompt(element_map, max_items=10)
                crop_map = {idx: element_map[idx] for idx in top_subset}
                if crop_map:
                    _, crop_bbox_path = _build_annotated_crop2x(shot_path, crop_map)
                    if crop_bbox_path:
                        images.append(("image/png", _encode_image(crop_bbox_path, fmt="png", quality=quality, resize_width=min(resize_width, 1400))))
            except Exception:
                pass

        derived = _derive_state(element_map)
        derived["structured_preview"] = _build_structured_observation(element_map, max_items=24)
        if history:
            derived["last_action_ok"] = bool(history[-1].get("ok"))
            derived["last_action"] = history[-1].get("action")
            derived["last_params"] = history[-1].get("params")
            derived["last_result"] = history[-1].get("result")
            derived["last_reason"] = history[-1].get("reason")
            if history[-1].get("plan") is not None:
                derived["last_plan"] = history[-1].get("plan")

        # Ask model
        data: Dict[str, Any] = {}
        allowed_actions = {
            "click_box",
            "double_click_box",
            "right_click_box",
            "hover_box",
            "click",
            "double_click",
            "right_click",
            "move",
            "type_text",
            "key",
            "hotkey",
            "scroll",
            "wait",
            "minimize_active",
            "remember",
            "forget",
            "done",
        }
        
        prompt = build_agent_prompt(goal, element_map, history, observations=derived, memory=working_memory)
        data = call_model(client, model, prompt, images, max_tokens=192)
        
        if verbose:
            print(f"\nStep {step} model JSON:\n{json.dumps(data, indent=2)[:1200]}")
        
        if "_error" in data:
            history.append({"action": "error", "result": data.get("_error"), "ok": False})
            time.sleep(0.3)
            continue
            
        a = (data.get("action") or "").lower().strip()
        if not a or a not in allowed_actions:
            history.append({"action": "error", "result": f"invalid action '{a}'", "ok": False})
            time.sleep(0.2)
            continue

        # Execute the chosen action
        action = data.get("action") if isinstance(data, dict) else None
        params = data.get("params") if isinstance(data, dict) and isinstance(data.get("params"), dict) else {}
        reason = data.get("reason") if isinstance(data, dict) else None
        ok, result = execute_action(
            client,
            model,
            action,
            params,
            shot_path,
            element_map,
            move_duration=move_duration,
            working_memory=working_memory,
        )

        # Update working memory on remember/forget actions
        if (action or "").lower() == "remember" and isinstance(params.get("key"), str):
            k = params.get("key"); v = str(params.get("value", ""))
            if k:
                working_memory[k] = v
        if (action or "").lower() == "forget" and isinstance(params.get("key"), str):
            k = params.get("key")
            if k in working_memory:
                working_memory.pop(k, None)
        history.append(
            {
                "step": step,
                "action": action,
                "params": params,
                "ok": ok,
                "result": result,
                "reason": reason,
                "plan": data.get("plan") if isinstance(data, dict) else None,
            }
        )
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
