import os
import time
import difflib
from typing import Optional, Dict

import pyautogui as pygui
from PIL import Image

# DRY imports
from bbox_detection import detect_ui_elements, create_bbox_overlay, test_bbox_detection


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


# ---------- MCP wrappers with safe fallback ----------

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


def mcp_move(x: int, y: int, duration: float = 0.3) -> bool:
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
            mod.click_mouse(button=button, clicks=clicks)
            return True
    except Exception:
        pass
    try:
        if clicks > 1:
            pygui.click(clicks=clicks, button=button)
        else:
            if button == "left":
                pygui.click()
            elif button == "right":
                pygui.rightClick()
            else:
                pygui.click(button=button)
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


# ---------- OCR-first selection ----------

def _choose_box_by_ocr(element_map: Dict[int, Dict], query: str) -> Optional[int]:
    q = (query or "").strip().lower()
    if not q:
        return None
    candidates = []
    for idx, meta in element_map.items():
        text = (meta.get("text") or "").lower()
        if not text:
            continue
        if q in text:
            candidates.append((idx, len(text), meta.get("area", 0)))
    if candidates:
        candidates.sort(key=lambda t: (t[1], -t[2]))
        return candidates[0][0]
    import difflib as _dif
    scored = []
    for idx, meta in element_map.items():
        text = (meta.get("text") or "").lower()
        if not text:
            continue
        score = _dif.SequenceMatcher(None, q, text).ratio()
        if score >= 0.55:
            scored.append((idx, score, meta.get("area", 0)))
    if scored:
        scored.sort(key=lambda t: (-t[1], -t[2]))
        return scored[0][0]
    return None


# ---------- Skills ----------

def ensure_desktop():
    mcp_hotkey("win", "d")
    time.sleep(0.6)


def open_directory(path: str) -> bool:
    if not path:
        return False
    # Win+E → Explorer
    mcp_hotkey("win", "e")
    time.sleep(0.8)
    # Ctrl+L → address bar, type path, Enter
    mcp_hotkey("ctrl", "l")
    time.sleep(0.2)
    mcp_type(path)
    time.sleep(0.05)
    mcp_key("enter")
    time.sleep(1.0)
    return True


def click_file_by_name(filename: str, double_click: bool = True, retries: int = 2) -> bool:
    if not filename:
        return False
    for _ in range(retries):
        shot = f"taskshot_{int(time.time()*1000)}.png"
        pygui.screenshot(shot)
        elements = detect_ui_elements(shot, min_area=200, min_width=20, min_height=20)
        _, element_map = create_bbox_overlay(shot, elements)
        # OCR fast path
        box_id = _choose_box_by_ocr(element_map, filename)
        if box_id:
            cx, cy = element_map[box_id]["center"]
            if mcp_move(cx, cy, duration=0.25):
                mcp_click(clicks=2 if double_click else 1)
                return True
        # VLM picker fallback
        desc = f'the file named "{filename}"'
        result = test_bbox_detection(desc, count_id=0, click=False, min_box_size=20)
        if result.get("success") and isinstance(result.get("box_number"), int):
            cx, cy = result["x"], result["y"]
            if mcp_move(cx, cy, duration=0.25):
                mcp_click(clicks=2 if double_click else 1)
                return True
        time.sleep(0.6)
    return False


def open_and_click(path: str, filename: str, ensure_show_desktop: bool = True) -> bool:
    _load_env_if_needed()
    if ensure_show_desktop:
        ensure_desktop()
        time.sleep(0.4)
    if not open_directory(path):
        return False
    return click_file_by_name(filename, double_click=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Directory path to open")
    p.add_argument("--name", required=True, help="Filename to open (display name)")
    p.add_argument("--no-desktop", action="store_true", help="Do not send Win+D first")
    args = p.parse_args()

    print("Starting task: open directory and click file...")
    success = open_and_click(args.path, args.name, ensure_show_desktop=not args.no_desktop)
    if success:
        print("Done.")
        raise SystemExit(0)
    else:
        print("Failed.")
        raise SystemExit(1)
