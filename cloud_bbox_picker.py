"""
Cloud BBox Picker
- Reuses existing detection/overlay/prompt utils from bbox_detection.py
- Reuses base64 encoder from script.py
- Captures a screenshot, detects UI elements, draws numbered boxes, builds a
    2x annotated crop, then sends both images plus a strict prompt to an
    OpenAI vision model to get the chosen box number.

CLI highlights:
- Prints ONLY the chosen number to stdout by default (quiet mode)
- Moves the mouse to that bbox center (and optionally clicks)

Usage (from repo root):
    1) Activate venv
         .\.venv\Scripts\Activate.ps1
    2) Install dependency if needed: pip install openai
    3) Run (quiet, prints just the number to stdout):
         python .\cloud_bbox_picker.py --target "the green box"
         # optionally click:
         python .\cloud_bbox_picker.py --target "the green box" --click

Flags:
    --api-key <KEY>        OpenAI API key (fallback: env OPENAI_API_KEY; otherwise prompt)
    --model <MODEL>        Model name (default: gpt-4o)
    --target <TEXT>        Target description
    --no-move              Do not move mouse (default is to move)
    --click                Click after moving the mouse
    --move-duration <S>    Mouse move duration in seconds (default: 0.2)
    --verbose              Verbose logs to stdout (default is quiet: only number to stdout)

This script intentionally uses DRY by importing helpers from bbox_detection.py
and script.py instead of re-implementing them.
"""

from __future__ import annotations
import os
import sys
import base64
import json
import re
import argparse
from io import BytesIO
from typing import Dict, Tuple, Optional, List

# Third-party
try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: openai. Install with 'pip install openai'.", file=sys.stderr)
    sys.exit(1)

import pyautogui as pygui
from PIL import Image, ImageDraw, ImageFont

# Local imports (DRY)
from script import encode_image_to_base64  # already implemented
from bbox_detection import (
    detect_ui_elements,
    create_bbox_overlay,
    create_bbox_prompt,
)


def take_screenshot(path: str) -> str:
    pygui.screenshot(path)
    return path


def _encode_image(path: str, fmt: str = "png", quality: int = 85, resize_width: Optional[int] = None) -> str:
    """Load, optionally resize, and encode image to base64 in requested format.
    fmt: png|jpeg|webp (case-insensitive)
    quality: 1-100 for lossy formats
    resize_width: if provided, scales width and keeps aspect ratio
    Returns raw base64 string (no data URI prefix)
    """
    fmt_l = fmt.lower()
    if fmt_l not in ("png", "jpeg", "jpg", "webp"):
        fmt_l = "png"
    img = Image.open(path).convert("RGB")
    if resize_width and img.width > resize_width:
        ratio = resize_width / float(img.width)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    save_kwargs = {}
    if fmt_l in ("jpeg", "jpg", "webp"):
        save_kwargs["quality"] = max(1, min(100, int(quality)))
        if fmt_l == "jpeg" or fmt_l == "jpg":
            fmt_l = "JPEG"
        elif fmt_l == "webp":
            fmt_l = "WEBP"
    else:
        fmt_l = "PNG"
    img.save(buf, format=fmt_l, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_annotated_crop2x(screenshot_path: str, element_map: Dict[int, Dict]) -> Tuple[Optional[str], Optional[str]]:
    """Create a 2x crop around the union of all boxes, and also an annotated variant
    that keeps the same numbering. Returns (crop_path, crop_path_bbox).
    """
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

        # Annotated crop with same numbers
        annot = crop_up.copy()
        draw = ImageDraw.Draw(annot)
        for idx in sorted(element_map.keys()):
            bx, by, bw, bh = element_map[idx]["bbox"]
            cx, cy = element_map[idx]["center"]
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            tx = int((bx - x1) * 2); ty = int((by - y1) * 2)
            tw = int(bw * 2);       th = int(bh * 2)

            # Draw box
            box_color = (0, 255, 0)
            thickness = max(2, min(tw, th) // 30)
            draw.rectangle([tx, ty, tx + tw, ty + th], outline=box_color, width=thickness)

            # Label number (match bbox_detection style: white bg + black border)
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


def _build_catalog_lines(element_map: Dict[int, Dict]) -> str:
    if not element_map:
        return "(no catalog available)"
    lines: List[str] = []
    for idx in sorted(element_map.keys()):
        meta = element_map[idx]
        x, y, w, h = meta.get("bbox", (0, 0, 0, 0))
        color = meta.get("color", "?")
        region = meta.get("region", "?")
        text = meta.get("text", "") or ""
        txt = text.replace("\n", " ").strip()
        if len(txt) > 60:
            txt = txt[:57] + "..."
        lines.append(f"- Box {idx}: region={region}, color={color}, size={w}x{h}, text=\"{txt}\"")
    return "\n".join(lines[:80])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--api-key", dest="api_key", default=None)
    p.add_argument("--model", dest="model", default="gpt-4o")
    p.add_argument("--target", dest="target", default=None)
    p.add_argument("--no-move", dest="move", action="store_false", default=True)
    p.add_argument("--click", dest="click", action="store_true", default=False)
    p.add_argument("--move-duration", dest="move_duration", type=float, default=0.2)
    p.add_argument("--verbose", dest="verbose", action="store_true", default=False)
    # Performance tuning
    p.add_argument("--image-format", dest="image_format", default=os.environ.get("CBP_IMAGE_FORMAT", "jpeg"))
    p.add_argument("--quality", dest="quality", type=int, default=int(os.environ.get("CBP_QUALITY", "80")))
    p.add_argument("--resize-width", dest="resize_width", type=int, default=int(os.environ.get("CBP_RESIZE_WIDTH", "1280")))
    p.add_argument("--no-crop", dest="no_crop", action="store_true", default=False)
    p.add_argument("--crop-only", dest="crop_only", action="store_true", default=False)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=int(os.environ.get("CBP_MAX_TOKENS", "96")))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    # Load .env if OPENAI_API_KEY not set (minimal loader to avoid extra deps)
    if not os.environ.get("OPENAI_API_KEY"):
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        try:
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            k, v = line.split('=', 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            # Set only if not already present
                            if k and (k not in os.environ):
                                os.environ[k] = v
        except Exception:
            pass

    # Logging helpers: stdout for number only (quiet default). Route extra logs to stderr.
    def log(msg: str):
        if args.verbose:
            print(msg)
        else:
            # quiet mode -> logs to stderr
            print(msg, file=sys.stderr)

    # Key & model
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if args.verbose:
            api_key = input("OpenAI API key: ").strip()
        else:
            print("OPENAI_API_KEY not provided.", file=sys.stderr)
            print(-1)
            sys.exit(2)
    model = args.model or "gpt-4o"

    # Target
    target = args.target or (input("Describe the target (e.g., 'the green box'): ").strip() if args.verbose else None)
    if not target:
        print("No target given.", file=sys.stderr)
        print(-1)
        sys.exit(2)

    # 1) Screenshot
    
    screenshot = "cloud_screenshot.png"
    take_screenshot(screenshot)
    log(f"📸 Screenshot saved: {screenshot}")

    # 2) Detect & overlay using existing helpers
    elements = detect_ui_elements(screenshot, min_area=200, min_width=20, min_height=20)
    bbox_path, element_map = create_bbox_overlay(screenshot, elements)
    log(f"🖼️  Annotated saved: {bbox_path}")

    # 3) Build crop + annotated crop
    crop_path, crop_bbox_path = _build_annotated_crop2x(screenshot, element_map)

    # 4) Build prompt — reuse existing create_bbox_prompt but we also build a compact catalog string
    #    The enhanced create_bbox_prompt already includes catalog internally, so we can use it directly.
    prompt = create_bbox_prompt(target, len(element_map), element_map)

    # 5) Send to OpenAI
    client = OpenAI(api_key=api_key)
    content = [{"type": "text", "text": prompt}]
    # Assemble images based on performance flags
    fmt = args.image_format
    q = args.quality
    rw = args.resize_width if args.resize_width and args.resize_width > 0 else None
    send_full = not args.crop_only
    send_crop = not args.no_crop
    if send_full:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{fmt};base64,{_encode_image(bbox_path, fmt=fmt, quality=q, resize_width=rw)}"}
        })
    if send_crop:
        if crop_bbox_path and os.path.exists(crop_bbox_path):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{fmt};base64,{_encode_image(crop_bbox_path, fmt=fmt, quality=q, resize_width=rw)}"}
            })
        elif crop_path and os.path.exists(crop_path):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{fmt};base64,{_encode_image(crop_path, fmt=fmt, quality=q, resize_width=rw)}"}
            })

    log("\n🔗 Calling OpenAI...")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You must return STRICT JSON only."},
                  {"role": "user", "content": content}],
        temperature=0,
        top_p=0.1,
        max_tokens=max(48, min(256, int(args.max_tokens))),
    )

    response_text = resp.choices[0].message.content or ""
    log("\nRaw response:\n" + (response_text[:1000] if response_text else ""))

    # 6) Parse
    box_number = None
    try:
        cleaned = response_text.strip()
        if cleaned.startswith('```'):
            m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.S)
            if m:
                cleaned = m.group(1).strip()
        data = json.loads(cleaned)
        box_number = data.get("box_number")
        if isinstance(box_number, str) and box_number.isdigit():
            box_number = int(box_number)
    except Exception as e:
        log(f"\n⚠️  Failed to parse JSON: {e}")
        print(-1)
        sys.exit(3)

    # Validate selection & move/click
    if not isinstance(box_number, int) or box_number not in element_map:
        log(f"Invalid or missing box_number: {box_number}")
        print(-1)
        sys.exit(4)

    # Move / click using pyautogui
    try:
        cx, cy = element_map[box_number]["center"]
        if args.move:
            pygui.moveTo(int(cx), int(cy), duration=max(0.0, float(args.move_duration)))
        if args.click:
            pygui.click()
    except Exception as e:
        log(f"Mouse action failed: {e}")
        # Still print the number, but non-zero exit to indicate mouse failure
        print(box_number)
        sys.exit(5)

    # Print ONLY the number to stdout (default quiet). Success exit code 0.
    print(box_number)
    return


if __name__ == "__main__":
    main()
