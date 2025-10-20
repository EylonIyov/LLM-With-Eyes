"""
Bounding Box Detection - Annotate all clickable elements with numbered boxes.
This makes it MUCH easier for the LLM - instead of guessing coordinates,
it just picks a number.

Accuracy improvement: +30-40% (one of the best techniques!)
"""

from PIL import Image, ImageDraw, ImageFont
import pyautogui as pygui
import cv2
import numpy as np
from script import encode_image_to_base64
import base64
import json
import re
import time
from openai import OpenAI
import os

# Optional OCR (graceful fallback if not installed)
try:
    import pytesseract  # Requires Tesseract OCR installed on system
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    OCR_AVAILABLE = False

# Configure pytesseract to point at the Tesseract binary on Windows if available
if OCR_AVAILABLE:
    try:
        import os, shutil
        # Priority: environment variables, then PATH, then common install paths
        candidate = os.getenv('TESSERACT_CMD') or os.getenv('TESSERACT_PATH')
        if not candidate or not os.path.exists(candidate):
            candidate = shutil.which('tesseract')
        if not candidate or not os.path.exists(candidate):
            for p in (
                r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
            ):
                if os.path.exists(p):
                    candidate = p
                    break
        if candidate and os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
    except Exception:
        # Leave default configuration; OCR calls will fail gracefully if missing
        pass


def _load_env_if_needed():
    """Load .env file if OPENAI_API_KEY not present in environment."""
    if os.environ.get("OPENAI_API_KEY"):
        return
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
                        if k and (k not in os.environ):
                            os.environ[k] = v
    except Exception:
        pass

def _move_via_mcp_tools(x: int, y: int, click: bool, duration: float = 1.0) -> bool:
    """Attempt to move/click using the MCP server's tool functions by importing its module.
    Returns True if successful, False otherwise.
    """
    try:
        # Dynamically load the MCP server module without static import
        import importlib.util
        module_path = os.path.join(os.path.dirname(__file__), 'mcp-mouse-keyboard-server', 'server.py')
        spec = importlib.util.spec_from_file_location('mcp_mouse_server', module_path)
        if not spec or not spec.loader:
            return False
        mcp_mouse_server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mcp_mouse_server)  # type: ignore[attr-defined]
        # Use exposed tool functions (decorated but callable directly)
        if not hasattr(mcp_mouse_server, 'move_mouse'):
            return False
        mcp_mouse_server.move_mouse(int(x), int(y), duration=float(duration))
        if click and hasattr(mcp_mouse_server, 'click_mouse'):
            mcp_mouse_server.click_mouse(button="left", clicks=1)
        return True
    except Exception:
        return False


def _preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    """Light preprocessing to improve OCR on UI screenshots.
    - Convert to grayscale
    - Increase contrast
    - Adaptive threshold
    Returns a PIL image suitable for Tesseract.
    """
    img = np.array(pil_image.convert("L"))  # grayscale
    # Slight denoise and sharpen edges
    img = cv2.bilateralFilter(img, 5, 50, 50)
    # Adaptive threshold to make text crisp
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    # Convert back to PIL
    return Image.fromarray(th)


def _ocr_text_from_bbox(pil_full_image: Image.Image, bbox: tuple) -> tuple[str, float] | tuple[str, None]:
    """Extract text within a bounding box using pytesseract if available.
    Returns (text, avg_confidence) or ("", None) if OCR unavailable or empty.
    """
    if not OCR_AVAILABLE:
        return "", None

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return "", None

    # Crop ROI safely
    W, H = pil_full_image.size
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    x1 = max(0, x)
    y1 = max(0, y)
    if x2 <= x1 or y2 <= y1:
        return "", None

    roi = pil_full_image.crop((x1, y1, x2, y2))
    roi = _preprocess_for_ocr(roi)

    try:
        data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
        words = []
        confs = []
        for txt, conf in zip(data.get('text', []), data.get('conf', [])):
            try:
                conf_val = float(conf)
            except Exception:
                conf_val = -1.0
            if txt and txt.strip() and conf_val >= 0:
                words.append(txt.strip())
                confs.append(conf_val)
        if not words:
            return "", None
        avg_conf = sum(confs) / max(1, len(confs)) if confs else None
        text = " ".join(words)
        # Normalize whitespace and trim
        text = re.sub(r"\s+", " ", text).strip()
        return text, avg_conf
    except Exception:
        return "", None


def detect_ui_elements(image_path, min_area=400, max_area=None, min_width=15, min_height=15):
    """
    Detect potential UI elements (buttons, icons, etc.) using computer vision.
    
    Args:
        image_path: Path to screenshot
        min_area: Minimum bounding box area (filters tiny elements)
        max_area: Maximum bounding box area (filters whole windows). If None, auto-calculated.
        min_width: Minimum width in pixels (filters very narrow elements)
        min_height: Minimum height in pixels (filters very short elements)
    
    Returns:
        List of bounding boxes: [(x, y, width, height), ...]
    """
    # Read image with OpenCV
    img = cv2.imread(image_path)
    
    # Auto-calculate max_area if not provided (80% of screen to avoid full-screen captures)
    if max_area is None:
        screen_area = img.shape[0] * img.shape[1]
        max_area = int(screen_area * 0.8)
        print(f"🔍 Auto-calculated max_area: {max_area:,} pixels (80% of screen)")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple detection methods
    elements = []
    
    # Method 1: Edge detection (for elements with clear boundaries)
    edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for better sensitivity
    
    # Method 2: Color-based detection (for solid colored elements like game boxes)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Broader, darker-tolerant blue; keep green and red
    lower_blue = np.array([90, 60, 30])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_red1 = np.array([0, 80, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Morphological cleanup to stabilize rounded shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Combine all color masks
    color_mask = cv2.bitwise_or(mask_blue, cv2.bitwise_or(mask_green, mask_red))

    # Find contours from color detection (use RETR_TREE to keep internal shapes)
    contours_color, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color_elements = 0
    filtered_by_size = 0
    for contour in contours_color:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Filter by minimum width and height (reject tiny boxes)
        if w < min_width or h < min_height:
            filtered_by_size += 1
            continue
        
        if min_area < area < max_area:
            aspect_ratio = w / h if h > 0 else 0
            # More permissive aspect ratio for large UI elements (0.1 to 10)
            if 0.1 < aspect_ratio < 10:
                elements.append((x, y, w, h))
                color_elements += 1
    
    print(f"   Color-based detection: {color_elements} elements (filtered {filtered_by_size} too small)")
    
    # Method 3: Find contours from edges (keep internal shapes)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    edge_elements = 0
    edge_filtered_by_size = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Filter by minimum width and height (reject tiny boxes)
        if w < min_width or h < min_height:
            edge_filtered_by_size += 1
            continue
        
        # Filter by area
        if min_area < area < max_area:
            # Filter by aspect ratio (avoid long thin lines) - more permissive for large elements
            aspect_ratio = w / h if h > 0 else 0
            if 0.1 < aspect_ratio < 10:
                elements.append((x, y, w, h))
                edge_elements += 1
    
    print(f"   Edge-based detection: {edge_elements} elements (filtered {edge_filtered_by_size} too small)")
    
    # Method 3: Template matching for common UI patterns
    # (You can add specific icon templates here)
    
    print(f"   Total before filtering: {len(elements)} elements")
    
    # Remove overlapping boxes (keep larger ones)
    elements = remove_overlapping_boxes(elements)
    
    print(f"   ✓ Final count after overlap removal: {len(elements)} elements")
    
    # Debug: show size distribution
    if elements:
        sizes = [w * h for x, y, w, h in elements]
        print(f"   Size range: {min(sizes):,} to {max(sizes):,} pixels")
    
    return elements


def remove_overlapping_boxes(boxes, iou_threshold=0.5, preserve_nested=True, container_drop_ratio=0.85):
    """
    Remove overlapping bounding boxes using Non-Maximum Suppression.
    
    Args:
        boxes: List of (x, y, w, h) tuples
        iou_threshold: Intersection over Union threshold
    
    Returns:
        Filtered list of boxes
    """
    if not boxes:
        return []
    
    # Convert to (x1, y1, x2, y2, area) format
    boxes_array = []
    for x, y, w, h in boxes:
        boxes_array.append([x, y, x + w, y + h, w * h])
    
    boxes_array = np.array(boxes_array)
    
    # Sort by area (descending)
    indices = np.argsort(boxes_array[:, 4])[::-1]
    
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes_array[current, :4]
        other_boxes = boxes_array[indices[1:], :4]
        
        # Calculate intersection
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        area_current = boxes_array[current, 4]
        area_others = boxes_array[indices[1:], 4]
        union = area_current + area_others - intersection
        
        # Calculate IoU
        iou = intersection / union
        
        # Determine which to keep based on IoU and containment
        remaining = []
        for idx, j in enumerate(indices[1:]):
            other = boxes_array[j, :4]
            # Check containment (preserve nested small boxes inside large ones)
            if preserve_nested:
                if (other[0] >= current_box[0] and other[1] >= current_box[1] and
                    other[2] <= current_box[2] and other[3] <= current_box[3]):
                    remaining.append(j)  # keep nested
                    continue
            if iou[idx] < iou_threshold:
                remaining.append(j)
        indices = np.array(remaining, dtype=int)
    
    # Convert back to original format and drop giant containers
    result = []
    if len(boxes_array) > 0:
        # screen-like reference from boxes extents (best-effort)
        xs1 = boxes_array[:, 0]; ys1 = boxes_array[:, 1]
        xs2 = boxes_array[:, 2]; ys2 = boxes_array[:, 3]
        W = max(xs2) - min(xs1)
        H = max(ys2) - min(ys1)
        screen_area_est = max(1, W * H)
    else:
        screen_area_est = 1
    for idx in keep:
        x1, y1, x2, y2, a = boxes_array[idx]
        if (x2 - x1) * (y2 - y1) >= container_drop_ratio * screen_area_est:
            # skip giant container
            continue
        result.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
    
    return result


def create_bbox_overlay(image_path, elements=None, save_path=None):
    """
    Draw numbered bounding boxes on all detected elements.
    
    Args:
        image_path: Path to screenshot
        elements: List of (x, y, w, h) or None to auto-detect
        save_path: Where to save annotated image
    
    Returns:
        (annotated_image_path, element_map)
        element_map: {box_number: {"bbox": (x, y, w, h), "center": (cx, cy)}}
    """
    # Auto-detect if not provided
    if elements is None:
        print("🔍 Detecting UI elements...")
        elements = detect_ui_elements(image_path)
        print(f"✓ Found {len(elements)} potential elements")
    
    # Open image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    img_np = np.array(img)  # RGB
    H_img, W_img = img_np.shape[0], img_np.shape[1]

    def _dominant_color_name(roi_rgb: np.ndarray) -> str:
        try:
            if roi_rgb.size == 0:
                return "unknown"
            roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            h = np.mean(roi_hsv[:, :, 0])
            s = np.mean(roi_hsv[:, :, 1])
            v = np.mean(roi_hsv[:, :, 2])
            if s < 30:
                if v > 200:
                    return "white"
                if v < 50:
                    return "black"
                return "gray"
            if (h <= 10) or (h >= 170):
                return "red"
            if 15 <= h <= 35:
                return "yellow"
            if 40 <= h <= 85:
                return "green"
            if 95 <= h <= 140:
                return "blue"
            return "other"
        except Exception:
            return "unknown"
    
    # Create element map
    element_map = {}
    
    # Draw boxes and numbers
    for i, (x, y, w, h) in enumerate(elements, start=1):
        # Calculate center
        cx = x + w // 2
        cy = y + h // 2
        
        # Store in map
        ocr_text = ""
        ocr_conf = None
        # Attempt OCR for each element (optional)
        try:
            ocr_text, ocr_conf = _ocr_text_from_bbox(img, (x, y, w, h))
        except Exception:
            ocr_text, ocr_conf = "", None

        # Derive simple color/region metadata for prompting
        x2 = x + w
        y2 = y + h
        roi = img_np[max(0, y):max(0, y2), max(0, x):max(0, x2)]
        color_name = _dominant_color_name(roi)
        # Region as 3x3 grid label
        nx = (cx / max(1, W_img))
        ny = (cy / max(1, H_img))
        col = "left" if nx < 1/3 else ("center" if nx < 2/3 else "right")
        row = "top" if ny < 1/3 else ("middle" if ny < 2/3 else "bottom")
        region_label = f"{row}-{col}"

        element_map[i] = {
            "bbox": (x, y, w, h),
            "center": (cx, cy),
            "area": w * h,
            "text": ocr_text,
            "text_confidence": ocr_conf,
            "color": color_name,
            "region": region_label
        }
        
        # Draw bounding box (thick green outline)
        box_color = (0, 255, 0)  # Green
        thickness = max(2, min(w, h) // 30)  # Scale thickness with box size
        draw.rectangle([x, y, x + w, y + h], outline=box_color, width=thickness)
        
        # Draw number label with background
        label = str(i)
        
        # Calculate font size relative to box size (use smaller dimension)
        min_dimension = min(w, h)
        font_size = max(16, min(64, int(min_dimension * 0.45)))  # Larger, high-contrast labels
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get accurate text size using textbbox
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            text_width = len(label) * (font_size // 2)
            text_height = font_size
        
        # Background rectangle for label (with padding)
        padding = max(3, font_size // 8)
        label_x = x + thickness + 2
        label_y = y + thickness + 2
        draw.rectangle(
            [label_x, label_y, label_x + text_width + padding * 2, label_y + text_height + padding * 2],
            fill=(255, 255, 255),  # White background for maximum contrast
            outline=(0, 0, 0),
            width=max(1, thickness // 2)
        )
        
        # Draw text label number
        draw.text((label_x + padding, label_y + padding), label, fill=(0, 0, 0), font=font)

        # Optionally draw a short OCR snippet below the label for readability
        if ocr_text:
            snippet = ocr_text[:24] + ("…" if len(ocr_text) > 24 else "")
            try:
                sn_bbox = draw.textbbox((0, 0), snippet, font=font)
                sn_w = sn_bbox[2] - sn_bbox[0]
                sn_h = sn_bbox[3] - sn_bbox[1]
            except Exception:
                sn_w = len(snippet) * (font_size // 2)
                sn_h = font_size

            sn_x = label_x
            sn_y = label_y + text_height + padding * 2 + 2
            draw.rectangle(
                [sn_x, sn_y, sn_x + sn_w + padding * 2, sn_y + sn_h + padding * 2],
                fill=(255, 255, 255),
                outline=(0, 0, 0),
                width=max(1, thickness // 2)
            )
            draw.text((sn_x + padding, sn_y + padding), snippet, fill=(0, 0, 0), font=font)
    
    # Save annotated image
    if save_path is None:
        save_path = image_path.replace('.png', '_bbox.png')
    img.save(save_path)
    
    print(f"✓ Annotated {len(elements)} elements")
    
    return save_path, element_map


def create_bbox_prompt(target_description, num_elements, element_map=None):
    """
    Create a prompt for bounding box based detection.
    
    Args:
        target_description: What to find
        num_elements: Total number of labeled boxes
    
    Returns:
        Prompt string
    """
    # Build a compact catalog per box (id, region, color, size, text)
    catalog_lines = []
    if element_map:
        for idx in sorted(element_map.keys()):
            meta = element_map[idx]
            x, y, w, h = meta.get("bbox", (0, 0, 0, 0))
            region = meta.get("region") or "?"
            color = meta.get("color") or "?"
            text = meta.get("text") or ""
            text_short = (text[:38] + "…") if len(text) > 38 else text
            catalog_lines.append(
                f"- Box {idx}: region={region}, color={color}, size={w}x{h}, text=\"{text_short}\""
            )
    catalog_block = "\n".join(catalog_lines)

    prompt = f"""You are looking at a screenshot with NUMBERED BOUNDING BOXES around all clickable UI elements.

Each green box has a NUMBER (1-{num_elements}).

YOUR TASK: Find {target_description}

INSTRUCTIONS:
1. Look at ALL the numbered boxes in the screenshot
2. Identify which box contains {target_description}
3. Return the BOX NUMBER

This is MUCH easier than guessing pixel coordinates - just pick the right box number!

Use this catalog to help you reason about the boxes:
{catalog_block if catalog_block else '(no catalog available)'}

EXAMPLES:
- "Find the blue box" - Look for blue box, check that this is the correct element according to the prompt and return its box number
Respond in STRICT JSON only (no extra text) with this schema:
{{
    "target_found": true | false,
    "box_number": integer  // 1 to {num_elements}
}}

IMPORTANT: Just pick the box number that contains the target element!
"""
    return prompt


def test_bbox_detection(target_description, count_id=0, auto_detect=True, click=False, min_box_size=20):
    """
    Test finding elements using bounding box detection.
    
    Args:
        target_description: What to find
        count_id: Screenshot counter
        auto_detect: If True, auto-detect UI elements. If False, detect manually.
        click: If True, click the element after finding it
        min_box_size: Minimum width/height in pixels (filters tiny boxes)
    
    Returns:
        Result dictionary
    """
    # Take screenshot
    screenshot_path = f"screenshot_pygui_{count_id}.png"
    pygui.screenshot(screenshot_path)
    print(f"📸 Screenshot saved: {screenshot_path}")
    
    # Detect elements and create bounding boxes
    if auto_detect:
        elements = detect_ui_elements(screenshot_path, min_area=200, min_width=min_box_size, min_height=min_box_size)
    else:
        # Manual mode: user clicks elements
        print("\n📍 Manual element detection mode")
        print("   This mode will be added in future version")
        elements = detect_ui_elements(screenshot_path, min_area=200, min_width=min_box_size, min_height=min_box_size)
    
    # Create annotated image
    bbox_path, element_map = create_bbox_overlay(screenshot_path, elements)

    # Build a tight crop around union of all boxes and upscale for better small-text fidelity
    try:
        if element_map:
            xs = []
            ys = []
            x2s = []
            y2s = []
            for meta in element_map.values():
                bx, by, bw, bh = meta["bbox"]
                xs.append(bx)
                ys.append(by)
                x2s.append(bx + bw)
                y2s.append(by + bh)
            x1 = max(0, min(xs) - 20)
            y1 = max(0, min(ys) - 20)
            x2 = max(x2s) + 20
            y2 = max(y2s) + 20

            full_img = Image.open(screenshot_path)
            W, H = full_img.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                crop = full_img.crop((x1, y1, x2, y2))
                # Upscale 2x for better readability
                crop_up = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
                crop_path = screenshot_path.replace('.png', '_crop2x.png')
                crop_up.save(crop_path)

                # Also draw the same numbered bounding boxes on the upscaled crop
                try:
                    annot = crop_up.copy()
                    draw_c = ImageDraw.Draw(annot)
                    for idx in sorted(element_map.keys()):
                        bx, by, bw, bh = element_map[idx]["bbox"]
                        # Keep only boxes whose center falls inside the crop region
                        cx, cy = element_map[idx]["center"]
                        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                            continue
                        # Transform to crop-up coordinates (offset and scale 2x)
                        tx = int((bx - x1) * 2)
                        ty = int((by - y1) * 2)
                        tw = int(bw * 2)
                        th = int(bh * 2)

                        # Draw box
                        box_color = (0, 255, 0)
                        thickness = max(2, min(tw, th) // 30)
                        draw_c.rectangle([tx, ty, tx + tw, ty + th], outline=box_color, width=thickness)

                        # Label number
                        label = str(idx)
                        min_dim = max(1, min(tw, th))
                        font_size = max(16, min(64, int(min_dim * 0.45)))
                        try:
                            font_c = ImageFont.truetype("arial.ttf", font_size)
                        except Exception:
                            font_c = ImageFont.load_default()
                        try:
                            bb = draw_c.textbbox((0, 0), label, font=font_c)
                            twd = bb[2] - bb[0]
                            thd = bb[3] - bb[1]
                        except Exception:
                            twd = len(label) * (font_size // 2)
                            thd = font_size
                        padding = max(3, font_size // 8)
                        lx = tx + thickness + 2
                        ly = ty + thickness + 2
                        draw_c.rectangle([lx, ly, lx + twd + padding * 2, ly + thd + padding * 2],
                                         fill=(255, 255, 255), outline=(0, 0, 0), width=max(1, thickness // 2))
                        draw_c.text((lx + padding, ly + padding), label, fill=(0, 0, 0), font=font_c)

                    crop_path_bbox = screenshot_path.replace('.png', '_crop2x_bbox.png')
                    annot.save(crop_path_bbox)
                except Exception:
                    crop_path_bbox = None
            else:
                crop_path = None
                crop_path_bbox = None
        else:
            crop_path = None
            crop_path_bbox = None
    except Exception:
        crop_path = None
        crop_path_bbox = None
    
    # Encode annotated image
    with open(bbox_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
    # Optionally encode the 2x crop
    base64_crop = None
    if crop_path_bbox:
        try:
            with open(crop_path_bbox, "rb") as cimg:
                base64_crop = base64.b64encode(cimg.read()).decode('utf-8')
        except Exception:
            base64_crop = None
    elif crop_path:
        try:
            with open(crop_path, "rb") as cimg:
                base64_crop = base64.b64encode(cimg.read()).decode('utf-8')
        except Exception:
            base64_crop = None
    
    # Create prompt (include OCR metadata)
    prompt = create_bbox_prompt(target_description, len(element_map), element_map)

    if not OCR_AVAILABLE:
        print("ℹ️  OCR not available (pytesseract not installed or tesseract binary missing). Proceeding without text hints.")
    
    # Remote-first selection: try OpenAI (cloud) to pick box number
    print(f"\n🔍 Searching for: {target_description}")
    selected_box = None
    _load_env_if_needed()
    remote_api_key = os.environ.get("OPENAI_API_KEY")
    if remote_api_key:
        try:
            content_remote = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ]
            if base64_crop:
                content_remote.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_crop}"}
                })

            client_remote = OpenAI(api_key=remote_api_key)
            resp_remote = client_remote.chat.completions.create(
                model=os.environ.get("OPENAI_VISION_MODEL", "gpt-4o"),
                messages=[
                    {"role": "system", "content": "You must return STRICT JSON only."},
                    {"role": "user", "content": content_remote}
                ],
                temperature=0,
                top_p=0.1,
                max_tokens=120,
            )
            remote_text = resp_remote.choices[0].message.content or ""
            # Parse JSON
            cleaned = remote_text.strip()
            if cleaned.startswith('```'):
                m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.S)
                if m:
                    cleaned = m.group(1).strip()
            data = json.loads(cleaned)
            box_number = data.get("box_number")
            if isinstance(box_number, str) and box_number.isdigit():
                box_number = int(box_number)
            if isinstance(box_number, int) and box_number in element_map:
                selected_box = box_number
                print(f"\n🌐 Remote model selected box #{selected_box}")
            else:
                print(f"\n⚠️  Remote selection invalid: {box_number}")
        except Exception as e:
            print(f"\n⚠️  Remote selection failed: {e}")
    else:
        print("ℹ️  OPENAI_API_KEY not set; skipping remote selection.")

    # If remote succeeded, optionally notify local LLM then move mouse
    if isinstance(selected_box, int):
        element = element_map[selected_box]
        x, y = element['center']
        bbox = element['bbox']

        # Send a brief acknowledgement prompt to local LLM for traceability
        try:
            client_local = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            move_prompt = (
                f"The remote vision model chose box {selected_box}. "
                f"Move the mouse to its center coordinates ({x}, {y}). Respond with JSON: {{\"ack\": true}}"
            )
            content_local = [
                {"type": "text", "text": move_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ]
            if base64_crop:
                content_local.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_crop}"}
                })
            _ = client_local.chat.completions.create(
                model="qwen/qwen2.5-vl-7b",
                messages=[{"role": "user", "content": content_local}],
                temperature=0,
                top_p=0,
                max_tokens=32,
            )
        except Exception:
            # Local LM may not be running; proceed regardless
            pass

        # Move mouse via MCP server tools if available; fallback to PyAutoGUI
        print(f"\n🖱️  Moving mouse to ({x}, {y})...")
        if not _move_via_mcp_tools(x, y, click, duration=1.0):
            pygui.moveTo(x, y, duration=1)
            if click:
                time.sleep(0.5)
                pygui.click()

        return {
            "success": True,
            "box_number": selected_box,
            "x": x,
            "y": y,
            "bbox": bbox,
            "via": "remote",
        }

    # Fallback: use local LLM to select box as before
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    content_payload = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
    ]
    if base64_crop:
        content_payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_crop}"}
        })

    resp = client.chat.completions.create(
        model="qwen/qwen2.5-vl-7b",
        messages=[{
            "role": "user",
            "content": content_payload
        }],
        temperature=0,
        top_p=0
    )
    
    response_text = resp.choices[0].message.content
    print("\nModel Response:")
    print(response_text)
    
    # Parse response
    try:
        # Clean JSON
        cleaned = response_text.strip()
        if cleaned.startswith('```'):
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(1).strip()
        
        response_json = json.loads(cleaned)
        
        if response_json.get('target_found'):
            box_number = response_json.get('box_number')
            # Normalize box_number to int if it's a string like "3"
            if isinstance(box_number, str):
                box_number = int(re.sub(r"[^0-9]", "", box_number)) if re.search(r"\d+", box_number) else None
            
            if box_number in element_map:
                element = element_map[box_number]
                x, y = element['center']
                bbox = element['bbox']
                
                print(f"\n✅ Found in box #{box_number}")
                print(f"   Bounding box: {bbox}")
                print(f"   Center: ({x}, {y})")
                print(f"   Description: {response_json.get('description')}")
                print(f"   Reasoning: {response_json.get('reasoning')}")
                print(f"   Confidence: {response_json.get('confidence')}")
                
                # Move mouse via MCP server tools if available; fallback to PyAutoGUI
                print(f"\n🖱️  Moving mouse to ({x}, {y})...")
                if not _move_via_mcp_tools(x, y, click, duration=1.0):
                    pygui.moveTo(x, y, duration=1)
                    if click:
                        time.sleep(0.5)
                        pygui.click()
                
                return {
                    "success": True,
                    "box_number": box_number,
                    "x": x,
                    "y": y,
                    "bbox": bbox,
                    "confidence": response_json.get('confidence'),
                    "via": "local",
                }
            else:
                print(f"\n❌ Invalid box number: {box_number} (max: {len(element_map)})")
                return {"success": False, "reason": "invalid_box_number"}
        else:
            print("\n❌ Element not found")
            return {"success": False, "reason": "not_found"}
    
    except Exception as e:
        print(f"\n❌ Error parsing response: {e}")
        print("   Raw model output (truncated):", (response_text or "")[:400])
        return {"success": False, "reason": "parse_error", "error": str(e)}


if __name__ == "__main__":
    print("📦 Bounding Box Detection Test\n")
    print("This mode detects all UI elements and numbers them.")
    print("The LLM just picks the right number - much easier!\n")
    if not OCR_AVAILABLE:
        print("ℹ️  Tip: Enable OCR text extraction by installing Tesseract and pytesseract.")
        print("   pip install pytesseract   (Python package)")
        print("   Install Tesseract OCR: https://tesseract-ocr.github.io/tessdoc/Home.html\n")
    
    # Test parameters
    target = input("What should I find? > ")
    
    do_click = input("Click after finding? (y/n) [default: n]: ").strip().lower()
    click = do_click == 'y'
    
    min_size_input = input("Minimum box size in pixels? [default: 20]: ").strip()
    min_box_size = int(min_size_input) if min_size_input else 20
    
    print(f"\n⏳ Taking screenshot in 3 seconds...")
    time.sleep(3)
    
    result = test_bbox_detection(target, count_id=0, click=click, min_box_size=min_box_size)
    
    if result['success']:
        print(f"\n🎉 Success! Element found in box #{result['box_number']}")
        print(f"   Check the *_bbox.png file to see the annotated screenshot!")
    else:
        print(f"\n❌ Failed: {result.get('reason', 'unknown')}")
        print(f"   Check the *_bbox.png file to see detected elements.")
