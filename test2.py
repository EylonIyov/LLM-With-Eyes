import cv2
import numpy as np
import pytesseract
import os
import re
from dataclasses import dataclass

# Configure pytesseract to use the installed Tesseract binary, with env override support
_cand = os.getenv('TESSERACT_CMD') or os.getenv('TESSERACT_PATH')
if not _cand or not os.path.exists(_cand):
    for _p in (
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
    ):
        if os.path.exists(_p):
            _cand = _p
            break
if _cand and os.path.exists(_cand):
    pytesseract.pytesseract.tesseract_cmd = _cand

@dataclass
class Rect:
    x:int; y:int; w:int; h:int
    def contains_point(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h
    def tl(self):
        return (self.x, self.y)

def ocr_words(image_bgr):
    """Return list of (text, x, y, w, h) from Tesseract."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        if not txt: 
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        words.append((txt, x, y, w, h))
    return words

def find_label_center(words, target="megabonk"):
    """Find center of the word (case-insensitive, allows partial)."""
    target = target.lower()
    cand = []
    for txt, x, y, w, h in words:
        if target in txt.lower():
            cand.append((x + w//2, y + h//2, w*h))
    if not cand:
        return None
    # pick the largest occurrence (most confident)
    cand.sort(key=lambda t: t[2], reverse=True)
    return (cand[0][0], cand[0][1])

def detect_green_rects(image_bgr):
    """Detect neon-green rectangles and return bounding boxes."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Neon green range (tweak if needed)
    lower = np.array([40, 120, 120])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up and find contours
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    H, W = image_bgr.shape[:2]
    min_area = (W*H) * 0.003   # ignore very tiny greens
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h >= min_area:
            rects.append(Rect(x,y,w,h))
    return rects

def read_corner_digit(image_bgr, rect: Rect):
    """Crop a small tile near the top-left inside the green box and OCR a single digit."""
    # A small inset (avoid the border)
    pad = max(6, min(rect.w, rect.h) // 60)
    tile_size = max(32, min(rect.w, rect.h) // 12)  # adaptive crop size
    x0 = rect.x + pad
    y0 = rect.y + pad
    x1 = min(x0 + tile_size, image_bgr.shape[1]-1)
    y1 = min(y0 + tile_size, image_bgr.shape[0]-1)

    crop = image_bgr[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return None

    # Preprocess for high-contrast digit OCR
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Try inverse/binary both ways for robustness
    best_txt = None
    for invert in (False, True):
        g = 255 - gray if invert else gray
        g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(g, (3,3), 0)
        thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # digits-only OCR
        cfg = "--psm 8 -c tessedit_char_whitelist=0123456789"
        txt = pytesseract.image_to_string(thr, config=cfg).strip()
        txt = re.sub(r"\D", "", txt)  # keep digits
        if txt:
            best_txt = txt
            break
    return best_txt

def number_for_box_with_text(image_path, target_text="Megabonk"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    words = ocr_words(img)
    center = find_label_center(words, target_text)
    if center is None:
        raise RuntimeError(f"Text '{target_text}' not found by OCR (try raising DPI or using EasyOCR).")

    rects = detect_green_rects(img)
    if not rects:
        raise RuntimeError("No green boxes detected. Adjust HSV thresholds.")

    # pick the green rect that contains (or is nearest to) the text center
    cx, cy = center
    chosen = None
    best_dist = 1e12
    for r in rects:
        if r.contains_point(cx, cy):
            chosen = r
            break
        # distance from center to rect
        rx = np.clip(cx, r.x, r.x + r.w)
        ry = np.clip(cy, r.y, r.y + r.h)
        d = (rx - cx)**2 + (ry - cy)**2
        if d < best_dist:
            best_dist, chosen = d, r

    digit = read_corner_digit(img, chosen)
    if not digit:
        raise RuntimeError("Failed to OCR the corner digit; adjust crop sizing or preprocessing.")
    return int(digit), chosen

if __name__ == "__main__":
    
    image_path = input("Enter path to screenshot image: ").strip()
    #image_path = r"desktop_boxes.png"  # <-- replace with your screenshot path
    target_text = input("Enter target text to find (e.g. 'Megabonk'): ").strip()
    #
    num, rect = number_for_box_with_text(image_path, target_text=target_text)
    print(f"Detected number for the box containing '{target_text}': {num}  (rect={rect})")