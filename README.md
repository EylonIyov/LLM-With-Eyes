# SeeClick‑style Vision Agent (Windows)

Grounded computer‑use agent that observes the screen, reasons over a structured list of detected UI elements, and acts via mouse/keyboard tools. Includes a simple training utility to collect few‑shot examples and test local VLMs.

## Highlights
- Grounded actions: Interact by referencing numbered boxes (click_box, hover_box, etc.) anchored to detected UI elements.
- Strict JSON outputs: The VLM returns one action per step as strict JSON for reliable execution.
- Multi‑view perception: Raw screenshot + annotated overlay with numbered boxes.
- Short post‑action delay: The agent waits ~1.5s after each action before taking the next screenshot, allowing the UI to update.
- Execution: MCP mouse/keyboard tools with PyAutoGUI fallback (Windows).

---

## How it Works
1. Observe
   - Capture a screenshot.
   - Detect UI elements and generate numbered bounding boxes with brief OCR text.
   - Build a compact “catalog” of boxes and attach annotated images to the prompt.

2. Reason
   - Send the goal, recent history, memory, and the structured box list to a VLM.
   - The VLM replies with strict JSON describing the next action.

3. Act
   - Execute the action (mouse/keyboard via MCP, with PyAutoGUI fallback).
   - Wait ~1.5 seconds, then take the next screenshot and repeat until done or max steps reached.

---

## Requirements
- Windows 10/11
- Python 3.10+
- Packages (install what your code imports):
  - openai
  - pyautogui
  - pillow
  - requests
  - plus any local modules (bbox_detection.py, script.py, advanced_prompts.py)
- Optional: LM Studio (OpenAI‑compatible server) for local VLM testing

Install:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install openai pyautogui pillow requests
```

If you use additional detectors (e.g., OCR/CV), install their packages too.

---

## Configuration
- API
  - OPENAI_API_KEY in your environment or .env (if calling OpenAI).
  - Or run a local server (e.g., LM Studio) and point your client to http://localhost:1234/v1.
- Permissions
  - On Windows, run VS Code or the terminal “as Administrator” so PyAutoGUI/MCP can control the mouse/keyboard.
- Display
  - Prefer 100% scaling to avoid coordinate offsets.

---

## Run the Vision Agent
Quick start (PowerShell):
```powershell
python .\vision_agent.py --goal "Open the folder C:\Users\user\Desktop\dev and open training.html" --max-steps 6 --verbose
```

Notes
- The agent expects the VLM to return strict JSON (no Markdown fences).
- A short delay (~1.5s) is applied after each action before the next screenshot to let the UI settle.

---

## Action Schema (VLM → Agent)
The model must return strict JSON like:
```json
{
  "action": "click_box",
  "params": { "box_id": 7 },
  "reason": "Focus the search input",
  "plan": "Then type the query"
}
```

Allowed actions
- Grounded mouse
  - click_box, double_click_box, right_click_box, hover_box
    - params: { "box_id": <int> }
- Typing
  - type_text
    - params: { "text": "<string>", "submit": <bool, optional> }
- Keys
  - key
    - params: { "key": "<string>" }           // e.g., "enter", "esc"
  - hotkey
    - params: { "keys": ["ctrl","l"] }
- Other
  - scroll
    - params: { "amount": <int> }             // positive=up, negative=down
  - wait
    - params: { "seconds": <float> }
  - remember / forget
    - params: { "key": "<string>", "value": "<any>" } // forget: { "key": "<string>" }
  - done
    - params: { "message": "<string>" }

Legacy coordinate/description clicks may exist but prefer the grounded _box variants.

---

## Example Trajectory
1) Focus search bar
```json
{ "action": "click_box", "params": { "box_id": 12 }, "reason": "Focus the search field" }
```
2) Type and submit
```json
{ "action": "type_text", "params": { "text": "Eylon Iyov", "submit": true } }
```
3) Finish
```json
{ "action": "done", "params": { "message": "Search submitted" } }
```

---

## Timing and Delays
- After each executed action, the agent waits ~1.5 seconds before taking the next screenshot. This prevents stale captures and lets pages/app windows update.
- You can still add explicit waits via the wait action if a step needs longer.

---

## Training Utilities (training.py)
Use local VLMs to practice coordinate grounding and build few‑shot examples.

- Backends
  - Uses OpenAI SDK pointed at LM Studio:
    - base_url: http://localhost:1234/v1
    - model: qwen/qwen2.5-vl-7b (adjust to your local model)

- Dataset
  - Saves examples to training_dataset.json with screenshot, ground‑truth coordinates, and optional model prediction error.

Run:
```powershell
python .\training.py
```

Modes
1. Predefined training dataset
2. Interactive training (describe a target; model finds it)
3. Single supervised example (you provide ground‑truth coordinates)
4. Find element without ground truth (just get model output)
5. Test: Find and click an element (optional verification + save corrections)
6. View training statistics

Key functions
- capture_screenshot, encode_image_to_base64 (from script.py)
- create_few_shot_prompt, create_region_prompt (from advanced_prompts.py)
- invoke_mouse_keyboard to execute moves/clicks via MCP

Tips
- Use “verify before click” to correct mistakes and save better examples.
- Keep your display at 100% scaling for coordinate consistency.

---

## Troubleshooting
- Model outputs Markdown code fences
  - Ensure the system prompt instructs “strict JSON only” and strip fences if needed.
- No boxes or incorrect detections
  - Check bbox_detection.py and its dependencies (OCR/CV).
- Mouse/keyboard not moving
  - Run the shell as Administrator; confirm PyAutoGUI works outside the agent.
- Screenshots appear stale
  - The built‑in ~1.5s post‑action delay helps; add a wait action for slow pages.

---

## Contributing
- Open issues for detection errors, action schema extensions, or model prompt improvements.
- Share tricky test cases and few‑shot examples.

License: MIT