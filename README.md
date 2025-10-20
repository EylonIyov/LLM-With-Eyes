# LLM With Eyes 👁️

> Give your local LLM vision and control over your computer's mouse and keyboard

A Python framework that enables vision-capable LLMs to see your screen and control your computer through natural language commands.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LM Studio](https://img.shields.io/badge/LM%20Studio-Compatible-green.svg)](https://lmstudio.ai/)
[![MCP](https://img.shields.io/badge/MCP-Protocol-orange.svg)](https://modelcontextprotocol.io/)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/EylonIyov/LLM-With-Eyes.git
cd LLM-With-Eyes

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**1. Test Vision + Mouse Control:**
```bash
python training.py
```
Choose option 5 to find and click UI elements.

**2. Grid-Based Detection (More Accurate):**
```bash
python grid_detection.py
```
Uses visual grid overlay for better spatial reasoning.

**3. Run with MCP Server:**
```bash
# See setup.md for MCP server configuration
python script.py
```

## ✨ Features

- 🖱️ **Computer Control** - Move mouse, click, type, and take screenshots
- 👁️ **Vision Integration** - LLM sees your screen and identifies UI elements
- 🔧 **MCP Integration** - Works with LM Studio and other MCP-compatible clients

## 📋 Requirements

- Python 3.8+
- LM Studio with a vision-capable model (e.g., Qwen2.5-VL)
- Windows (currently - mouse/keyboard control is OS-specific)

## 🎓 How It Works

1. **Screenshot** - Captures your screen
2. **Vision Analysis** - LLM sees the screenshot with optional grid overlay
3. **Element Detection** - Model identifies UI element coordinates
4. **Action Execution** - Controls mouse/keyboard via PyAutoGUI or MCP

## 📖 Documentation

- **[Quick Wins Guide](QUICK_WINS_README.md)** - Improve accuracy in minutes
- **[Training Guide](TRAINING_GUIDE.md)** - Advanced training strategies
- **[MCP Setup](setup.md)** - Configure MCP server (if using MCP)

## 🛠️ Key Scripts

| Script | Description |
|--------|-------------|
| `training.py` | Main training and testing interface |
| `grid_detection.py` | Grid-based element detection |
| `script.py` | Core functions and MCP integration |
| `create_training_dataset.py` | Build custom training datasets |
| `advanced_prompts.py` | Prompt engineering utilities |
| `cloud_bbox_picker.py` | Cloud bbox-based selection via remote VLM; prints box number and can move/click |

## 🎯 Example Usage

```python
# Find and click the Chrome icon
python training.py
# Choose option 5
# Enter: "the Chrome icon in the taskbar"
# Verify and click!
```

## ☁️ Cloud BBox Picker (remote model)

Run a remote vision model (e.g., GPT-4o) to choose a numbered bounding box from an annotated screenshot, then optionally move/click the mouse locally.

Prerequisites:
- `pip install openai`
- Set your API key via environment or `.env` file (key: `OPENAI_API_KEY`)

Basic usage (quiet by default: prints only the chosen number):
```powershell
python .\cloud_bbox_picker.py --target "the green box"
```

Click after moving:
```powershell
python .\cloud_bbox_picker.py --target "the green box" --click
```

Provide API key explicitly (overrides env/.env):
```powershell
python .\cloud_bbox_picker.py --target "OK button" --api-key "<YOUR_KEY>"
```

Full CLI syntax:
```text
cloud_bbox_picker.py [--api-key KEY] [--model MODEL] --target TEXT [--no-move] [--click]
					 [--move-duration SECS] [--verbose]
					 [--image-format png|jpeg|webp] [--quality 1-100]
					 [--resize-width PIXELS] [--no-crop] [--crop-only]
					 [--max-tokens N]
```

Flags:
- --api-key KEY           Use this API key (fallback: env OPENAI_API_KEY or .env)
- --model MODEL           Remote model (default: gpt-4o)
- --target TEXT           What to find (e.g., "red circle", "OK button")
- --no-move               Don’t move the mouse (default is to move)
- --click                 Click after moving
- --move-duration SECS    Mouse move duration (default: 0.2)
- --verbose               Verbose logs to stdout (quiet mode prints only the number)

Performance flags (reduce upload size and latency):
- --image-format FMT      png|jpeg|webp (default: jpeg)
- --quality Q             1–100 (lossy formats; default: 80)
- --resize-width W        Resize width in pixels, keep aspect ratio (default: 1280; 0 disables)
- --no-crop               Don’t send the crop image
- --crop-only             Send only the crop (fastest; skips full image)
- --max-tokens N          Completion tokens cap (default: 96)

Examples:
- Fast and small, crop-only:
```powershell
python .\cloud_bbox_picker.py --target "red circle" --crop-only --image-format jpeg --quality 70 --resize-width 1024 --max-tokens 64
```

- Higher fidelity (larger payload):
```powershell
python .\cloud_bbox_picker.py --target "play button" --image-format png --resize-width 0 --max-tokens 120
```

Notes:
- The script takes a screenshot, detects UI elements, draws numbered boxes, and builds a 2× annotated crop. It sends the prompt plus one or two images to the remote model, parses the returned JSON box_number, and moves/clicks locally unless disabled.
- API key resolution order: `--api-key` CLI > `OPENAI_API_KEY` env var > `.env` file at repo root.

## � Vision Agent (Reason → Act loop)

If you want the model to see the current screen, decide the next step, and use the mouse/keyboard like a normal user, run the agent loop:

```powershell
# Ensure OPENAI_API_KEY is set or present in .env
python .\vision_agent.py --goal "Open the folder C:\\Users\\user\\Desktop\\dev and open training.html" --max-steps 6 --verbose
```

Key flags:
- --goal: Natural-language objective; the agent plans one small action per step
- --model: OpenAI model (default gpt-4o)
- --api-key: API key (else uses env/.env)
- --max-steps: Safety cap (default 6)
- --move-duration: Mouse move duration (default 0.2s)
- --verbose: Print model JSON and executed actions each step
- --image-format/--quality/--resize-width/--no-crop: Performance tuning similar to the cloud picker

Actions the model can return:
- click or move on a numbered box (preferred)
- type_text, key, hotkey, scroll, wait
- done (when goal is satisfied)

The agent re-detects elements every step and prefers grounded clicks via the numbered boxes.

## �🤝 Contributing

Contributions welcome! This project is experimental and there's lots of room for improvement.

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Built with [PyAutoGUI](https://pyautogui.readthedocs.io/)
- Uses [OpenAI API](https://platform.openai.com/docs/api-reference) format
- Compatible with [LM Studio](https://lmstudio.ai/)
- Supports [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)

---

**⚠️ Warning:** This tool can control your mouse and keyboard. Use verification mode when testing!
