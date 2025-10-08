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
- 🎯 **Grid Detection** - Chess-board overlay for improved spatial accuracy
- 📚 **Learning System** - Saves corrections to improve over time
- 🔧 **MCP Integration** - Works with LM Studio and other MCP-compatible clients
- 🎮 **Interactive Training** - Build custom datasets for your workflow

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

## 🎯 Example Usage

```python
# Find and click the Chrome icon
python training.py
# Choose option 5
# Enter: "the Chrome icon in the taskbar"
# Verify and click!
```

## 📊 Accuracy

- **Basic prompting:** ~30-40%
- **With Quick Wins:** ~60-75%
- **With grid overlay:** ~75-85%
- **With training data (50+ examples):** ~80-90%

## 🤝 Contributing

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

**🎮 Have fun giving your LLM eyes!**
