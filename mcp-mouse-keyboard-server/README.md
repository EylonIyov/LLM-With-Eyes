# MCP Mouse & Keyboard Control Server

A Model Context Protocol (MCP) server that provides mouse and keyboard control capabilities to any LLM through PyAutoGUI.

## Features

This server exposes the following tools to LLMs:

### Mouse Control
- `get_mouse_position()` - Get current cursor position
- `move_mouse(x, y, duration)` - Move mouse to absolute coordinates
- `move_mouse_relative(x, y, duration)` - Move mouse relative to current position
- `click_mouse(button, clicks)` - Click mouse button (left/right/middle)
- `double_click()` - Perform double click
- `right_click()` - Perform right click
- `drag_mouse(x, y, duration)` - Drag mouse to coordinates
- `scroll_mouse(clicks)` - Scroll mouse wheel up/down

### Keyboard Control
- `type_text(text, interval)` - Type text
- `press_key(key)` - Press a single key
- `press_hotkey(*keys)` - Press key combination (e.g., Ctrl+C)
- `hold_key(key)` - Hold down a key
- `release_key(key)` - Release a held key

### Screen Utilities
- `get_screen_size()` - Get screen dimensions
- `take_screenshot(filename)` - Capture and save screenshot

## Installation

1. Activate your existing virtual environment:
```bash
..\.venv\Scripts\activate  # Windows (using parent folder's venv)
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Server Standalone
```bash
python server.py
```

### Connecting to Claude Desktop

Add to your Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "mouse-keyboard": {
      "command": "python",
      "args": ["C:\\Users\\user\\Desktop\\dev\\mcp-mouse-keyboard-server\\server.py"]
    }
  }
}
```

Or use `uv` for better dependency management:

```json
{
  "mcpServers": {
    "mouse-keyboard": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\Users\\user\\Desktop\\dev\\mcp-mouse-keyboard-server",
        "run",
        "server.py"
      ]
    }
  }
}
```

## Safety Features

- **Failsafe**: Move mouse to any corner of the screen to abort operations
- **Pause**: Small delay between actions to prevent accidents
- **User Approval**: All actions should be reviewed and approved by the user through the MCP client

## Example Usage

Once connected to an MCP client like Claude Desktop, you can ask:

- "Move my mouse to the center of the screen"
- "Type 'Hello World' for me"
- "Take a screenshot and save it as test.png"
- "Press Ctrl+C to copy"
- "Click at coordinates 500, 300"

## Security Warning

⚠️ **This server can control your mouse and keyboard!** 

- Only use with trusted MCP clients
- Always review and approve actions before execution
- The failsafe feature allows you to stop operations by moving mouse to screen corner
- Consider restricting which tools are available in production use

## Requirements

- Python 3.10+
- PyAutoGUI
- MCP Python SDK

## License

MIT
