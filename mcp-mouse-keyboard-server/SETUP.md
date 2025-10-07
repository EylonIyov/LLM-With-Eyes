# Quick Setup Guide

### What You Have

A fully functional MCP server that exposes **17 tools** for controlling your mouse and keyboard to any LLM:

**Mouse Tools (8):**
- Get position, move (absolute/relative), click, double-click, right-click, drag, scroll

**Keyboard Tools (5):**
- Type text, press key, hotkeys, hold/release keys

**Screen Tools (2):**
- Get screen size, take screenshots

### How to Connect to Claude Desktop

1. **Open Claude Desktop config file:**
   - Press `Win + R`, type: `%APPDATA%\Claude\claude_desktop_config.json`
   - Or manually navigate to: `C:\Users\user\AppData\Roaming\Claude\claude_desktop_config.json`

2. **Add this configuration:**
```json
{
  "mcpServers": {
    "mouse-keyboard": {
      "command": "C:\\Users\\user\\Desktop\\dev\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\user\\Desktop\\dev\\mcp-mouse-keyboard-server\\server.py"]
    }
  }
}
```

3. **Restart Claude Desktop**

4. **Test it!** Ask Claude:
   - "What's my current mouse position?"
   - "Move my mouse to the center of the screen"
   - "Type 'Hello from MCP!' for me"

### Safety Notes

⚠️ **Important:**
- All mouse/keyboard actions require your approval in Claude
- Move mouse to any screen corner to abort (failsafe feature)
- Only use with trusted MCP clients

### Files Created

```
mcp-mouse-keyboard-server/
├── server.py           # Main MCP server with all tools
├── requirements.txt    # Python dependencies
├── README.md          # Full documentation
└── SETUP.md           # This file
```

### Next Steps

1. Connect to Claude Desktop (instructions above)
2. Test the tools
3. Explore what's possible with AI-controlled mouse/keyboard!

### Troubleshooting

If the server doesn't appear in Claude:
1. Check the config file path is correct
2. Make sure paths use double backslashes (`\\`) in JSON
3. Restart Claude Desktop completely
4. Check Claude's MCP logs: `%APPDATA%\Claude\logs\`

---

**Enjoy your new MCP-powered automation! 🚀**
