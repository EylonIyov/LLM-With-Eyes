"""
MCP Server for Mouse and Keyboard Control using PyAutoGUI
This server exposes tools for controlling mouse and keyboard through MCP.
"""
import os
import asyncio
from mcp.server.fastmcp import FastMCP
import pyautogui as pygui

# Initialize FastMCP server
mcp = FastMCP("mouse-keyboard-control")

# Configure PyAutoGUI safety settings
pygui.FAILSAFE = True  # Move mouse to corner to abort
pygui.PAUSE = 0.1  # Small delay between actions


# ===== MOUSE TOOLS =====

@mcp.tool()
def get_mouse_position() -> dict:
    """Get the current mouse cursor position."""
    x, y = pygui.position()
    return {"x": x, "y": y}


@mcp.tool()
def move_mouse(x: int, y: int, duration: float = 0.5) -> str:
    """
    Move the mouse to absolute coordinates.
    
    Args:
        x: Target X coordinate
        y: Target Y coordinate
        duration: Time in seconds for the movement (default 0.5)
    """
    pygui.moveTo(x, y, duration=duration)
    return f"Mouse moved to ({x}, {y})"


@mcp.tool()
def move_mouse_relative(x: int, y: int, duration: float = 0.5) -> str:
    """
    Move the mouse relative to current position.
    
    Args:
        x: Pixels to move horizontally (positive=right, negative=left)
        y: Pixels to move vertically (positive=down, negative=up)
        duration: Time in seconds for the movement (default 0.5)
    """
    pygui.move(x, y, duration=duration)
    return f"Mouse moved by ({x}, {y})"


@mcp.tool()
def click_mouse(button: str = "left", clicks: int = 1) -> str:
    """
    Click the mouse button.
    
    Args:
        button: Which button to click - "left", "right", or "middle" (default "left")
        clicks: Number of clicks (default 1)
    """
    pygui.click(button=button, clicks=clicks)
    return f"Clicked {button} button {clicks} time(s)"


@mcp.tool()
def double_click() -> str:
    """Perform a double click."""
    pygui.doubleClick()
    return "Double clicked"


@mcp.tool()
def right_click() -> str:
    """Perform a right click."""
    pygui.rightClick()
    return "Right clicked"


@mcp.tool()
def drag_mouse(x: int, y: int, duration: float = 0.5) -> str:
    """
    Drag the mouse to absolute coordinates.
    
    Args:
        x: Target X coordinate
        y: Target Y coordinate
        duration: Time in seconds for the drag (default 0.5)
    """
    pygui.dragTo(x, y, duration=duration)
    return f"Dragged mouse to ({x}, {y})"


@mcp.tool()
def scroll_mouse(clicks: int) -> str:
    """
    Scroll the mouse wheel.
    
    Args:
        clicks: Number of "clicks" to scroll. Positive scrolls up, negative scrolls down
    """
    pygui.scroll(clicks)
    direction = "up" if clicks > 0 else "down"
    return f"Scrolled {abs(clicks)} clicks {direction}"


# ===== KEYBOARD TOOLS =====

@mcp.tool()
def type_text(text: str, interval: float = 0.0) -> str:
    """
    Type text using the keyboard.
    
    Args:
        text: The text to type
        interval: Seconds between each keystroke (default 0.0)
    """
    pygui.write(text, interval=interval)
    return f"Typed: {text}"


@mcp.tool()
def press_key(key: str) -> str:
    """
    Press and release a single key.
    
    Args:
        key: Key name (e.g., 'enter', 'space', 'a', 'shift', 'ctrl', 'alt', 'tab', 'f1', etc.)
    """
    pygui.press(key)
    return f"Pressed key: {key}"


@mcp.tool()
def press_hotkey(*keys: str) -> str:
    """
    Press a combination of keys simultaneously (hotkey).
    
    Args:
        keys: Keys to press together (e.g., 'ctrl', 'c' for copy)
    """
    pygui.hotkey(*keys)
    return f"Pressed hotkey: {'+'.join(keys)}"


@mcp.tool()
def hold_key(key: str) -> str:
    """
    Hold down a key (doesn't release).
    
    Args:
        key: Key to hold down
    """
    pygui.keyDown(key)
    return f"Holding key: {key}"


@mcp.tool()
def release_key(key: str) -> str:
    """
    Release a held key.
    
    Args:
        key: Key to release
    """
    pygui.keyUp(key)
    return f"Released key: {key}"


# ===== SCREEN TOOLS =====

@mcp.tool()
def get_screen_size() -> dict:
    """Get the screen width and height."""
    width, height = pygui.size()
    return {"width": width, "height": height}


@mcp.tool()
def take_screenshot(filename: str = "screenshot.png") -> str:
    """
    Take a screenshot and save it to a file.
    
    Args:
        filename: Path where to save the screenshot (default "screenshot.png")
    """
    screenshot = pygui.screenshot()
    screenshot.save(filename)
    return f"Screenshot saved to {filename}"


def main():
    """Run the MCP server."""
    os.environ['MCP_SERVER_PORT'] = '8000'
    mcp.run(transport='sse')
    print("MCP Mouse and Keyboard Control Server is running on port 8000...")

if __name__ == "__main__":
    main()
