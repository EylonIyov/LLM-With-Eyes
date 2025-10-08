"""
Grid-based coordinate grounding for improved UI element detection.
Overlays a chess-board style grid on screenshots to help the model with spatial reasoning.
"""

from PIL import Image, ImageDraw, ImageFont
import pyautogui as pygui
from script import encode_image_to_base64
import base64
from io import BytesIO


def create_grid_overlay(image_path, grid_size=8, save_path=None):
    """
    Add a chess-board style grid overlay to an image.
    
    Args:
        image_path: Path to the screenshot
        grid_size: Number of grid divisions (8 = 8x8 grid like chess)
        save_path: Where to save the gridded image
    
    Returns:
        Path to gridded image, grid_info dict
    """
    # Open image
    img = Image.open(image_path)
    width, height = img.size
    
    # Create a copy with transparency for overlay
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Calculate cell dimensions
    cell_width = width / grid_size
    cell_height = height / grid_size
    
    # Draw grid lines (semi-transparent)
    line_color = (0, 255, 0, 100)  # Green, semi-transparent
    line_width = 2
    
    # Vertical lines
    for i in range(grid_size + 1):
        x = int(i * cell_width)
        draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)
    
    # Horizontal lines
    for i in range(grid_size + 1):
        y = int(i * cell_height)
        draw.line([(0, y), (width, y)], fill=line_color, width=line_width)
    
    # Add labels (A-H for columns, 1-8 for rows)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    label_color = (255, 255, 0, 200)  # Yellow, semi-transparent
    
    # Column labels (A, B, C, ...)
    for i in range(grid_size):
        label = chr(65 + i)  # A=65 in ASCII
        x = int(i * cell_width + cell_width / 2)
        draw.text((x - 5, 5), label, fill=label_color, font=font)
        draw.text((x - 5, height - 25), label, fill=label_color, font=font)
    
    # Row labels (1, 2, 3, ...)
    for i in range(grid_size):
        label = str(i + 1)
        y = int(i * cell_height + cell_height / 2)
        draw.text((5, y - 10), label, fill=label_color, font=font)
        draw.text((width - 20, y - 10), label, fill=label_color, font=font)
    
    # Composite the overlay onto the original image
    img_rgba = img.convert('RGBA')
    result = Image.alpha_composite(img_rgba, overlay)
    result = result.convert('RGB')
    
    # Save
    if save_path is None:
        save_path = image_path.replace('.png', '_grid.png')
    result.save(save_path)
    
    # Create grid info for coordinate conversion
    grid_info = {
        "grid_size": grid_size,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "image_width": width,
        "image_height": height
    }
    
    return save_path, grid_info


def grid_to_coordinates(cell_notation, grid_info):
    """
    Convert grid cell notation (e.g., "D4") to pixel coordinates.
    
    Args:
        cell_notation: String like "D4" (column D, row 4)
        grid_info: Dictionary from create_grid_overlay
    
    Returns:
        (x, y) tuple of center coordinates
    """
    # Parse notation
    col_letter = cell_notation[0].upper()
    row_number = int(cell_notation[1:])
    
    # Convert to indices (A=0, B=1, etc.)
    col_index = ord(col_letter) - 65
    row_index = row_number - 1
    
    # Calculate center of cell
    x = int((col_index + 0.5) * grid_info['cell_width'])
    y = int((row_index + 0.5) * grid_info['cell_height'])
    
    return x, y


def coordinates_to_grid(x, y, grid_info):
    """
    Convert pixel coordinates to grid cell notation.
    
    Args:
        x, y: Pixel coordinates
        grid_info: Dictionary from create_grid_overlay
    
    Returns:
        Grid cell notation like "D4"
    """
    col_index = int(x / grid_info['cell_width'])
    row_index = int(y / grid_info['cell_height'])
    
    # Ensure within bounds
    col_index = max(0, min(col_index, grid_info['grid_size'] - 1))
    row_index = max(0, min(row_index, grid_info['grid_size'] - 1))
    
    col_letter = chr(65 + col_index)
    row_number = row_index + 1
    
    return f"{col_letter}{row_number}"


def create_grid_prompt(target_description, grid_info, few_shot_examples=""):
    """
    Create a prompt that uses grid notation.
    
    Args:
        target_description: What to find
        grid_info: Grid information
        few_shot_examples: Optional examples
    
    Returns:
        Prompt string
    """
    grid_size = grid_info['grid_size']
    
    prompt = f"""You are an expert at identifying UI elements in screenshots.

This screenshot has a {grid_size}x{grid_size} GRID OVERLAY to help with positioning:
- Columns are labeled A-{chr(64 + grid_size)} (left to right)
- Rows are labeled 1-{grid_size} (top to bottom)
- Example: "D4" means column D, row 4

EXAMPLES:
{few_shot_examples if few_shot_examples else '''
"Start button is in cell A8" (bottom-left corner)
"Chrome icon is in cell E8" (bottom taskbar, center-left)
"Close button is in cell H1" (top-right corner)
'''}

YOUR TASK: Find {target_description}

INSTRUCTIONS:
1. Look at the grid overlay on the screenshot
2. Find the element anywhere on screen
3. Identify which grid cell(s) it's in
4. Report the PRIMARY grid cell where the element's center is located

Respond in JSON format:
{{
    "target_found": true/false,
    "grid_cell": "letter+number (e.g., D4)",
    "description": "what you see there",
    "reasoning": "why this cell contains the element",
    "confidence": "high/medium/low"
}}

IMPORTANT: Use the visible grid lines to determine the correct cell!
"""
    return prompt


def test_grid_based_finding(target_description, count_id=0, grid_size=8):
    """
    Test finding elements using grid-based approach.
    
    Args:
        target_description: What to find
        count_id: Screenshot counter
        grid_size: Grid divisions (default 8x8)
    
    Returns:
        Result dictionary with coordinates
    """
    from openai import OpenAI
    import json
    import re
    
    # Take screenshot
    screenshot_path = f"screenshot_pygui_{count_id}.png"
    pygui.screenshot(screenshot_path)
    
    # Add grid overlay
    print(f"📐 Adding {grid_size}x{grid_size} grid overlay...")
    gridded_path, grid_info = create_grid_overlay(screenshot_path, grid_size=grid_size)
    print(f"✓ Grid overlay created: {gridded_path}")
    
    # Encode gridded image
    with open(gridded_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Create grid-based prompt
    prompt = create_grid_prompt(target_description, grid_info)
    
    # Send to model
    print(f"\n🔍 Searching for: {target_description}")
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    resp = client.chat.completions.create(
        model="qwen/qwen2.5-vl-7b",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }],
        temperature=0.1
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
            grid_cell = response_json.get('grid_cell')
            description = response_json.get('description', 'N/A')
            reasoning = response_json.get('reasoning', 'N/A')
            confidence = response_json.get('confidence', 'N/A')
            
            # Convert grid cell to coordinates
            x, y = grid_to_coordinates(grid_cell, grid_info)
            
            print(f"\n✅ Found in grid cell: {grid_cell}")
            print(f"   Coordinates: ({x}, {y})")
            print(f"   Confidence: {confidence}")
            print(f"   Description: {description}")
            print(f"   Reasoning: {reasoning}")
            
            return {
                "success": True,
                "grid_cell": grid_cell,
                "x": x,
                "y": y,
                "confidence": confidence,
                "description": description,
                "reasoning": reasoning
            }
        else:
            print("\n❌ Element not found")
            return {"success": False, "reason": "not_found"}
    
    except Exception as e:
        print(f"\n❌ Error parsing response: {e}")
        return {"success": False, "reason": "parse_error", "error": str(e)}


if __name__ == "__main__":
    import time
    
    print("🎮 Grid-Based Element Detection Test\n")
    
    # Test parameters
    target = input("What should I find? > ")
    grid_size = input("Grid size (4/8/16/24) [default: 8]: ").strip()
    grid_size = int(grid_size) if grid_size else 8
    
    print(f"\n⏳ Taking screenshot in 3 seconds...")
    time.sleep(3)
    
    result = test_grid_based_finding(target, count_id=0, grid_size=grid_size)
    
    if result['success']:
        print(f"\n🎉 Success! Element found at grid cell {result['grid_cell']}")
        print(f"   Moving mouse to ({result['x']}, {result['y']})...")
        
        # Move mouse to demonstrate
        pygui.moveTo(result['x'], result['y'], duration=1)
    else:
        print(f"\n❌ Failed: {result.get('reason', 'unknown')}")
