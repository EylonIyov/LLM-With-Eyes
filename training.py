"""
Training script for teaching the vision model to identify UI elements
and control mouse/keyboard through MCP.
"""

from openai import OpenAI
import pyautogui as pygui
import json
import re
import time
import requests

# Import functions from script.py
from script import (
    invoke_mouse_keyboard,
    capture_screenshot,
    encode_image_to_base64
)


def train_move_to_location(target_description, x_coord=None, y_coord=None, count_id=0):
    """
    Train the model to move the mouse to a specific location by:
    1. Taking a screenshot
    2. Showing it to the vision model
    3. Asking it to identify where to click
    4. Executing the move
    
    Args:
        target_description (str): What to look for ("the red button", "the search bar", etc.)
        x_coord (int, optional): If provided, tell the model this is the target X
        y_coord (int, optional): If provided, tell the model this is the target Y
        count_id (int): Screenshot counter
        
    Returns:
        dict: Response from the model including coordinates
    """
    
    # Capture screenshot
    image_path = capture_screenshot(count_id)
    base64_image = encode_image_to_base64(image_path)
    
    # Build the prompt
    if x_coord is not None and y_coord is not None:
        prompt = f"""Look at this screenshot. I want you to move the mouse to {target_description}.
The correct coordinates are X={x_coord}, Y={y_coord}.

Please:
1. Describe what you see at those coordinates
2. Explain why those coordinates make sense for "{target_description}"
3. Then use the move_mouse tool to move there

Respond in JSON format with your reasoning and the coordinates you'll use."""
    else:
        prompt = f"""Look at this screenshot. Find {target_description} and tell me the X,Y coordinates where I should move the mouse.

Analyze the image carefully and respond in JSON format like this:
{{
    "target_found": true/false,
    "description": "what you see",
    "x": coordinate,
    "y": coordinate,
    "reasoning": "why these coordinates"
}}"""
    
    # Send to vision model using standard chat completions endpoint
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    resp = client.chat.completions.create(
        model="qwen/qwen2.5-vl-7b",
        messages=[
            {
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
            }
        ],
        temperature=0.1
    )
    
    response_text = resp.choices[0].message.content
    print("Model Response:")
    print(response_text)
    
    # Try to parse coordinates from response
    target_x = None
    target_y = None
    mcp_result = None
    
    try:
        # Try to parse as JSON first
        response_json = json.loads(response_text)
        target_x = response_json.get('x')
        target_y = response_json.get('y')
    except:
        # If not JSON, try to extract numbers
        coords = re.findall(r'[xX][:\s=]+(\d+).*?[yY][:\s=]+(\d+)', response_text)
        if coords:
            target_x, target_y = int(coords[0][0]), int(coords[0][1])
    
    # If we got coordinates, move the mouse using MCP
    if target_x and target_y:
        print(f"\n🎯 Moving mouse to ({target_x}, {target_y})...")
        move_instruction = f"Move the mouse to coordinates x={target_x}, y={target_y}"
        mcp_result = invoke_mouse_keyboard(instruction=move_instruction)
        print("MCP Response:")
        print(json.dumps(mcp_result, indent=2))
        
    return {
        "model_response": response_text,
        "coordinates": {"x": target_x, "y": target_y} if target_x else None,
        "mcp_result": mcp_result if target_x else None
    }


def create_training_dataset():
    """
    Create a training dataset by:
    1. Taking screenshots
    2. You manually provide correct coordinates
    3. Model learns to identify them
    """
    
    training_examples = [
        {
            "description": "the center of the screen",
            "x": pygui.size()[0] // 2,
            "y": pygui.size()[1] // 2
        },
        {
            "description": "the top-left corner",
            "x": 100,
            "y": 100
        },
        {
            "description": "the bottom-right area",
            "x": pygui.size()[0] - 200,
            "y": pygui.size()[1] - 200
        }
    ]
    
    results = []
    for i, example in enumerate(training_examples):
        print(f"\n{'='*60}")
        print(f"Training Example {i+1}: {example['description']}")
        print(f"{'='*60}")
        
        result = train_move_to_location(
            target_description=example['description'],
            x_coord=example['x'],
            y_coord=example['y'],
            count_id=i
        )
        results.append(result)
        
        # Wait a bit between examples
        time.sleep(2)
    
    return results


def interactive_training():
    """
    Interactive mode: You describe what to find, model tries to locate it
    """
    print("🎮 Interactive Training Mode")
    print("Describe what you want the model to find on screen.")
    print("Type 'quit' to exit.\n")
    
    count = 0
    while True:
        description = input("What should I find? > ")
        if description.lower() == 'quit':
            break
            
        result = train_move_to_location(
            target_description=description,
            count_id=count
        )
        count += 1
        print("\n" + "="*60 + "\n")


def supervised_training_session(target_description, x_coord, y_coord):
    """
    Run a single supervised training example where you provide the correct answer.
    
    Args:
        target_description (str): What the model should find
        x_coord (int): Correct X coordinate
        y_coord (int): Correct Y coordinate
    """
    print(f"\n{'='*60}")
    print(f"Supervised Training: {target_description}")
    print(f"Ground Truth: X={x_coord}, Y={y_coord}")
    print(f"{'='*60}\n")
    
    result = train_move_to_location(
        target_description=target_description,
        x_coord=x_coord,
        y_coord=y_coord,
        count_id=0
    )
    
    # Check if model got it right
    if result['coordinates']:
        model_x = result['coordinates']['x']
        model_y = result['coordinates']['y']
        error_x = abs(model_x - x_coord)
        error_y = abs(model_y - y_coord)
        
        print(f"\n📊 Training Results:")
        print(f"   Model predicted: ({model_x}, {model_y})")
        print(f"   Ground truth:    ({x_coord}, {y_coord})")
        print(f"   Error:           X±{error_x}, Y±{error_y}")
        
        if error_x < 50 and error_y < 50:
            print("   ✅ Good prediction!")
        else:
            print("   ❌ Needs improvement")
    
    return result


def test_find_and_click(target_description, click_type="left", count_id=0, verify_before_click=True):
    """
    Test function: Find an element on screen and click it.
    
    This function:
    1. Takes a screenshot
    2. Asks the vision model to find the specified element
    3. Moves the mouse to those coordinates
    4. Performs a click action
    
    Args:
        target_description (str): What to find and click (e.g., "the Chrome icon", "the Start button")
        click_type (str): Type of click - "left", "right", "double" (default "left")
        count_id (int): Screenshot counter
        verify_before_click (bool): If True, move mouse first and ask for confirmation before clicking
        
    Returns:
        dict: Results including coordinates and click confirmation
    """
    print(f"\n{'='*60}")
    print(f"🎯 Test: Find and Click - {target_description}")
    print(f"{'='*60}\n")
    
    # Step 1: Capture screenshot
    image_path = capture_screenshot(count_id)
    base64_image = encode_image_to_base64(image_path)
    
    # Step 2: Ask model to find the element
    screen_width, screen_height = pygui.size()
    prompt = f"""Look at this screenshot and find {target_description}.

IMPORTANT CONTEXT:
- Screen resolution: {screen_width}x{screen_height}
- The Windows taskbar is typically at the BOTTOM of the screen (y coordinate close to {screen_height})
- Desktop icons are usually on the left side
- Application windows are in the center

Analyze the image VERY carefully and tell me the exact X,Y coordinates where I should click.
Pay special attention to the actual location of the element you're identifying.

Respond in JSON format like this:
{{
    "target_found": true/false,
    "description": "what you see at that location",
    "x": coordinate,
    "y": coordinate,
    "reasoning": "why these coordinates (mention which part of screen: top/bottom/left/right)",
    "confidence": "high/medium/low"
}}"""
    
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    resp = client.chat.completions.create(
        model="qwen/qwen2.5-vl-7b",
        messages=[
            {
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
            }
        ],
        temperature=0.1
    )
    
    response_text = resp.choices[0].message.content
    print("Model Response:")
    print(response_text)
    print()
    
    # Step 3: Parse coordinates - try multiple methods
    target_x = None
    target_y = None
    target_found = True  # Assume found unless explicitly stated otherwise
    
    # Method 1: Try to parse as JSON
    try:
        # Remove markdown code blocks if present
        cleaned_response = response_text.strip()
        if cleaned_response.startswith('```'):
            # Extract content between ```json and ```
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned_response, re.DOTALL)
            if json_match:
                cleaned_response = json_match.group(1).strip()
        
        response_json = json.loads(cleaned_response)
        target_x = response_json.get('x')
        target_y = response_json.get('y')
        target_found = response_json.get('target_found', True)
        
        if not target_found:
            print("❌ Model could not find the target element")
            return {"success": False, "reason": "target_not_found", "model_response": response_text}
    except Exception as e:
        print(f"Note: JSON parsing failed ({e}), trying regex extraction...")
    
    # Method 2: Try multiple regex patterns
    if not target_x or not target_y:
        # Pattern 1: x=123, y=456 or x: 123, y: 456
        coords = re.findall(r'["\']?[xX]["\']?\s*[:\s=]+\s*(\d+).*?["\']?[yY]["\']?\s*[:\s=]+\s*(\d+)', response_text, re.DOTALL)
        if coords:
            target_x, target_y = int(coords[0][0]), int(coords[0][1])
            print(f"✓ Extracted coordinates using pattern 1: ({target_x}, {target_y})")
    
    if not target_x or not target_y:
        # Pattern 2: (x, y) format like (123, 456)
        coords = re.findall(r'\((\d+)\s*,\s*(\d+)\)', response_text)
        if coords:
            target_x, target_y = int(coords[0][0]), int(coords[0][1])
            print(f"✓ Extracted coordinates using pattern 2: ({target_x}, {target_y})")
    
    if not target_x or not target_y:
        # Pattern 3: Any two numbers that might be coordinates
        all_numbers = re.findall(r'\b(\d{2,4})\b', response_text)
        if len(all_numbers) >= 2:
            target_x, target_y = int(all_numbers[0]), int(all_numbers[1])
            print(f"⚠️  Guessed coordinates from numbers found: ({target_x}, {target_y})")
            print(f"   (This might not be accurate)")
    
    if not target_x or not target_y:
        print("❌ Could not extract coordinates from model response")
        print("\nDebugging info:")
        print(f"Response length: {len(response_text)} characters")
        print(f"First 200 chars: {response_text[:200]}")
        return {"success": False, "reason": "no_coordinates", "model_response": response_text}
    
    # Step 4: Move mouse to target
    print(f"🖱️  Moving mouse to ({target_x}, {target_y})...")
    move_instruction = f"Move the mouse to coordinates x={target_x}, y={target_y}"
    move_result = invoke_mouse_keyboard(instruction=move_instruction)
    time.sleep(0.5)  # Brief pause after moving
    
    # Step 4.5: Optional verification
    click_result = None
    if verify_before_click:
        print("\n⚠️  Mouse is now at the target location.")
        print("   Look at your screen - is the mouse pointing at the correct element?")
        confirm = input("   Proceed with click? (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("\n🚫 Click cancelled by user")
            return {
                "success": False,
                "reason": "user_cancelled",
                "target_description": target_description,
                "coordinates": {"x": target_x, "y": target_y},
                "model_response": response_text
            }
    
    # Step 5: Perform click action
    print(f"👆 Performing {click_type} click...")
    
    if click_type == "double":
        click_instruction = "Double click the mouse"
    elif click_type == "right":
        click_instruction = "Right click the mouse"
    else:
        click_instruction = "Click the left mouse button"
    
    click_result = invoke_mouse_keyboard(instruction=click_instruction)
    
    print("\n✅ Click action completed!")
    print(f"   Target: {target_description}")
    print(f"   Coordinates: ({target_x}, {target_y})")
    print(f"   Click type: {click_type}")
    
    return {
        "success": True,
        "target_description": target_description,
        "coordinates": {"x": target_x, "y": target_y},
        "click_type": click_type,
        "model_response": response_text,
        "move_result": move_result,
        "click_result": click_result
    }


if __name__ == "__main__":
    print("🚀 Vision-Based Mouse Control Training\n")
    print("Choose a training mode:")
    print("1. Predefined training dataset")
    print("2. Interactive training")
    print("3. Single supervised example")
    print("4. Find element without ground truth")
    print("5. Test: Find and click an element\n")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        print("\nRunning predefined training dataset...")
        create_training_dataset()
    
    elif choice == "2":
        interactive_training()
    
    elif choice == "3":
        print("\nEnter training details:")
        description = input("What should the model find? > ")
        x = int(input("Correct X coordinate: "))
        y = int(input("Correct Y coordinate: "))
        supervised_training_session(description, x, y)
    
    elif choice == "4":
        description = input("What should the model find? > ")
        train_move_to_location(target_description=description, count_id=0)
    
    elif choice == "5":
        print("\nTest: Find and Click")
        description = input("What should I find and click? > ")
        click_type = input("Click type (left/right/double) [default: left]: ").strip().lower()
        if click_type not in ["left", "right", "double"]:
            click_type = "left"
        
        verify = input("Verify before clicking? (y/n) [default: y]: ").strip().lower()
        verify_before_click = verify != 'n'
        
        print(f"\n⏳ Starting in 3 seconds... (move to your target window)")
        time.sleep(3)
        
        result = test_find_and_click(
            target_description=description,
            click_type=click_type,
            count_id=0,
            verify_before_click=verify_before_click
        )
        
        if result["success"]:
            print("\n🎉 Test completed successfully!")
        else:
            print(f"\n❌ Test failed: {result.get('reason', 'unknown')}")
    
    else:
        print("Invalid choice. Running default example...")
        train_move_to_location(
            target_description="the Windows taskbar",
            count_id=0
        )
