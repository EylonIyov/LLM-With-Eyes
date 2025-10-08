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
import os

# Import functions from script.py
from script import (
    invoke_mouse_keyboard,
    capture_screenshot,
    encode_image_to_base64
)

# Import advanced prompting techniques
from advanced_prompts import create_few_shot_prompt, create_region_prompt


# Load training examples if available
def load_training_examples():
    """Load existing training data to use as few-shot examples"""
    if os.path.exists("training_dataset.json"):
        with open("training_dataset.json", 'r') as f:
            return json.load(f)
    return []


def save_training_example(description, screenshot_path, correct_x, correct_y, model_prediction=None):
    """
    Save a training example to the dataset
    
    Args:
        description: What element was being found
        screenshot_path: Path to the screenshot
        correct_x, correct_y: Correct coordinates
        model_prediction: Tuple of (x, y) that model predicted (optional)
    """
    # Load existing data
    training_data = load_training_examples()
    
    # Get base64 of screenshot
    base64_image = encode_image_to_base64(screenshot_path)
    
    # Create new example
    new_example = {
        "id": len(training_data),
        "description": description,
        "screenshot_path": screenshot_path,
        "screenshot_base64": base64_image,
        "x": correct_x,
        "y": correct_y,
        "screen_resolution": {
            "width": pygui.size()[0],
            "height": pygui.size()[1]
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if model_prediction:
        new_example["model_prediction"] = {
            "x": model_prediction[0],
            "y": model_prediction[1],
            "error_x": abs(model_prediction[0] - correct_x),
            "error_y": abs(model_prediction[1] - correct_y)
        }
    
    # Append and save
    training_data.append(new_example)
    
    with open("training_dataset.json", 'w') as f:
        json.dump(training_data, indent=2, fp=f)
    
    # Reload global examples
    global TRAINING_EXAMPLES
    TRAINING_EXAMPLES = training_data
    
    print(f"   💾 Saved to training_dataset.json (now contains {len(training_data)} examples)")


TRAINING_EXAMPLES = load_training_examples()
if TRAINING_EXAMPLES:
    print(f"✓ Loaded {len(TRAINING_EXAMPLES)} training examples for few-shot learning")


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
    
    # Step 2: Build advanced prompt with few-shot examples
    screen_width, screen_height = pygui.size()
    
    print("💡 Using few-shot prompting with training examples")
    
    # Build few-shot examples from training data
    few_shot_examples = ""
    if TRAINING_EXAMPLES:
        examples_to_use = TRAINING_EXAMPLES[:3]  # Use up to 3 examples
        for i, ex in enumerate(examples_to_use, 1):
            few_shot_examples += f"""
EXAMPLE {i}:
Task: Find "{ex['description']}"
Answer: {{"target_found": true, "x": {ex['x']}, "y": {ex['y']}, "reasoning": "Located at the specified position"}}
"""
    else:
        # Default examples if no training data
        few_shot_examples = f"""
EXAMPLE 1:
Task: Find "the Start button"
Answer: {{"target_found": true, "x": 25, "y": {screen_height - 20}, "reasoning": "Windows Start button is at bottom-left corner of taskbar"}}

EXAMPLE 2:
Task: Find "the Chrome icon in taskbar"
Answer: {{"target_found": true, "x": 670, "y": {screen_height - 20}, "reasoning": "Chrome icon is in the taskbar at the bottom"}}

EXAMPLE 3:
Task: Find "the File Explorer icon"
Answer: {{"target_found": true, "x": 550, "y": {screen_height - 20}, "reasoning": "File Explorer icon is in the taskbar at the bottom"}}
"""
    
    prompt = f"""You are an expert at identifying UI elements in screenshots.
{few_shot_examples}
NOW YOUR TASK:
Screen resolution: {screen_width}x{screen_height}
Task: Find {target_description}

INSTRUCTIONS:
1. Scan the ENTIRE screenshot carefully
2. Look for ANY element that matches "{target_description}" - it could be ANYWHERE on screen
3. Common locations (but not limited to):
   - Taskbar icons: usually near bottom (y ≈ {screen_height - 30})
   - Desktop icons: usually on left side (x < 300)
   - Window controls: usually at top (y < 50)
   - But the element could be in a window, dialog box, or anywhere else!
4. When you find it, give the CENTER coordinates of that element
5. If you don't see it anywhere on the screenshot, only then say target_found: false

IMPORTANT: Don't limit your search to specific areas. Look everywhere on the screen.

Respond in JSON format:
{{
    "target_found": true/false,
    "description": "what you see at that location",
    "x": coordinate,
    "y": coordinate,
    "reasoning": "where you found it and why you're confident it matches",
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
    
    # Step 4.5: Optional verification with learning
    click_result = None
    if verify_before_click:
        print("\n⚠️  Mouse is now at the target location.")
        print("   Look at your screen - is the mouse pointing at the correct element?")
        confirm = input("   Proceed with click? (y/n/correct): ").strip().lower()
        
        if confirm == 'correct':
            # User wants to provide correct coordinates
            print("\n📝 Let's record the correct coordinates for training...")
            print("   Move your mouse to the CORRECT location and press Enter")
            input("   Press Enter when ready...")
            correct_x, correct_y = pygui.position()
            
            # Save as training example
            save_training_example(
                description=target_description,
                screenshot_path=image_path,
                correct_x=correct_x,
                correct_y=correct_y,
                model_prediction=(target_x, target_y)
            )
            
            print(f"\n✅ Training example saved! Correct: ({correct_x}, {correct_y}), Model predicted: ({target_x}, {target_y})")
            print("   This will help improve future predictions.")
            
            # Update coordinates to correct ones
            target_x, target_y = correct_x, correct_y
            
            # Move to correct location
            print(f"\n🖱️  Moving to correct location ({target_x}, {target_y})...")
            move_instruction = f"Move the mouse to coordinates x={target_x}, y={target_y}"
            move_result = invoke_mouse_keyboard(instruction=move_instruction)
            time.sleep(0.5)
            
        elif confirm != 'y':
            print("\n🚫 Click cancelled by user")
            
            # Ask if they want to provide correct coordinates
            save_correct = input("   Save correct coordinates for training? (y/n): ").strip().lower()
            if save_correct == 'y':
                print("   Move your mouse to the CORRECT location and press Enter")
                input("   Press Enter when ready...")
                correct_x, correct_y = pygui.position()
                
                save_training_example(
                    description=target_description,
                    screenshot_path=image_path,
                    correct_x=correct_x,
                    correct_y=correct_y,
                    model_prediction=(target_x, target_y)
                )
                print(f"   ✅ Training example saved!")
            
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
    
    # Show quick wins status
    print("📊 Quick Wins Status:")
    print(f"   ✓ Few-shot prompting: {'ENABLED' if TRAINING_EXAMPLES else 'ENABLED (using defaults)'}")
    print(f"   ✓ Region-based hints: ENABLED")
    print(f"   ✓ Verification mode: ENABLED")
    print(f"   ✓ Training data: {len(TRAINING_EXAMPLES)} examples loaded")
    print()
    
    print("Choose a training mode:")
    print("1. Predefined training dataset")
    print("2. Interactive training")
    print("3. Single supervised example")
    print("4. Find element without ground truth")
    print("5. Test: Find and click an element")
    print("6. View training statistics\n")
    
    choice = input("Enter choice (1-6): ").strip()
    
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
            
            # Ask if it was accurate
            accurate = input("\nWas the click accurate? (y/n): ").strip().lower()
            if accurate == 'y':
                # Save as positive training example
                current_x, current_y = pygui.position()
                save_training_example(
                    description=description,
                    screenshot_path=f"screenshot_pygui_0.png",
                    correct_x=current_x,
                    correct_y=current_y
                )
                print("✅ Saved as training example!")
        else:
            print(f"\n❌ Test failed: {result.get('reason', 'unknown')}")
    
    elif choice == "6":
        print("\n📊 Training Statistics")
        print("=" * 60)
        if not TRAINING_EXAMPLES:
            print("No training data collected yet.")
            print("\nRun option 5 and save corrections to build your dataset!")
        else:
            print(f"Total examples: {len(TRAINING_EXAMPLES)}")
            print(f"\nExamples:")
            for ex in TRAINING_EXAMPLES[:10]:  # Show first 10
                print(f"  • {ex['description']}: ({ex['x']}, {ex['y']})")
                if 'model_prediction' in ex:
                    error = ex['model_prediction']
                    print(f"    Error: X±{error['error_x']}, Y±{error['error_y']}")
            
            if len(TRAINING_EXAMPLES) > 10:
                print(f"  ... and {len(TRAINING_EXAMPLES) - 10} more")
            
            # Calculate average error if available
            errors = [ex.get('model_prediction', {}) for ex in TRAINING_EXAMPLES]
            errors = [e for e in errors if e]
            if errors:
                avg_error_x = sum(e['error_x'] for e in errors) / len(errors)
                avg_error_y = sum(e['error_y'] for e in errors) / len(errors)
                print(f"\nAverage error: X±{avg_error_x:.1f}px, Y±{avg_error_y:.1f}px")
    
    else:
        print("Invalid choice. Running default example...")
        train_move_to_location(
            target_description="the Windows taskbar",
            count_id=0
        )
