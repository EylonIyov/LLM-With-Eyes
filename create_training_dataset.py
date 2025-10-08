"""
Script to create a training dataset for UI element detection.
Collects screenshots with human-labeled coordinates.
"""

import json
import pyautogui as pygui
from script import capture_screenshot, encode_image_to_base64
import time

def collect_training_example():
    """
    Interactive tool to collect training examples:
    1. You describe what you want to click
    2. You click on it manually
    3. Saves: screenshot + description + coordinates
    """
    
    training_data = []
    example_count = 0
    
    print("🎓 Training Data Collection Tool")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Describe a UI element you want to find")
    print("2. The tool will take a screenshot in 3 seconds")
    print("3. Click on the element you described")
    print("4. The coordinates will be recorded")
    print("\nType 'done' when finished.\n")
    
    while True:
        description = input(f"\nExample {example_count + 1} - Describe element (or 'done'): ").strip()
        
        if description.lower() == 'done':
            break
        
        print(f"⏳ Taking screenshot in 3 seconds...")
        time.sleep(3)
        
        # Capture screenshot
        screenshot_path = capture_screenshot(example_count)
        base64_image = encode_image_to_base64(screenshot_path)
        
        print("✓ Screenshot captured!")
        print(f"📸 Saved as: {screenshot_path}")
        print("\n👆 Now click on the element you described...")
        print("   Waiting for click...")
        
        # Wait for user to click
        time.sleep(1)
        initial_pos = pygui.position()
        
        while True:
            current_pos = pygui.position()
            # Detect if mouse moved significantly (click happened)
            if abs(current_pos[0] - initial_pos[0]) > 5 or abs(current_pos[1] - initial_pos[1]) > 5:
                # Wait a moment for user to settle the click
                time.sleep(0.5)
                final_pos = pygui.position()
                break
            time.sleep(0.1)
        
        x, y = final_pos
        
        print(f"✓ Recorded click at: ({x}, {y})")
        
        # Confirm
        confirm = input(f"   Correct? (y/n): ").strip().lower()
        
        if confirm == 'y':
            training_example = {
                "id": example_count,
                "description": description,
                "screenshot_path": screenshot_path,
                "screenshot_base64": base64_image,
                "x": x,
                "y": y,
                "screen_resolution": {
                    "width": pygui.size()[0],
                    "height": pygui.size()[1]
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            training_data.append(training_example)
            example_count += 1
            print(f"✅ Example {example_count} saved!")
        else:
            print("❌ Example discarded")
    
    # Save training data
    if training_data:
        output_file = "training_dataset.json"
        with open(output_file, 'w') as f:
            json.dump(training_data, indent=2, fp=f)
        
        print(f"\n🎉 Training dataset saved to: {output_file}")
        print(f"   Total examples: {len(training_data)}")
        
        # Print summary
        print("\n📊 Dataset Summary:")
        for example in training_data:
            print(f"   - {example['description']}: ({example['x']}, {example['y']})")
    else:
        print("\n⚠️  No training data collected")
    
    return training_data


def alternative_click_collector():
    """
    Alternative: Just click and describe afterward
    Faster for collecting many examples
    """
    
    print("🎓 Quick Training Data Collection")
    print("=" * 60)
    
    print("\n⏳ Taking screenshot in 3 seconds...")
    time.sleep(3)
    
    screenshot_path = capture_screenshot(0)
    print(f"✓ Screenshot saved: {screenshot_path}")
    
    training_data = []
    example_count = 0
    
    print("\nNow for each element you want to label:")
    print("1. Click on it")
    print("2. Press Enter")
    print("3. Describe what you clicked")
    print("\nType 'done' to finish.\n")
    
    while True:
        input("Press Enter after clicking an element (or Ctrl+C to stop)...")
        
        x, y = pygui.position()
        description = input(f"What is at ({x}, {y})? > ").strip()
        
        if description.lower() == 'done':
            break
        
        training_data.append({
            "id": example_count,
            "description": description,
            "screenshot_path": screenshot_path,
            "x": x,
            "y": y,
            "screen_resolution": {
                "width": pygui.size()[0],
                "height": pygui.size()[1]
            }
        })
        
        example_count += 1
        print(f"✅ Labeled: {description} at ({x}, {y})")
    
    # Save
    if training_data:
        output_file = f"training_dataset_quick_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(training_data, indent=2, fp=f)
        print(f"\n🎉 Saved {len(training_data)} examples to: {output_file}")
    
    return training_data


if __name__ == "__main__":
    print("Choose collection method:")
    print("1. Guided collection (screenshot per example)")
    print("2. Quick collection (one screenshot, multiple labels)")
    
    choice = input("\nChoice (1-2): ").strip()
    
    if choice == "1":
        collect_training_example()
    elif choice == "2":
        alternative_click_collector()
    else:
        print("Invalid choice")
