from openai import OpenAI
import pyautogui as pygui
import base64
import requests, json

from PIL import ImageGrab as ImageGrab

def invoke_mouse_keyboard(model_name = "qwen/qwen2.5-vl-7b" , instruction = "Move the mouse to the middle of the screen"):
    """
    Executes the LM Studio MCP command to control the mouse.
    
    Args:
        instruction (str): Natural language instruction for moving the mouse
                          Example: "Move the mouse to the middle of the screen"
                                  "Move mouse to coordinates 100, 200"
                                  "Move cursor to top-left corner"
    
    Returns:
        dict: Response from LM Studio
        
    Example:
        >>> invoke_move_mouse("Move the mouse to the middle of the screen")
        >>> invoke_move_mouse("Move mouse to x=500, y=300")
    """
    
    # LM Studio endpoint
    url = "http://127.0.0.1:1234/v1/responses"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer lm-studio"
    }
    
    # Request payload - matches your curl command exactly
    payload = {
        "model": model_name,
        "tools": [{
            "type": "mcp",
            "server_label": "mouse-keyboard-control",
            "server_url": "http://localhost:8000/sse",
            "allowed_tools": ["get_mouse_position",
                                "move_mouse",
                                "move_mouse_relative",
                                "click_mouse",
                                "double_click",
                                "right_click",
                                "drag_mouse",
                                "scroll_mouse",
                                "type_text",
                                "press_key",
                                "press_hotkey",
                                "hold_key",
                                "release_key",
                                "get_screen_size",
                                "take_screenshot"]
        }],
        "input": instruction
    }
    
    # Make the request
    response = requests.post(url, headers=headers, json=payload)
    
    # Return the JSON response
    return response.json()



def example_middle_screen():
    """
    Example 1: Move mouse to middle of screen (same as your curl command)
    """
    print("=== Example 1: Move to middle of screen ===")
    result = invoke_mouse_keyboard("qwen/qwen2.5-vl-7b","Move the mouse to the middle of the screen")
    print(json.dumps(result, indent=2))
    print()


def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def health_function():
    """A simple health check function to verify API connectivity. """
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    resp = client.chat.completions.create(
        model="qwen2.5-vl-32b-instruct",
        messages=[{
            "role": "user",
    "content":[
      {"type":"text","text":"Describe the diagram and extract the key numbers."}
    ]
  }],
    temperature=0.1
)
    print(resp.choices[0].message.content)


def promot_model(prompt, model_name = "qwen/qwen2.5-vl-7b"):
    """A simple health check function to verify API connectivity. """
    client = OpenAI(base_url="http://localhost:1234/v1/responses", api_key="lm-studio")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
    "content":[
      {"type":"text","text":prompt}
    ]
  }],
    temperature=0.1
)
    print(resp.choices[0].message.content)
    

def capture_screenshot(count_id = 0):
    """Capture a screenshot and save it with a unique name."""
    pygui.screenshot(f"/screenshots/screenshot_pygui_{count_id}.png") # Save the screenshot with a unique name, will later be deleted, currently not deleted for debugging
    return f"/screenshots/screenshot_pygui_{count_id}.png" # Return the path of the saved screenshot


def send_screenshot_to_model(count_id = 0, prompt="Describe the image"):
    """""Send a captured screenshot to the model and print the response."""
    # Capture the screenshot
    image_path = capture_screenshot(count_id)
    
    # Encode the image to base64
    base64_image = encode_image_to_base64(image_path)
    # Send to the model
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
    # Print the model's response
    print(resp.choices[0].message.content)
    


    
    
if __name__ == "__main__":
    prompt = input("Enter your prompt for the model: ")
    try:
        promot_model(prompt, model_name = "qwen2.5-vl-32b-instruct")
    except Exception as e:
        print(f"Error occurred: {e}")
        health_function()
    #invoke_mouse_keyboard(prompt)
    
    
## Learn how to make the LLM use tools like mouse and keyboard to interact with the screen, by using the MCP API.

