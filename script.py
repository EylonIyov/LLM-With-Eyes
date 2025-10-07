from openai import OpenAI
import pyautogui as pygui
import base64

from PIL import ImageGrab as ImageGrab



def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def health_function():
    """A simple health check function to verify API connectivity. """
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    resp = client.chat.completions.create(
        model="qwen/qwen2.5-vl-7b",
        messages=[{
            "role": "user",
    "content":[
      {"type":"text","text":"Describe the diagram and extract the key numbers."}
    ]
  }],
    temperature=0.1
)
    print(resp.choices[0].message.content)
    
    
def promot_model(prompt):
    """A simple health check function to verify API connectivity. """
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    resp = client.chat.completions.create(
        model="qwen/qwen2.5-vl-7b",
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
    pygui.screenshot(f"screenshot_pygui_{count_id}.png") # Save the screenshot with a unique name, will later be deleted, currently not deleted for debugging
    return f"screenshot_pygui_{count_id}.png" # Return the path of the saved screenshot
    

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
    
    
def let_qwen_use_tools():
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    resp = client.responses.create(
    model="qwen/qwen2.5-vl-7b",  # or whichever ID /v1/models returns
    input="Move the mouse 30px to the right, then click once.",
    tools=[{
        "type": "mcp",
        "server_label": "mouse-keyboard",             # must match mcp.json key
        "server_url": "http://127.0.0.1:9999/mcp",    # explicit is safest
        "allowed_tools": [
            "get_mouse_position","move_mouse","move_mouse_relative",
            "click_mouse","double_click","right_click","drag_mouse","scroll_mouse",
            "type_text","press_key","press_hotkey","hold_key","release_key",
            "get_screen_size","take_screenshot"
        ]
    }],
    tool_choice="auto"
    )

    print(resp.output_text)

    
    
if __name__ == "__main__":
        pygui.sleep(5)
        let_qwen_use_tools()


## Learn how to make the LLM use tools like mouse and keyboard to interact with the screen, by using the MCP API.