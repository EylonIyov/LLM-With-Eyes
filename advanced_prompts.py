"""
Advanced prompting techniques for better UI element detection
"""

# Technique 1: Few-shot learning with examples
def create_few_shot_prompt(target_description, base64_image, screen_width, screen_height):
    """
    Include examples in the prompt to teach the model the task
    """
    prompt = f"""You are an expert at identifying UI elements in screenshots.

EXAMPLE 1:
Task: Find "the Start button"
Answer: {{"target_found": true, "x": 25, "y": 760, "reasoning": "Windows Start button is at bottom-left corner of taskbar"}}

EXAMPLE 2:
Task: Find "the Chrome icon in taskbar"
Answer: {{"target_found": true, "x": 670, "y": 752, "reasoning": "Chrome icon is in the taskbar at the bottom, roughly center-left area"}}

NOW YOUR TASK:
Screen resolution: {screen_width}x{screen_height}
Task: Find {target_description}

GUIDELINES:
- Taskbar is at y ≈ {screen_height - 40} (bottom of screen)
- Desktop icons are typically at x < 200 (left side)
- Pay close attention to icon appearance and position
- If multiple matches exist, choose the most obvious one

Respond in JSON format:
{{
    "target_found": true/false,
    "description": "what you see",
    "x": coordinate,
    "y": coordinate,
    "reasoning": "detailed explanation with screen area mentioned",
    "confidence": "high/medium/low"
}}"""
    return prompt


# Technique 2: Chain-of-thought reasoning
def create_cot_prompt(target_description, screen_width, screen_height):
    """
    Ask model to think step-by-step
    """
    prompt = f"""Find {target_description} in this screenshot.

Think step-by-step:

STEP 1: What type of element am I looking for?
- Is it an icon, button, text, menu item, or window?
- Where do these typically appear on screen?

STEP 2: Scan the image by regions:
- Top bar (y: 0-100): Menu bars, title bars
- Left side (x: 0-200): Desktop icons, file explorer
- Bottom bar (y: {screen_height-50}-{screen_height}): Taskbar with pinned apps
- Center: Application windows

STEP 3: Identify the target:
- What color, shape, or text should I look for?
- What is its approximate position?

STEP 4: Determine precise coordinates:
- Center of the element for best click accuracy

Screen: {screen_width}x{screen_height}

Provide your analysis and final answer in JSON:
{{
    "step1_element_type": "...",
    "step2_likely_region": "...",
    "step3_visual_match": "...",
    "target_found": true/false,
    "x": coordinate,
    "y": coordinate,
    "confidence": "high/medium/low"
}}"""
    return prompt


# Technique 3: Self-correction prompting
def create_self_correction_prompt(target_description, first_attempt_x, first_attempt_y, screen_width, screen_height):
    """
    Ask model to verify its own answer
    """
    prompt = f"""You previously identified {target_description} at coordinates ({first_attempt_x}, {first_attempt_y}).

Screen resolution: {screen_width}x{screen_height}

VERIFICATION QUESTIONS:
1. Does this coordinate make sense for this element type?
   - Taskbar icons should have y ≈ {screen_height - 40}
   - Did you identify something else by mistake?

2. Is this in the correct screen region?
   - Top: y < 100
   - Bottom: y > {screen_height - 100}
   - Left: x < 200
   - Right: x > {screen_width - 200}

3. Could there be a more accurate position?

If your previous answer was wrong, provide the corrected coordinates.
If it was right, confirm it.

Respond in JSON:
{{
    "previous_correct": true/false,
    "correction_needed": "...",
    "final_x": coordinate,
    "final_y": coordinate,
    "confidence": "high/medium/low"
}}"""
    return prompt


# Technique 4: Region-based search
def create_region_prompt(target_description, expected_region, screen_width, screen_height):
    """
    Tell model which region to focus on
    """
    regions = {
        "taskbar": f"bottom area (y: {screen_height-60} to {screen_height})",
        "desktop": f"left side area (x: 0 to 200, y: 50 to {screen_height-100})",
        "top_bar": "top area (y: 0 to 50)",
        "system_tray": f"bottom-right corner (x: {screen_width-300} to {screen_width}, y: {screen_height-50} to {screen_height})"
    }
    
    region_desc = regions.get(expected_region, "anywhere on screen")
    
    prompt = f"""Find {target_description} in this screenshot.

FOCUS AREA: {region_desc}
Screen resolution: {screen_width}x{screen_height}

Only look in the specified focus area. Ignore matches outside this region.

Response in JSON:
{{
    "target_found": true/false,
    "x": coordinate,
    "y": coordinate,
    "in_focus_area": true/false,
    "confidence": "high/medium/low"
}}"""
    return prompt
