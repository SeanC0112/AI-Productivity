#!/usr/bin/env python3
"""
Screenshot analyzer using Ollama vision model.
Captures your screen and sends it to Ollama for analysis.
"""

import requests
import base64
import sys
from PIL import ImageGrab
from datetime import datetime


def capture_screenshot():
    """Capture the current screen."""
    print("📸 Capturing screenshot...")
    screenshot = ImageGrab.grab()
    return screenshot


def screenshot_to_base64(image):
    """Convert PIL Image to base64 string."""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def send_to_ollama(image_base64, prompt):
    """Send screenshot to Ollama vision model."""
    try:
        print("🔍 Sending to Ollama...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma4:e2b",
                "prompt": prompt,
                "images": [image_base64],
                "stream": True,
            },
            timeout=300,
            stream=True,
        )

        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return

        print("\n📤 Response:")
        print("-" * 50)
        for line in response.iter_lines():
            if line:
                data = line.json() if hasattr(line, 'json') else __import__('json').loads(line)
                print(data.get("response", ""), end="", flush=True)
        print("\n" + "-" * 50)

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama on localhost:11434")
        print("   Make sure Ollama is running: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def interactive_mode():
    """Interactive screenshot analysis."""
    print("=" * 50)
    print("🖥️  OLLAMA SCREENSHOT ANALYZER")
    print("=" * 50)
    print("Commands:")
    print("  'screenshot' - Capture and analyze current screen")
    print("  'quit'       - Exit")
    print("=" * 50)

    while True:
        user_input = input("\n> ").strip().lower()

        if user_input == "quit" or user_input == "exit":
            print("👋 Goodbye!")
            break

        elif user_input == "screenshot":
            prompt = input("What do you want to know about the screenshot? > ").strip()
            if not prompt:
                prompt = "What's on the screen? Describe everything you see."

            screenshot = capture_screenshot()
            image_b64 = screenshot_to_base64(screenshot)
            print(f"✅ Screenshot captured ({screenshot.size[0]}x{screenshot.size[1]})")
            send_to_ollama(image_b64, prompt)

        else:
            print("❓ Unknown command. Try 'screenshot' or 'quit'")


def one_shot_mode(prompt=None):
    """Single screenshot analysis."""
    screenshot = capture_screenshot()
    image_b64 = screenshot_to_base64(screenshot)
    print(f"✅ Screenshot captured ({screenshot.size[0]}x{screenshot.size[1]})")

    if not prompt:
        prompt = "What's on the screen? Describe everything you see."

    send_to_ollama(image_b64, prompt)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # One-shot mode with custom prompt
        custom_prompt = " ".join(sys.argv[1:])
        one_shot_mode(custom_prompt)
    else:
        # Interactive mode
        interactive_mode()
