import requests, sys, base64, time, io
from PIL import ImageGrab
from datetime import datetime
from tkinter import *

window = Tk()
window.image = PhotoImage(file="Cat/Idle/tile000.png")
window.wm_attributes("-topmost", True)
window.geometry("+600+200")
window.wm_attributes("-transparent", True)
# Set the root window background color to a transparent color
window.config(bg='systemTransparent')
window.overrideredirect(True)

# window.mainloop()
display1 = Label(window, image=window.image)
display1.grid(row=1, column=0, padx=0, pady=0)  #Display 1
display1.config(bg='systemTransparent')
display1.pack()
display1.mainloop()


def capture_screenshot():
    """Capture the current screen."""
    screenshot = ImageGrab.grab()
    return screenshot

def screenshot_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def send_to_ollama(image_base64, prompt):
    """Send screenshot to Ollama vision model."""
    try:
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

        for line in response.iter_lines():
            if line:
                data = line.json() if hasattr(line, 'json') else __import__('json').loads(line)
                print(data.get("response", ""), end="", flush=True)

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama on localhost:11434")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    while True:
        # screenshot = capture_screenshot()
        # image_base64 = screenshot_to_base64(screenshot)
        # prompt = "What is on my screen? Describe in detail."
        # send_to_ollama(image_base64, prompt)

        time.sleep(10)  # Capture every 10 seconds