import requests, sys, base64, time, io
from PIL import ImageGrab, Image
from datetime import datetime
from tkinter import *
import os
import math

class CatImage:
    window = Tk()
    state = ""
    frame = 0
    frame_max = 1
    frame_min = 0

    def __init__(self):
        self.window.wm_attributes("-topmost", True)
        self.window.wm_attributes("-transparent", True)
        self.window.config(bg='gray')
        self.window.overrideredirect(True)
        self.window.update_idletasks()
        self.window.geometry("+600+300")

    def update_image(self, image_path):
        self.window.image = PhotoImage(file=image_path)
        display1 = Label(self.window, image=self.window.image)
        display1.grid(row=1, column=0, padx=0, pady=0)  #Display 1
        display1.config(image=self.window.image, takefocus=1)
        display1.focus_set()
        self.window.update()
        display1.destroy()

    def main(self):
        if(self.state == ""):
            return
        frame_current = self.frame_min + self.frame
        zeroes = "".join("0" for i in range(2-math.floor(math.log10(max(frame_current, 1)))))
        image_path = f"Cat/{self.state}/tile{zeroes}{frame_current}.png"
        self.update_image(image_path)
        self.frame += 1
        self.frame %= self.frame_max

    def set_state(self, state):
        self.state = state
        # self.frame = 0 #convert to ints
        if(self.state == ""):
            self.window.withdraw()
            return
        else:
            self.window.deiconify()
        self.frame_max = int(len(os.listdir(f"Cat/{state}")))
        print(sorted(os.listdir(f"Cat/{state}")))
        self.frame_min = int(sorted(os.listdir(f"Cat/{state}"))[0].split("tile")[1].split(".png")[0])
        print("frame min ", self.frame_min)
        
class Bot:
    cat = CatImage()
    curr_screenshot = 0

    def __init__(self):
        self.cat.set_state("")

    def capture_screenshot(self):
        """Capture the current screen."""
        screenshot = ImageGrab.grab()
        return screenshot

    def screenshot_to_base64(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

    def send_to_ollama(self, image_base64, prompt):
        """Send screenshot to Ollama vision model."""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma4:e2b",
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": True,
                    "json": {"type":"object", "properties":{"response":"string", "productive":"boolean"}}
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
                    print(data.get("productive"))

        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama on localhost:11434")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    def main(self):
        self.curr_screenshot = self.capture_screenshot()
        self.curr_screenshot = self.screenshot_to_base64(self.curr_screenshot)
        self.send_to_ollama(self.curr_screenshot, "Is this productive?")


bot = Bot()
bot.main()
