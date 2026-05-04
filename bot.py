import requests, sys, base64, time, io
from PIL import ImageGrab, Image
from datetime import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QTimer
from pydantic import BaseModel
import sys
import json
import os
import math


app = QtWidgets.QApplication(sys.argv)

class Cat_Image(QObject):
    pixmap = QtGui.QPixmap("Cat/Idle/tile000.png")
    label = QtWidgets.QLabel()
    state = ""
    frame = 0
    frame_max = 1
    frame_min = 0

    def __init__(self):
        super().__init__()
        # Remove window borders and set it to stay on top
        self.label.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        # Allow the background to be transparent
        self.label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.label.setPixmap(self.pixmap)
        self.label.move(50, 50)

    def update_image(self, image_path):
        self.pixmap = QtGui.QPixmap(image_path)
        self.label.setPixmap(self.pixmap)
        self.label.show()

    def clear_image(self):
        self.label.clear()
        self.label.hide()

    def main(self):
        if(self.state == ""):
            self.clear_image()
            return
        frame_current = self.frame_min + self.frame
        zeroes = "".join("0" for i in range(2-math.floor(math.log10(max(frame_current, 1)))))
        image_path = f"Cat/{self.state}/tile{zeroes}{frame_current}.png"
        self.update_image(image_path)
        self.frame += 1
        self.frame %= self.frame_max

    def set_state(self, state):
        self.state = state
        if state == "":
            self.clear_image()
            return
        self.frame_max = int(len(os.listdir(f"Cat/{state}")))
        print(sorted(os.listdir(f"Cat/{state}")))
        self.frame_min = int(sorted(os.listdir(f"Cat/{state}"))[0].split("tile")[1].split(".png")[0])
        self.frame %= self.frame_max
        print("frame min ", self.frame_min)
        
class Productive(BaseModel):
    productive: bool
    reasoning: str

class Bot(QObject):    
    curr_screenshot = None
    prev_screenshot = None

    cat = None

    width = 0
    height = 0

    def __init__(self, cat):
        super().__init__()
        self.cat = cat
        self.cat.set_state("")
        # Get screen size from PyQt5
        screen = app.primaryScreen()
        self.width = screen.geometry().width()
        self.height = screen.geometry().height()

    def capture_screenshot(self):
        """Capture the current screen."""
        screenshot = ImageGrab.grab()
        return screenshot

    def screenshot_to_base64(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def check_has_changed(self, img1, img2, max_diff):
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")

        img1 = img1.load()
        img2 = img2.load()

        sum = 0

        for x in range(100):
            for y in range(100):
                a = x * self.width // 100
                b = y * self.height // 100
                color1 = img1[a, b]
                color2 = img2[a, b]
                sum += (abs(color1[0] - color2[0]) + abs(color1[1] - color2[1]) + abs(color1[2] - color2[2])) / 3

        avg_diff = sum / (100 * 100)
        return avg_diff > max_diff

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
                    "format": Productive.model_json_schema()
                },
                timeout=300,
                stream=True,
            )

            if response.status_code != 200:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                return None, None

            response_text = ""
            productive = None
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    response_text += data.get("response", "")
            
            # Try to parse the complete response as JSON
            try:
                parsed = json.loads(response_text)
                if isinstance(parsed, dict):
                    productive = parsed.get("productive", False)
                    reasoning = parsed.get("reasoning", response_text)
                else:
                    reasoning = response_text
            except json.JSONDecodeError:
                # If not valid JSON, treat entire response as reasoning
                reasoning = response_text
            
            return reasoning, productive

        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama on localhost:11434")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    def main(self):
        print("bot main")
        self.curr_screenshot = self.capture_screenshot()
        self.prev_screenshot = self.curr_screenshot if self.prev_screenshot is None else self.prev_screenshot

        if(self.check_has_changed(self.curr_screenshot, self.prev_screenshot, 20) or self.curr_screenshot == self.prev_screenshot):
            self.prev_screenshot = self.curr_screenshot
            self.cat.set_state("")
            self.curr_screenshot = self.screenshot_to_base64(self.curr_screenshot)
            response_text, productive = self.send_to_ollama(self.curr_screenshot, "Is this productive for a high school student who wants to get into MIT and therefore should either be doing his school work or working on STEM passion projects? It would be unproductive if it is not school work, which would be looking at blackbaud (the school site for lick-wilmerding), google docs, maybe forms, discussing classroom experiences, etc, or doing a stem passion project like coding or robotics or cad, etc.")
            
            if response_text:
                print(f"\nResponse: {response_text}")
                print(f"Productive: {productive}")


            if not productive:
                self.cat.set_state("Melt")
            else:
                self.cat.set_state("")


cat = Cat_Image()
bot = Bot(cat)

cat_main = QTimer()
cat_main.timeout.connect(lambda: cat.main())
cat_main.start(250)

bot_main = QTimer()
bot_main.timeout.connect(lambda: bot.main())
bot_main.start(60000)

sys.exit(app.exec())