import neural_network.index as neural_network
import neural_network.indexClass as nn 
import asyncio
import lcd
import database
import buzzer
import camera
import main as g
import cv2

# Create the recognizer instance once (initializing all models only one time)
recognizer = nn.LicensePlateRecognizer()

async def detect_plate():
    while True:
        await asyncio.sleep(0.1)
        plate, score = recognizer.license_plate_recognition_pipeline(g.global_frame)
        if score > 0.8:
            print(score, plate)
            cv2.imshow("detect_plate", g.global_frame)
            cv2.waitKey(500)
            has_log, type, severity = database.get_plate_from_database(plate)
            if has_log:
                camera.init_recording()
                str1 = f"{plate} {severity}              "
                str1[15] = "*"
                lcd.escrever_lcd(str1, f"{type}")
                await buzzer.buzzer()

def release():
    """Releases all RKNN resources."""
    recognizer.release()