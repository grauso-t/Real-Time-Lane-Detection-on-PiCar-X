import keyboard
import time
import os
import csv
from datetime import datetime
from threading import Thread

from picamera2 import Picamera2
from picarx import Picarx

import cv2

# Inizializzazioni
px = Picarx()
camera = Picamera2()
camera.configure(camera.create_still_configuration())
camera.start()

# Variabili
angle_x = 0  # angolo attuale impostato
capture_dir = "./images"
csv_file = "image_data.csv"

px.set_dir_servo_angle(0)

# Crea cartella immagini se non esiste
os.makedirs(capture_dir, exist_ok=True)

# Crea CSV con intestazione se non esiste
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "filename", "angle_x"])

# Mappa dei tasti e angoli
positions = {
    '1': -45,
    '2': -30,
    '3': -15,
    '4': 0,
    '5': 15,
    '6': 30,
    '7': 45,
}

def set_position(key):
    global angle_x
    angle_x = positions[key]
    px.set_servo_angle(angle_x)
    print(f"Angolo X impostato a {angle_x}°")

def move_forward():
    px.forward(5)
    print("Avanti a velocità 5")

def stop():
    px.stop()
    print("Stop")

def capture_images():
    while True:
        for i in range(5):  # 5 immagini al secondo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}.jpg"
            filepath = os.path.join(capture_dir, filename)

            image = camera.capture_array()
            cv2.imwrite(filepath, image)

            # Salva nel CSV
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, filename, angle_x])

            time.sleep(0.2)  # 5 Hz
        time.sleep(0.01)

# Avvia thread cattura immagini
capture_thread = Thread(target=capture_images, daemon=True)
capture_thread.start()

# Info comandi
print("Controlli:")
print("1-7 = angolo X (-45 a +45)")
print("8 = avanti a velocità 5")
print("9 = stop")

# Loop principale
try:
    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            key = event.name
            if key in positions:
                set_position(key)
            elif key == '8':
                move_forward()
            elif key == '9':
                stop()
        time.sleep(0.05)

except KeyboardInterrupt:
    px.stop()
    print("\nUscita manuale e stop")