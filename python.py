import cv2
import numpy as np
from picarx import Picarx
from time import sleep
from picamera2 import Picamera2

class KalmanFilter1D:
    def __init__(self, process_variance=1, measurement_variance=2):
        self.x = 0.0
        self.P = 1.0
        self.Q = process_variance
        self.R = measurement_variance

    def predict(self):
        self.P += self.Q
        return self.x

    def update(self, measurement):
        K = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P = (1 - K) * self.P
        return self.x

# Inizializza la camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
sleep(2)

frame = picam2.capture_array()
cv2.imshow("Frame originale", frame)
h, w = frame.shape[:2]
src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst_points = np.float32([[0, 0], [400, 0], [400, 600], [0, 600]])
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
output_size = (400, 600)

prev_left = None
prev_right = None

kalman_left = KalmanFilter1D()
kalman_right = KalmanFilter1D()

px = Picarx()
px.set_dir_servo_angle(0)
px.forward(0)

def separa_linee(lines, width):
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.5:
            left_lines.append(line[0])
        elif slope > 0.5:
            right_lines.append(line[0])
    return left_lines, right_lines

def media_linea(linee):
    if len(linee) == 0:
        return None
    x = []
    y = []
    for x1, y1, x2, y2 in linee:
        x += [x1, x2]
        y += [y1, y2]
    poly = np.polyfit(y, x, 1)
    y1, y2 = 600, 300
    x1, x2 = int(poly[0]*y1 + poly[1]), int(poly[0]*y2 + poly[1])
    return (x1, y1, x2, y2)

def accorcia_linea(linea, percentuale=0.3):
    if linea is None:
        return None
    x1, y1, x2, y2 = linea
    x_mid = int(x1 + (x2 - x1) * percentuale)
    y_mid = int(y1 + (y2 - y1) * percentuale)
    return (x1, y1, x_mid, y_mid)

def rileva_corsie(frame, prev_left, prev_right):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white_hls = np.array([0, 200, 0])
    upper_white_hls = np.array([180, 255, 255])
    mask_white_hls = cv2.inRange(hls, lower_white_hls, upper_white_hls)

    lower_yellow_hsv = np.array([15, 80, 80])
    upper_yellow_hsv = np.array([40, 255, 255])
    mask_yellow_hsv = cv2.inRange(hsv, lower_yellow_hsv, upper_yellow_hsv)

    combined_mask = cv2.bitwise_or(mask_white_hls, mask_yellow_hsv)

    blur = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    roi_corners = np.array([[ (0, output_size[1]), (0, int(output_size[1] * 0.5)),
                              (output_size[0], int(output_size[1] * 0.5)), (output_size[0], output_size[1]) ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=50)

    line_image = np.zeros_like(frame)
    left_line, right_line = None, None

    if lines is not None:
        left_lines, right_lines = separa_linee(lines, output_size[0])
        left_line = media_linea(left_lines)
        right_line = media_linea(right_lines)

    if left_line is None and prev_left is not None:
        left_line = accorcia_linea(prev_left, 0.3)
    if right_line is None and prev_right is not None:
        right_line = accorcia_linea(prev_right, 0.3)

    if left_line:
        cv2.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 5)
    if right_line:
        cv2.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 5)

    if left_line and right_line:
        punti = np.array([[ (left_line[0], left_line[1]), (left_line[2], left_line[3]),
                            (right_line[2], right_line[3]), (right_line[0], right_line[1]) ]], dtype=np.int32)
        cv2.fillPoly(line_image, punti, (0, 255, 255))

    return line_image, left_line, right_line, mask_white_hls, mask_yellow_hsv, combined_mask

# --- Funzione modalità autonoma ---
def guida_autonoma():
    global prev_left, prev_right
    while True:
        frame = picam2.capture_array()
        bird_eye = cv2.warpPerspective(frame, matrix, output_size)
        overlay, prev_left, prev_right, _, _, combined_mask = rileva_corsie(bird_eye, prev_left, prev_right)
        combined = cv2.addWeighted(bird_eye, 1.0, overlay, 0.7, 0)

        max_angle = 45
        max_offset = output_size[0] // 2

        x_left = prev_left[2] if prev_left else int(kalman_left.predict())
        if not prev_left:
            kalman_left.update(x_left)

        x_right = prev_right[2] if prev_right else int(kalman_right.predict())
        if not prev_right:
            kalman_right.update(x_right)

        if prev_left and prev_right:
            centro_strada = (x_left + x_right) // 2
            centro_frame = output_size[0] // 2
            offset = centro_strada - centro_frame
            angle = int((offset / max_offset) * max_angle)
            angle = np.clip(angle, -max_angle, max_angle)
            px.set_dir_servo_angle(angle)
            px.forward(20)
        elif prev_left or prev_right:
            centro_frame = output_size[0] // 2
            offset = (x_left + 50 - centro_frame) if prev_left else (x_right - 50 - centro_frame)
            angle = int((offset / max_offset) * max_angle)
            angle = np.clip(angle, -15, 15)
            px.set_dir_servo_angle(angle)
            px.forward(10)
        else:
            px.stop()

        cv2.imshow("Corsie rilevate", overlay)
        cv2.imshow("Vista con sovrapposizione", combined)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    px.stop()

# --- Funzione controllo manuale ---
def controllo_wasd():
    print("Controllo manuale attivo: usa W/A/S/D per muovere, Q per uscire.")
    px.set_dir_servo_angle(0)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('w'):
            px.forward(20)
        elif key == ord('s'):
            px.backward(20)
        elif key == ord('a'):
            px.set_dir_servo_angle(-30)
            px.forward(20)
        elif key == ord('d'):
            px.set_dir_servo_angle(30)
            px.forward(20)
        elif key == ord('x'):
            px.stop()
        elif key == ord('q'):
            px.stop()
            break
        else:
            px.set_dir_servo_angle(0)

# --- Selettore modalità ---
print("Premi '8' per guida autonoma o '9' per controllo manuale WASD.")
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('8'):
        guida_autonoma()
        break
    elif key == ord('9'):
        controllo_wasd()
        break
    elif key == ord('q'):
        print("Uscita.")
        break

# Arresto e chiusura
px.stop()
picam2.close()
cv2.destroyAllWindows()