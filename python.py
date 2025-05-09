from picamera2 import Picamera2
from sunfounder_picarx import Picarx
import cv2
import numpy as np
import time

# Inizializzazione fotocamera e motori
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.preview_configuration.align()
picam2.start()

px = Picarx()
px.forward(0)  # Assicura che parta fermo

# Funzioni di elaborazione (come nel tuo codice originale)
def region_of_interest(img):
    height, width = img.shape[:2]
    roi_corners = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.4), int(height * 0.6)),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, roi_corners, 255)
    return cv2.bitwise_and(img, mask)

def average_slope_intercept(lines):
    left, right = [], []
    if lines is None:
        return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            angle = np.degrees(np.arctan(slope))
            if abs(angle) < 20 or abs(angle) > 160:
                continue
            intercept = y1 - slope * x1
            if slope < 0:
                left.append((slope, intercept))
            else:
                right.append((slope, intercept))
    left_avg = np.mean(left, axis=0) if left else None
    right_avg = np.mean(right, axis=0) if right else None
    return left_avg, right_avg

def make_line_points(y1, y2, line_params):
    if line_params is None:
        return None
    slope, intercept = line_params
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def draw_lines(img, left_line, right_line, left_lost=False, right_lost=False):
    overlay = img.copy()
    height = img.shape[0]

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        color = (0, 0, 255)
        label = 'SX' if not left_lost else 'SX PERSA'
        cv2.line(overlay, (x1, y1), (x2, y2), color, 10)
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        color = (0, 255, 0)
        label = 'DX' if not right_lost else 'DX PERSA'
        cv2.line(overlay, (x1, y1), (x2, y2), color, 10)
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if not left_lost and not right_lost:
        pts = np.array([
            [left_line[0], left_line[1]],
            [left_line[2], left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]]
        ])
        cv2.fillPoly(overlay, [pts], (255, 255, 0))

    return cv2.addWeighted(overlay, 0.8, img, 0.2, 0)

def dynamic_white_threshold(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[:, :, 2])
    if avg_brightness > 150:
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])
    else:
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 40, 255])
    return lower_white, upper_white

# Main loop
try:
    while True:
        frame = picam2.capture_array()
        lower_white, upper_white = dynamic_white_threshold(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        white_filtered = cv2.bitwise_and(frame, frame, mask=mask)

        gray = cv2.cvtColor(white_filtered, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        cropped = region_of_interest(edges)

        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, threshold=60, minLineLength=80, maxLineGap=30)
        left_avg, right_avg = average_slope_intercept(lines)

        if left_avg is not None and right_avg is not None:
            if left_avg[0] * right_avg[0] > 0:
                print("Linee sospette: pendenze simili.")
                left_avg = None
                right_avg = None

        height = frame.shape[0]
        y1, y2 = height, int(height * 0.6)
        left_line = make_line_points(y1, y2, left_avg)
        right_line = make_line_points(y1, y2, right_avg)

        left_lost = False
        right_lost = False

        if left_line is None and right_line is None:
            print("Entrambe le linee perse.")
            px.stop()
            continue

        if left_line is None:
            left_lost = True
            print("Linea sinistra persa.")
            width = frame.shape[1]
            left_line = np.array([50, height, 150, height - 50])

        if right_line is None:
            right_lost = True
            print("Linea destra persa.")
            width = frame.shape[1]
            right_line = np.array([width - 50, height, width - 150, height - 50])

        # Imposta la velocità dei motori
        if not left_lost and not right_lost:
            px.forward(20)
        else:
            px.forward(5)

        output = draw_lines(frame, left_line, right_line, left_lost, right_lost)
        cv2.imshow("Lane Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    px.stop()
    picam2.close()
    cv2.destroyAllWindows()
