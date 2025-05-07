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

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
sleep(2)

frame = picam2.capture_array()
h, w = frame.shape[:2]
src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst_points = np.float32([[0, 0], [400, 0], [400, 600], [0, 600]])
perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
output_size = (400, 600)

# Picarx setup
px = Picarx()
px.set_dir_servo_angle(0)
px.forward(0)

# Kalman Filters and previous values
prev_left = None
prev_right = None
kalman_left = KalmanFilter1D()
kalman_right = KalmanFilter1D()
last_angle = 0  # For smoothing

# --- Utility Functions ---
def split_lines(lines, width):
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue  # Skip vertical lines
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.5:
            left_lines.append(line[0])
        elif slope > 0.5:
            right_lines.append(line[0])
    return left_lines, right_lines

def average_line(lines):
    if len(lines) == 0:
        return None
    x, y = [], []
    for x1, y1, x2, y2 in lines:
        x.extend([x1, x2])
        y.extend([y1, y2])
    poly = np.polyfit(y, x, 1)
    y1, y2 = 600, 300
    x1, x2 = int(poly[0]*y1 + poly[1]), int(poly[0]*y2 + poly[1])
    return (x1, y1, x2, y2)

def shorten_line(line, ratio=0.3):
    if line is None:
        return None
    x1, y1, x2, y2 = line
    x_mid = int(x1 + (x2 - x1) * ratio)
    y_mid = int(y1 + (y2 - y1) * ratio)
    return (x1, y1, x_mid, y_mid)

def detect_lanes(frame, prev_left, prev_right):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White lane mask (HLS)
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([180, 255, 255])
    mask_white = cv2.inRange(hls, lower_white, upper_white)

    # Yellow lane mask (HSV)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Region of Interest (ROI)
    mask = np.zeros_like(edges)
    roi_corners = np.array([[
        (0, output_size[1]), (0, int(output_size[1]*0.5)),
        (output_size[0], int(output_size[1]*0.5)), (output_size[0], output_size[1])
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=50)
    left_line, right_line = None, None
    overlay = np.zeros_like(frame)

    if lines is not None:
        left_lines, right_lines = split_lines(lines, output_size[0])
        left_line = average_line(left_lines)
        right_line = average_line(right_lines)

    # Fallback to previous lines if current lines not detected
    if left_line is None and prev_left is not None:
        left_line = shorten_line(prev_left, 0.3)
    if right_line is None and prev_right is not None:
        right_line = shorten_line(prev_right, 0.3)

    # Draw lines
    if left_line:
        cv2.line(overlay, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 5)
    if right_line:
        cv2.line(overlay, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 5)

    # Fill region between lines
    if left_line and right_line:
        area = np.array([[
            (left_line[0], left_line[1]), (left_line[2], left_line[3]),
            (right_line[2], right_line[3]), (right_line[0], right_line[1])
        ]], dtype=np.int32)
        cv2.fillPoly(overlay, area, (0, 255, 255))

    return overlay, left_line, right_line, combined_mask

# --- Autonomous Driving Function ---
def autonomous_drive():
    global prev_left, prev_right, last_angle

    max_angle = 45
    max_offset = output_size[0] // 2

    while True:
        frame = picam2.capture_array()
        bird_eye = cv2.warpPerspective(frame, perspective_matrix, output_size)
        overlay, left_line, right_line, _ = detect_lanes(bird_eye, prev_left, prev_right)
        combined_view = cv2.addWeighted(bird_eye, 1.0, overlay, 0.7, 0)

        prev_left = left_line
        prev_right = right_line

        x_left = kalman_left.update(left_line[2]) if left_line else int(kalman_left.predict())
        x_right = kalman_right.update(right_line[2]) if right_line else int(kalman_right.predict())

        if left_line and right_line:
            road_center = (x_left + x_right) // 2
            frame_center = output_size[0] // 2
            offset = road_center - frame_center
            angle = int((offset / max_offset) * max_angle)
        elif left_line or right_line:
            frame_center = output_size[0] // 2
            offset = (x_left + 50 - frame_center) if left_line else (x_right - 50 - frame_center)
            angle = int((offset / max_offset) * max_angle)
            angle = np.clip(angle, -15, 15)
        else:
            px.stop()
            continue

        # Smooth steering with exponential smoothing
        smoothed_angle = int(0.7 * last_angle + 0.3 * angle)
        last_angle = smoothed_angle

        px.set_dir_servo_angle(smoothed_angle)
        px.forward(20)

        cv2.imshow("Detected Lanes", overlay)
        cv2.imshow("Combined View", combined_view)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    px.stop()

# --- Manual Control Function ---
def manual_control():
    print("Manual control active: Use W/A/S/D to drive, Q to quit.")
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

# --- Mode Selection ---
print("Press '8' for autonomous driving or '9' for manual control.")
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('8'):
        autonomous_drive()
        break
    elif key == ord('9'):
        manual_control()
        break
    elif key == ord('q'):
        print("Exiting...")
        break

# Cleanup
px.stop()
picam2.close()
cv2.destroyAllWindows()