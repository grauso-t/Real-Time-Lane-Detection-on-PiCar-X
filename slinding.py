import cv2
import numpy as np
from picarx import Picarx
from picamera2 import Picamera2
import time

# Inizializza la macchina e la fotocamera
picarx = Picarx()
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()
time.sleep(2)

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

last_direction = 0

while True:
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (640, 480))

    # ROI e bird's eye
    tl, bl, tr, br = (70,220), (0,472), (570,220), (640,472)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    birdseye = cv2.warpPerspective(frame, matrix, (640, 480))

    # Maschera HSV
    hsv = cv2.cvtColor(birdseye, cv2.COLOR_BGR2HSV)
    lower = np.array([cv2.getTrackbarPos("L - H", "Trackbars"),
                      cv2.getTrackbarPos("L - S", "Trackbars"),
                      cv2.getTrackbarPos("L - V", "Trackbars")])
    upper = np.array([cv2.getTrackbarPos("U - H", "Trackbars"),
                      cv2.getTrackbarPos("U - S", "Trackbars"),
                      cv2.getTrackbarPos("U - V", "Trackbars")])
    mask = cv2.inRange(hsv, lower, upper)

    # Istogramma
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding window
    y = 472
    window_height = 40
    left_x, right_x, centers = [], [], []
    sliding_vis = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    while y > 0:
        lw = mask[y-window_height:y, left_base-50:left_base+50]
        rw = mask[y-window_height:y, right_base-50:right_base+50]

        # Linea sinistra
        contours, _ = cv2.findContours(lw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                left_base = left_base - 50 + cx
                left_x.append(left_base)

        # Linea destra
        contours, _ = cv2.findContours(rw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                right_base = right_base - 50 + cx
                right_x.append(right_base)

        # Visualizzazione sliding window
        cv2.rectangle(sliding_vis, (left_base-50, y-window_height), (left_base+50, y), (255,0,0), 2)
        cv2.rectangle(sliding_vis, (right_base-50, y-window_height), (right_base+50, y), (0,255,0), 2)
        if left_base and right_base:
            center = (left_base + right_base) // 2
            centers.append(center)
            cv2.circle(sliding_vis, (center, y - window_height//2), 4, (0,0,255), -1)

        y -= window_height

    # Comportamento guida
    frame_center = 320
    if left_x and right_x:
        lane_center = (int(np.mean(left_x)) + int(np.mean(right_x))) // 2
        centers.append(lane_center)
        error = lane_center - frame_center

        curve_strength = np.var(centers)
        curva = curve_strength > 3000  # soglia da calibrare
        angle = -int(error / (2 if curva else 4))
        speed = 10 if curva else 20

        last_direction = angle
        picarx.forward(speed)
        picarx.set_dir_servo_angle(angle)

    elif left_x and not right_x:
        lane_center = int(np.mean(left_x)) + 100
        error = lane_center - frame_center
        angle = -int(error / 3)
        last_direction = angle
        picarx.forward(15)
        picarx.set_dir_servo_angle(angle)

    elif right_x and not left_x:
        lane_center = int(np.mean(right_x)) - 100
        error = lane_center - frame_center
        angle = -int(error / 3)
        last_direction = angle
        picarx.forward(15)
        picarx.set_dir_servo_angle(angle)

    else:
        # fallback: continua dritto mantenendo ultima direzione
        picarx.forward(10)
        picarx.set_dir_servo_angle(last_direction)

    # Visualizzazione
    cv2.imshow("Bird's Eye", birdseye)
    cv2.imshow("Mask", mask)
    cv2.imshow("Sliding Window", sliding_vis)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()