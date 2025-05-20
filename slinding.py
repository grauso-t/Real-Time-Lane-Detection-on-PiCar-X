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
time.sleep(2)  # attesa per stabilizzare la fotocamera

# Trackbar per calibrazione HSV
def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Variabili per la sterzata fluida
last_direction = 0
current_angle = 0
alpha = 0.3  # smoothing factor
frame_center = 320

while True:
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (640, 480))

    # ROI e trasformazione prospettica
    tl = (70, 220)
    bl = (0, 472)
    tr = (570, 220)
    br = (640, 472)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    # Maschera HSV
    hsv = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower, upper)

    # Istogramma base
    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding windows
    y = 472
    window_height = 40
    left_x, right_x = [], []

    while y > 0:
        left_win = mask[y - window_height:y, max(0, left_base - 50):min(640, left_base + 50)]
        right_win = mask[y - window_height:y, max(0, right_base - 50):min(640, right_base + 50)]

        contours, _ = cv2.findContours(left_win, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                left_base = left_base - 50 + cx
                left_x.append(left_base)

        contours, _ = cv2.findContours(right_win, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                right_base = right_base - 50 + cx
                right_x.append(right_base)

        y -= window_height

    # Calcolo centro corsia e comando di guida
    if left_x and right_x:
        lane_center = (int(np.mean(left_x)) + int(np.mean(right_x))) // 2
        error = lane_center - frame_center
        if abs(error) > 10:
            angle = -int(error / 5)
        else:
            angle = 0
        last_direction = angle
    elif left_x and not right_x:
        lane_center = int(np.mean(left_x)) + 100
        error = lane_center - frame_center
        angle = -int(error / 5)
        last_direction = angle
    elif right_x and not left_x:
        lane_center = int(np.mean(right_x)) - 100
        error = lane_center - frame_center
        angle = -int(error / 5)
        last_direction = angle
    else:
        angle = last_direction

    # Smoothing dell’angolo e movimento
    smoothed_angle = int((1 - alpha) * current_angle + alpha * angle)
    current_angle = smoothed_angle
    picarx.forward(20)
    picarx.set_dir_servo_angle(smoothed_angle)

    # Mostra immagini per debug
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) == 27:  # ESC per uscire
        break

# Pulizia finale
picarx.stop()
cv2.destroyAllWindows()