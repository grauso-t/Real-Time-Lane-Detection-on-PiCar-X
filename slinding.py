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

# Video writer per salvare il video con annotazioni
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('lane_debug.avi', fourcc, 20.0, (640, 480))

# Trackbar per calibrazione
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
last_valid_lane_center = 320
running = True

while True:
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (640, 480))

    # ROI per trasformazione prospettica
    tl = (70, 220)
    bl = (0, 472)
    tr = (570, 220)
    br = (640, 472)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    # Soglia HSV
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

    # Istogramma
    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding windows
    y = 472
    window_height = 40
    left_x, right_x = [], []
    valid_frame = True

    while y > 0:
        # finestre sliding
        left_win = mask[y - window_height:y, left_base - 50:left_base + 50]
        right_win = mask[y - window_height:y, right_base - 50:right_base + 50]

        # Visualizza finestre sliding
        cv2.rectangle(transformed_frame, (left_base - 50, y - window_height),
                      (left_base + 50, y), (255, 0, 255), 2)  # viola
        cv2.rectangle(transformed_frame, (right_base - 50, y - window_height),
                      (right_base + 50, y), (0, 255, 0), 2)  # verde

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

    # Validità del frame
    if left_x:
        mean_left = int(np.mean(left_x))
        if mean_left > 320:
            valid_frame = False
    if right_x:
        mean_right = int(np.mean(right_x))
        if mean_right < 320:
            valid_frame = False
    if left_x and right_x and abs(mean_right - mean_left) < 50:
        print("Linee troppo vicine, ignoro il frame")
        valid_frame = False

    # Calcolo centro corsia
    frame_center = 320
    if not valid_frame:
        lane_center = last_valid_lane_center
    else:
        if left_x and right_x:
            lane_center = (mean_left + mean_right) // 2
        elif left_x:
            lane_center = mean_left + 100
        elif right_x:
            lane_center = mean_right - 100
        else:
            lane_center = last_valid_lane_center
        last_valid_lane_center = lane_center

    # Calcolo errore e angolo di sterzo
    error = lane_center - frame_center
    angle = -int(error / 3)

    # Debug grafico
    cv2.line(transformed_frame, (lane_center, 480), (lane_center, 240), (255, 255, 255), 2)
    cv2.line(transformed_frame, (frame_center, 480), (frame_center, 240), (0, 255, 255), 2)

    # Controllo stato
    if running:
        picarx.forward(20)
        picarx.set_dir_servo_angle(angle)
        last_direction = angle
    else:
        picarx.stop()

    # Mostra e salva frame
    cv2.imshow("Frame", transformed_frame)
    out.write(transformed_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        running = False
        print("Macchina ferma")
    elif key == ord('r'):
        running = True
        print("Macchina in movimento")

# Pulizia
out.release()
cv2.destroyAllWindows()