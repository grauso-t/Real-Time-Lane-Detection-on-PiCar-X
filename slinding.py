import cv2
import numpy as np
from picarx import Picarx
from picamera2 import Picamera2
import time

picarx = Picarx()
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()
time.sleep(2)

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

# Per salvataggio video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

last_direction = 0
running = True  # Stato del movimento

while True:
    frame = picam2.capture_array()
    frame = cv2.resize(frame, (640, 480))
    debug_frame = frame.copy()

    # ROI
    tl = (70, 220)
    bl = (0, 472)
    tr = (570, 220)
    br = (640, 472)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))
    debug_warp = transformed_frame.copy()

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

    # Sliding windows
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    y = 472
    window_height = 40
    left_x, right_x = [], []

    while y > 0:
        top_y = y - window_height
        bottom_y = y

        # Sinistra
        cv2.rectangle(debug_warp, (left_base-50, top_y), (left_base+50, bottom_y), (255, 0, 255), 2)
        left_win = mask[top_y:bottom_y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(left_win, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                left_base = left_base - 50 + cx
                left_x.append(left_base)

        # Destra
        cv2.rectangle(debug_warp, (right_base-50, top_y), (right_base+50, bottom_y), (0, 255, 0), 2)
        right_win = mask[top_y:bottom_y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(right_win, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                right_base = right_base - 50 + cx
                right_x.append(right_base)

        y -= window_height

    # Calcolo centro corsia
    frame_center = 320
    if left_x and right_x:
        lane_center = (int(np.mean(left_x)) + int(np.mean(right_x))) // 2
    elif left_x:
        lane_center = int(np.mean(left_x)) + 100
    elif right_x:
        lane_center = int(np.mean(right_x)) - 100
    else:
        lane_center = frame_center

    error = lane_center - frame_center
    angle = -int(error / 3)
    last_direction = angle

    print(f"Lane center: {lane_center}, Error: {error}, Angle: {angle}")

    if running:
        picarx.forward(20)
        picarx.set_dir_servo_angle(angle)
    else:
        picarx.stop()

    # Visualizzazioni
    cv2.line(debug_warp, (lane_center, 0), (lane_center, 480), (0, 0, 255), 2)
    cv2.imshow("Frame", debug_frame)
    cv2.imshow("Warp", debug_warp)
    cv2.imshow("Mask", mask)
    out.write(debug_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        running = False
    elif key == ord('r'):
        running = True

picarx.stop()
cv2.destroyAllWindows()
out.release()