import cv2
import numpy as np
from picarx import Picarx
from simple_pid import PID
import time

# Inizializzazione
picarx = Picarx()
cap = cv2.VideoCapture(0)
pid = PID(Kp=0.5, Ki=0.01, Kd=0.1, setpoint=0)
pid.output_limits = (-30, 30)

frame_center = 320  # Mezzanotte per una camera 640x480
last_direction = 0

# Media mobile
def media_mobile(valori, N=5):
    if len(valori) < N:
        return int(np.mean(valori))
    return int(np.mean(valori[-N:]))

def prospettiva(frame):
    h, w = frame.shape[:2]
    src = np.float32([[0, h], [w, h], [w//2 + 50, h//2], [w//2 - 50, h//2]])
    dst = np.float32([[w//4, h], [w*3//4, h], [w*3//4, 0], [w//4, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (w, h))

def regione_interesse(img):
    mask = np.zeros_like(img)
    h, w = img.shape[:2]
    roi_corners = np.array([[(0, h), (w, h), (w, h//2), (0, h//2)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 255)
    return cv2.bitwise_and(img, mask)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Filtro per giallo/bianco
    lower = np.array([0, 0, 180])
    upper = np.array([255, 30, 255])
    mask = cv2.inRange(hsv, lower, upper)

    roi = regione_interesse(mask)
    bird_eye = prospettiva(roi)

    histogram = np.sum(bird_eye[bird_eye.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = np.int(bird_eye.shape[0] // nwindows)
    nonzero = bird_eye.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_current = left_base
    right_current = right_base
    margin = 50
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    out_img = np.dstack((bird_eye, bird_eye, bird_eye)) * 255

    for window in range(nwindows):
        win_y_low = bird_eye.shape[0] - (window + 1) * window_height
        win_y_high = bird_eye.shape[0] - window * window_height
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            left_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    left_x = nonzerox[left_lane_inds]
    right_x = nonzerox[right_lane_inds]

    left_x_filtered, right_x_filtered = [], []
    if len(left_x) >= 3:
        left_x_filtered.append(int(np.mean(left_x)))
    if len(right_x) >= 3:
        right_x_filtered.append(int(np.mean(right_x)))

    if left_x_filtered and right_x_filtered:
        lane_center = (media_mobile(left_x_filtered) + media_mobile(right_x_filtered)) // 2
    elif left_x_filtered:
        lane_center = media_mobile(left_x_filtered) + 100
    elif right_x_filtered:
        lane_center = media_mobile(right_x_filtered) - 100
    else:
        lane_center = None

    if lane_center is not None:
        error = lane_center - frame_center
        angle = -int(pid(error))
        last_direction = angle
        picarx.forward(20)
        picarx.set_dir_servo_angle(angle)
    else:
        picarx.forward(20)
        picarx.set_dir_servo_angle(last_direction)

    # Finestre di output
    cv2.imshow("Video originale", frame)
    cv2.imshow("Regione di interesse", roi)
    cv2.imshow("Visione prospettica (bird-eye)", bird_eye)
    cv2.imshow("Sliding Windows", out_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
picarx.stop()