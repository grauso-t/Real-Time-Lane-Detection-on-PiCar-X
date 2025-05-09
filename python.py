import cv2
import numpy as np
import time
import picar

# Inizializza PiCar-X (motori e servo sterzo)
picar.setup()
from picar import front_wheels, back_wheels

fw = front_wheels.Front_Wheels()
bw = back_wheels.Back_Wheels()

fw.center()
bw.speed = 0

# Parametri sliding windows
n_windows = 9
margin = 50
minpix = 50

def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    window_height = np.int32(binary.shape[0] // n_windows)
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    out_img = np.dstack((binary, binary, binary)) * 255  # Per disegno a colori

    for window in range(n_windows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Disegna finestre
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    rightx = nonzerox[right_lane_inds]

    img_center_x = frame.shape[1] // 2
    cv2.line(out_img, (img_center_x, 0), (img_center_x, frame.shape[0]), (255, 0, 0), 2)  # Asse centrale

    left_dist, right_dist = None, None

    if len(leftx) > 0:
        left_mean_x = np.int32(np.mean(leftx))
        left_dist = left_mean_x - img_center_x
        cv2.circle(out_img, (left_mean_x, frame.shape[0]//2), 8, (255, 0, 255), -1)

    if len(rightx) > 0:
        right_mean_x = np.int32(np.mean(rightx))
        right_dist = right_mean_x - img_center_x
        cv2.circle(out_img, (right_mean_x, frame.shape[0]//2), 8, (255, 0, 255), -1)

    # Decisione direzione
    if left_dist is not None and right_dist is not None:
        center_offset = (left_dist + right_dist) / 2
    elif left_dist is not None:
        center_offset = left_dist
    elif right_dist is not None:
        center_offset = right_dist
    else:
        center_offset = 0

    if center_offset < -15:
        direction = "DESTRA"
    elif center_offset > 15:
        direction = "SINISTRA"
    else:
        direction = "DRITTO"

    return out_img, left_dist, right_dist, center_offset, direction

def control_car(direction):
    if direction == "DESTRA":
        fw.turn(80)   # gira a destra (meno di 90)
        bw.forward()
        bw.speed = 30
    elif direction == "SINISTRA":
        fw.turn(100)  # gira a sinistra (più di 90)
        bw.forward()
        bw.speed = 30
    else:
        fw.center()
        bw.forward()
        bw.speed = 30

def add_info_overlay(frame, left_dist, right_dist, offset, direction):
    text = f"L dist: {left_dist} | R dist: {right_dist} | Offset: {offset:.1f}"
    cv2.putText(frame, text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(frame, f"Direzione: {direction}", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    return frame

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, left_dist, right_dist, offset, direction = process_image(frame)
        control_car(direction)
        output_frame = add_info_overlay(processed_frame, left_dist, right_dist, offset, direction)

        cv2.imshow("Lane Detection with Info", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    fw.center()
    bw.stop()