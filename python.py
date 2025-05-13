import cv2
import numpy as np
from picarx import Picarx
from time import sleep

# Setup
px = Picarx()
px.forward(0)  # Ferma il veicolo all'inizio

# Parametri sliding window
n_windows = 9
margin = 50
minpix = 50
buffer_size = 5  # Per media mobile

# Buffers per media mobile
left_x_buffer = []
right_x_buffer = []

# Funzione per filtrare con media mobile
def moving_average(buffer, new_value, size):
    buffer.append(new_value)
    if len(buffer) > size:
        buffer.pop(0)
    return int(np.mean(buffer))

# Funzione per maschera ROI
def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),
        (0, int(height*0.6)),
        (width, int(height*0.6)),
        (width, height),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

# Funzione per prospettiva
def warp_perspective(img):
    height, width = img.shape[:2]
    src = np.float32([
        [width*0.2, height*0.6],
        [width*0.8, height*0.6],
        [width*0.1, height],
        [width*0.9, height]
    ])
    dst = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

# Funzione per trovare le linee con sliding windows
def find_lane_lines(binary_img):
    histogram = np.sum(binary_img[binary_img.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = binary_img.shape[0] // n_windows
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_x_current = left_base
    right_x_current = right_base

    left_lane_inds = []
    right_lane_inds = []

    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

    for window in range(n_windows):
        win_y_low = binary_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_img.shape[0] - window * window_height
        win_xleft_low = left_x_current - margin
        win_xleft_high = left_x_current + margin
        win_xright_low = right_x_current - margin
        win_xright_high = right_x_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            left_x_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_x_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    left_x = nonzerox[left_lane_inds]
    right_x = nonzerox[right_lane_inds]

    # Sicurezza: se troppi pochi punti, considera non affidabile
    if len(left_x) < 100 or len(right_x) < 100:
        return None, out_img

    left_mean = int(np.mean(left_x))
    right_mean = int(np.mean(right_x))

    left_filtered = moving_average(left_x_buffer, left_mean, buffer_size)
    right_filtered = moving_average(right_x_buffer, right_mean, buffer_size)

    center = (left_filtered + right_filtered) // 2
    return center, out_img

# Loop principale
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = region_of_interest(frame)
        warped = warp_perspective(roi)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        center, debug_img = find_lane_lines(binary)

        if center is not None:
            img_center = binary.shape[1] // 2
            error = img_center - center
            if abs(error) < 20:
                px.set_dir_servo_angle(0)
            elif error > 0:
                px.set_dir_servo_angle(20)
            else:
                px.set_dir_servo_angle(-20)
            px.forward(20)
        else:
            px.forward(0)
            px.set_dir_servo_angle(0)

        # Finestre di output per debug
        cv2.imshow("Original", frame)
        cv2.imshow("ROI", roi)
        cv2.imshow("Warped", warped)
        cv2.imshow("Sliding Windows", debug_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrotto manualmente")

finally:
    cap.release()
    cv2.destroyAllWindows()
    px.forward(0)