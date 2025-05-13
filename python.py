import cv2
import numpy as np
from picarx import Picarx
from picamera2 import Picamera2
from filterpy.kalman import KalmanFilter

# Setup iniziale
px = Picarx()
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (320, 240)}))
picam2.start()

# Crea filtro di Kalman per stimare l'inclinazione delle linee
def create_kalman():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.], [0.]])  # Stato iniziale: [pendenza, offset]
    kf.F = np.array([[1., 0.], [0., 1.]])  # Matrice di transizione
    kf.H = np.array([[1., 0.]])  # Matrice di osservazione
    kf.P *= 1000.  # Incertezza iniziale
    kf.R = 10  # Rumore della misura
    kf.Q = 1e-3  # Rumore di processo
    return kf

kf_left = create_kalman()
kf_right = create_kalman()

# Parametri sliding window
n_windows = 9
margin = 30
minpix = 20

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)

    histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    midpoint = np.int32(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int32(binary.shape[0] // n_windows)
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(n_windows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

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
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = right_fit = None
    left_detected = right_detected = False

    if len(leftx) > 100:
        left_fit = np.polyfit(lefty, leftx, 1)
        kf_left.predict()
        kf_left.update(left_fit[0])
        left_detected = True
    else:
        kf_left.predict()

    if len(rightx) > 100:
        right_fit = np.polyfit(righty, rightx, 1)
        kf_right.predict()
        kf_right.update(right_fit[0])
        right_detected = True
    else:
        kf_right.predict()

    center = frame.shape[1] // 2
    deviation = 0

    if left_detected and right_detected:
        lane_center = (np.polyval(left_fit, frame.shape[0]) + np.polyval(right_fit, frame.shape[0])) / 2
        deviation = lane_center - center
    elif left_detected:
        lane_center = np.polyval(left_fit, frame.shape[0]) + margin
        deviation = lane_center - center
    elif right_detected:
        lane_center = np.polyval(right_fit, frame.shape[0]) - margin
        deviation = lane_center - center

    return deviation

# Main loop
try:
    while True:
        frame = picam2.capture_array()
        deviation = process_frame(frame)

        if deviation > 20:
            px.set_dir_servo_angle(-20)
        elif deviation < -20:
            px.set_dir_servo_angle(20)
        else:
            px.set_dir_servo_angle(0)

        px.forward(10)

except KeyboardInterrupt:
    px.stop()
    picam2.stop()