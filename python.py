import cv2
import numpy as np
from picarx import Picarx
from picamera2 import Picamera2

# Inizializzazione PiCar-X e fotocamera
px = Picarx()
camera = Picamera2()
camera.preview_configuration.main.size = (640, 480)
camera.preview_configuration.main.format = "BGR888"
camera.preview_configuration.align()
camera.configure("preview")
camera.start()

# Inizializza salvataggio video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('lane_debug.avi', fourcc, 20.0, (640, 480))

def region_of_interest(img):
    mask = np.zeros_like(img)
    height, width = img.shape
    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def sliding_window_lane_detection(binary_warped, nwindows=9, margin=100, minpix=50):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
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
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    if len(left_lane_inds) == 0 or len(right_lane_inds) == 0:
        return None, None

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def compute_steering(left_fit, right_fit, shape):
    height, width = shape
    y_eval = height - 1
    left_x = left_fit[0]
    y_eval*2 + left_fit[1]*y_eval + left_fit[2]
    right_x = right_fit[0]
    y_eval*2 + right_fit[1]*y_eval + right_fit[2]
    lane_center = (left_x + right_x) / 2
    vehicle_center = width / 2
    deviation = lane_center - vehicle_center

    max_deviation = width / 2
    normalized_deviation = deviation / max_deviation
    angle = -normalized_deviation * 45  # sterzata da -45° a +45°
    return np.clip(angle, -45, 45)

def draw_lane_overlay(frame, left_fit, right_fit):
    ploty = np.linspace(0, frame.shape[0]-1, frame.shape[0])
    color_warp = np.zeros_like(frame)

    try:
        left_fitx = left_fit[0]
        ploty*2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]
        ploty*2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        return frame

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
    result = cv2.addWeighted(frame, 1, color_warp, 0.3, 0)
    return result

try:
    while True:
        frame = camera.capture_array()
        original = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        roi = region_of_interest(edges)

        left_fit, right_fit = sliding_window_lane_detection(roi)

        if left_fit is not None and right_fit is not None:
            angle = compute_steering(left_fit, right_fit, roi.shape)
            px.set_dir_servo_angle(angle)
            px.forward(30)

            annotated = draw_lane_overlay(original, left_fit, right_fit)

            height, width = frame.shape[:2]
            midpoint = width // 2
            y_eval = height - 1
            left_x = left_fit[0]
            y_eval*2 + left_fit[1]*y_eval + left_fit[2]
            right_x = right_fit[0]
            y_eval*2 + right_fit[1]*y_eval + right_fit[2]
            lane_center = int((left_x + right_x) / 2)
            deviation = lane_center - midpoint

            # Linee di riferimento
            cv2.line(annotated, (midpoint, height), (midpoint, height - 40), (255, 0, 0), 2)
            cv2.line(annotated, (lane_center, height), (lane_center, height - 40), (0, 255, 255), 2)

            # Testo di debug
            cv2.putText(annotated, f"Steering angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(annotated, f"Deviation: {deviation:+.1f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Lane Detection Debug", annotated)
            out.write(annotated)
        else:
            print("[WARN] Linee non trovate. Fermando il veicolo.")
            px.stop()
            cv2.imshow("Lane Detection Debug", frame)
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interruzione manuale del programma.")
finally:
    out.release()
    px.stop()
    cv2.destroyAllWindows()
    print("Risorse rilasciate correttamente.")