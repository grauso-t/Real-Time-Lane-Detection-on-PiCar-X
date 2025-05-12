from picamera2 import Picamera2
from picarx import Picarx
import time
import numpy as np
import cv2

def process_video():
    px = Picarx()
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": (640, 480)}))
    cam.start()
    time.sleep(2)

    last_left_fit = None
    last_right_fit = None
    offset_threshold = 30  # soglia per decidere se sterzare

    try:
        while True:
            frame = cam.capture_array()
            height, width = frame.shape[:2]
            roi_height_start = int(height * 0.55)
            roi_height_end = int(height * 0.95)

            roi = frame[roi_height_start:roi_height_end, 0:width]
            roi_height, roi_width = roi.shape[:2]

            # Filtro colore
            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
            white_mask = cv2.inRange(hls, (0, 170, 0), (255, 255, 255))
            yellow_mask = cv2.inRange(hls, (10, 40, 40), (50, 255, 255))
            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
            masked = cv2.bitwise_and(roi, roi, mask=combined_mask)
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

            histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
            midpoint = np.int32(histogram.shape[0]//2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            nonzero = binary.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            left_lane_inds = ((nonzerox < midpoint) & (binary[nonzeroy, nonzerox] > 0)).nonzero()[0]
            right_lane_inds = ((nonzerox >= midpoint) & (binary[nonzeroy, nonzerox] > 0)).nonzero()[0]

            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            left_detected = len(leftx) > 50
            right_detected = len(rightx) > 50

            if left_detected:
                left_fit = np.polyfit(lefty, leftx, 2)
                last_left_fit = left_fit
            elif last_left_fit is not None:
                left_fit = last_left_fit
                left_detected = True
            else:
                left_fit = None
                left_detected = False

            if right_detected:
                right_fit = np.polyfit(righty, rightx, 2)
                last_right_fit = right_fit
            elif last_right_fit is not None:
                right_fit = last_right_fit
                right_detected = True
            else:
                right_fit = None
                right_detected = False

            if left_detected and right_detected:
                ploty = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
                left_x = left_fit[0]*ploty[-1]**2 + left_fit[1]*ploty[-1] + left_fit[2]
                right_x = right_fit[0]*ploty[-1]**2 + right_fit[1]*ploty[-1] + right_fit[2]
                lane_center = (left_x + right_x) / 2
                car_center = roi_width / 2
                offset = lane_center - car_center

                # Sterzata proporzionale all'offset
                if abs(offset) > offset_threshold:
                    angle = np.clip(offset * 0.3, -30, 30)
                    px.set_dir_servo_angle(int(angle))
                else:
                    px.set_dir_servo_angle(0)

                # Velocità in base alla presenza delle linee
                px.set_motor_speed(20)
            elif left_detected or right_detected:
                # Solo una linea: curva -> riduci velocità
                px.set_motor_speed(5)
            else:
                # Nessuna linea trovata
                px.stop()
                print("Linee non rilevate.")
                continue

            # Mostra anteprima
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.stop()
        px.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
