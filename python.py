import cv2
import numpy as np
import time
from picamera2 import Picamera2
from picarx import Picarx


def lane_detection_control():
    """
    Acquisisce immagini dalla PiCamera2, effettua rilevamento di corsie e controlla PiCar-X.
    - Velocità 20% quando entrambe le linee sono rilevate.
    - Velocità 5% quando viene rilevata solo una linea.
    - Ferma il veicolo se non vengono rilevate linee.
    - Sterzata calcolata con angolo massimo di 40° in base alla posizione centrale della corsia.
    """
    # Inizializza PiCamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Lascia stabilizzare l'esposizione

    # Inizializza il robot PiCar-X
    px = Picarx()
    px.set_dir_servo_angle(0)  # Centro

    # Variabili per mantenere l'ultimo fit dei polinomi
    last_left_fit = None
    last_right_fit = None

    try:
        while True:
            # Acquisisci frame
            frame = picam2.capture_array()
            height, width = frame.shape[:2]

            # Definisci ROI: parte bassa del frame
            y_start = int(height * 0.55)
            y_end = int(height * 0.95)
            roi = frame[y_start:y_end, :]

            # Pre-elaborazione: blur + HLS + maschere
            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)

            # Maschera bianco
            lower_white = np.array([0, 170, 0], dtype=np.uint8)
            upper_white = np.array([255, 255, 255], dtype=np.uint8)
            white_mask = cv2.inRange(hls, lower_white, upper_white)

            # Maschera giallo
            lower_yellow = np.array([10, 40, 40], dtype=np.uint8)
            upper_yellow = np.array([50, 255, 255], dtype=np.uint8)
            yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
            masked = cv2.bitwise_and(roi, roi, mask=combined_mask)

            # Binario
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

            # Istogramma per trovare base delle linee
            histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
            midpoint = int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Parametri per sliding windows
            nwindows = 9
            window_height = int(binary.shape[0] / nwindows)
            margin = 100
            minpix = 50

            nonzero = binary.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            leftx_current = leftx_base
            rightx_current = rightx_base
            left_lane_inds = []
            right_lane_inds = []

            # Scorri finestre verticali
            for window in range(nwindows):
                win_y_low = binary.shape[0] - (window + 1) * window_height
                win_y_high = binary.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # Trova indici dei pixel bianchi
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

            # Concatena indici
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Estrai coordinate
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Determina rilevamento
            left_detected = len(leftx) > minpix
            right_detected = len(rightx) > minpix

            # Calcola fit polinomial se disponibili
            if left_detected:
                left_fit = np.polyfit(lefty, leftx, 2)
                last_left_fit = left_fit
            else:
                left_fit = last_left_fit

            if right_detected:
                right_fit = np.polyfit(righty, rightx, 2)
                last_right_fit = right_fit
            else:
                right_fit = last_right_fit

            # Calcola centro corsia e angolo sterzata
            if left_fit is not None and right_fit is not None:
                ploty = binary.shape[0] - 1
                left_x = int(left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2])
                right_x = int(right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2])
                lane_center = (left_x + right_x) // 2
                frame_center = binary.shape[1] // 2
                offset = lane_center - frame_center
                # Mappa offset in angolo -40..40 gradi
                max_angle = 40
                angle = int((offset / frame_center) * max_angle)
                angle = max(-max_angle, min(max_angle, angle))
            else:
                angle = 0  # Default centro

            # Imposta direzione
            px.set_dir_servo_angle(angle)

            # Decisione di movimento
            if left_detected and right_detected:
                px.set_motor_speed(20)
                px.forward()
            elif left_detected or right_detected:
                px.set_motor_speed(5)
                px.forward()
            else:
                px.stop()

            # Per debug video
            cv2.imshow("Lane Detection", roi)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrotto manualmente")

    finally:
        px.stop()
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    lane_detection_control()
