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

# Trackbar per calibrazione
def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 180, 255, nothing) # Aumentato valore iniziale per V
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 80, 255, nothing) # Aumentata tolleranza per S
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Parametri per smorzamento e controllo
STEERING_SMOOTHING_FACTOR = 0.4
STEERING_GAIN = 5.0 
TOLERANCE_PX = 15
EXPECTED_LANE_WIDTH_PX = 300 # DA CALIBRARE! (es. 250-400 per 640px di larghezza trasformata)
SERVO_ANGLE_LIMIT = 35

# Variabili per stato/cronologia
smoothed_angle = 0.0
last_valid_lane_center = None
consecutive_no_lines_frames = 0
MAX_CONSECUTIVE_NO_LINES = 5

# Proprietà per il testo di debug
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color_info = (255, 255, 255) # Bianco
font_color_warning = (0, 255, 255) # Giallo per avvisi
font_color_error = (0, 0, 255)     # Rosso per errori gravi
thickness = 1
line_type = cv2.LINE_AA

# Colori per il debug visivo
color_roi_points = (0, 0, 255) # Rosso per punti ROI
color_left_window = (255, 100, 0) # Blu per finestra sinistra
color_right_window = (0, 100, 255) # Arancione/Rosso per finestra destra
color_left_points = (255, 180, 100) # Azzurro per punti linea sinistra
color_right_points = (100, 180, 255) # Rosa per punti linea destra
color_frame_center = (0, 255, 255) # Giallo per centro frame
color_lane_center_estimated = (255, 0, 255) # Magenta per centro corsia stimato
color_mean_lines = (0,255,0) # Verde per linee medie (se disegnate)


try:
    while True:
        frame_orig = picam2.capture_array()
        frame_height, frame_width = frame_orig.shape[:2]
        
        # Copia per disegnare ROI sul frame originale
        frame_with_roi = frame_orig.copy()

        # ROI (Region of Interest)
        roi_top_y = int(frame_height * 0.55) 
        roi_bottom_y = frame_height - 20 # Un po' più su dal fondo per evitare bordi
        
        roi_top_left_x = int(frame_width * 0.15) # Stringi un po'
        roi_top_right_x = int(frame_width * 0.85)
        roi_bottom_left_x = int(frame_width * 0.05)
        roi_bottom_right_x = int(frame_width * 0.95)

        tl = (roi_top_left_x, roi_top_y)
        bl = (roi_bottom_left_x, roi_bottom_y)
        tr = (roi_top_right_x, roi_top_y)
        br = (roi_bottom_right_x, roi_bottom_y)

        # Disegna i punti ROI sul frame originale
        for pt in [tl, bl, tr, br]:
            cv2.circle(frame_with_roi, pt, 7, color_roi_points, -1)
        # Disegna il poligono ROI
        cv2.polylines(frame_with_roi, [np.array([tl,tr,br,bl], dtype=np.int32)], isClosed=True, color=(0,255,0), thickness=2)


        pts1 = np.float32([tl, bl, tr, br])
        transformed_width, transformed_height = 640, 480
        pts2 = np.float32([[0,0], [0,transformed_height], [transformed_width,0], [transformed_width,transformed_height]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv2.warpPerspective(frame_orig, matrix, (transformed_width, transformed_height))
        
        # Crea una copia a colori per il disegno di debug sulla vista trasformata
        transformed_frame_display = transformed_frame.copy()

        # Sogliatura HSV
        hsv = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        lower_white = np.array([l_h, l_s, l_v])
        upper_white = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Istogramma
        histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0]/2)
        left_base_candidate = np.argmax(histogram[:midpoint])
        right_base_candidate = np.argmax(histogram[midpoint:]) + midpoint

        # Sliding window
        num_windows = 10
        window_height = int(transformed_frame.shape[0] / num_windows)
        minpix = 50 
        window_half_width = 75 # Larghezza mezza finestra

        current_left_x_points = [] # Lista di tuple (x,y)
        current_right_x_points = []# Lista di tuple (x,y)
        
        current_left_x_base = left_base_candidate
        current_right_x_base = right_base_candidate
        
        debug_texts = [] # Lista per messaggi di debug testuali

        for window_idx in range(num_windows):
            win_y_low = transformed_frame.shape[0] - (window_idx + 1) * window_height
            win_y_high = transformed_frame.shape[0] - window_idx * window_height
            win_y_center = (win_y_low + win_y_high) // 2
            
            # Finestra sinistra
            win_xleft_low = current_left_x_base - window_half_width
            win_xleft_high = current_left_x_base + window_half_width
            cv2.rectangle(transformed_frame_display,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),color_left_window, 1) 
            
            good_left_inds_y, good_left_inds_x = (mask[win_y_low:win_y_high, max(0,win_xleft_low):min(transformed_width,win_xleft_high)] == 255).nonzero()
            
            if len(good_left_inds_x) > minpix:
                new_left_x_base = max(0,win_xleft_low) + int(np.mean(good_left_inds_x))
                current_left_x_points.append((new_left_x_base, win_y_center))
                cv2.circle(transformed_frame_display, (new_left_x_base, win_y_center), 5, color_left_points, -1)
                current_left_x_base = new_left_x_base

            # Finestra destra
            win_xright_low = current_right_x_base - window_half_width
            win_xright_high = current_right_x_base + window_half_width
            cv2.rectangle(transformed_frame_display,(win_xright_low,win_y_low),(win_xright_high,win_y_high),color_right_window, 1)
            
            good_right_inds_y, good_right_inds_x = (mask[win_y_low:win_y_high, max(0, win_xright_low):min(transformed_width,win_xright_high)] == 255).nonzero()

            if len(good_right_inds_x) > minpix:
                new_right_x_base = max(0,win_xright_low) + int(np.mean(good_right_inds_x))
                current_right_x_points.append((new_right_x_base, win_y_center))
                cv2.circle(transformed_frame_display, (new_right_x_base, win_y_center), 5, color_right_points, -1)
                current_right_x_base = new_right_x_base
        
        # Logica di rilevamento e calcolo del centro corsia
        left_line_detected = len(current_left_x_points) > 2 
        right_line_detected = len(current_right_x_points) > 2
        debug_texts.append(f"Left Line: {'Yes' if left_line_detected else 'No'}")
        debug_texts.append(f"Right Line: {'Yes' if right_line_detected else 'No'}")

        frame_center_x = transformed_frame.shape[1] / 2.0
        cv2.line(transformed_frame_display, (int(frame_center_x), 0), (int(frame_center_x), transformed_height), color_frame_center, 1)


        current_estimated_lane_center = None
        left_x_mean, right_x_mean = None, None # Per visualizzazione

        if left_line_detected and right_line_detected:
            left_x_mean = np.mean([p[0] for p in current_left_x_points])
            right_x_mean = np.mean([p[0] for p in current_right_x_points])
            
            detected_width = right_x_mean - left_x_mean
            if left_x_mean < right_x_mean - (EXPECTED_LANE_WIDTH_PX * 0.3) and \
               detected_width > EXPECTED_LANE_WIDTH_PX * 0.5 and \
               detected_width < EXPECTED_LANE_WIDTH_PX * 1.5:
                current_estimated_lane_center = (left_x_mean + right_x_mean) / 2.0
                last_valid_lane_center = current_estimated_lane_center
                consecutive_no_lines_frames = 0
            else:
                debug_texts.append("Warn: L/R Crossed or Width Invalid")
                if last_valid_lane_center is not None:
                    current_estimated_lane_center = last_valid_lane_center
        
        elif left_line_detected:
            left_x_mean = np.mean([p[0] for p in current_left_x_points])
            if left_x_mean < frame_center_x - (EXPECTED_LANE_WIDTH_PX * 0.1):
                current_estimated_lane_center = left_x_mean + EXPECTED_LANE_WIDTH_PX / 2.0
                # last_valid_lane_center = current_estimated_lane_center # Aggiorna con più cautela se solo una linea
                consecutive_no_lines_frames = 0
            else:
                debug_texts.append("Warn: Left Line Too Far Right")
                if last_valid_lane_center is not None:
                    current_estimated_lane_center = last_valid_lane_center

        elif right_line_detected:
            right_x_mean = np.mean([p[0] for p in current_right_x_points])
            if right_x_mean > frame_center_x + (EXPECTED_LANE_WIDTH_PX * 0.1):
                current_estimated_lane_center = right_x_mean - EXPECTED_LANE_WIDTH_PX / 2.0
                # last_valid_lane_center = current_estimated_lane_center # Aggiorna con più cautela
                consecutive_no_lines_frames = 0
            else:
                debug_texts.append("Warn: Right Line Too Far Left")
                if last_valid_lane_center is not None:
                    current_estimated_lane_center = last_valid_lane_center
        
        else: 
            consecutive_no_lines_frames += 1
            debug_texts.append(f"No Lines! ({consecutive_no_lines_frames})")
            if last_valid_lane_center is not None and consecutive_no_lines_frames < MAX_CONSECUTIVE_NO_LINES:
                current_estimated_lane_center = last_valid_lane_center
        
        # Disegna linee medie rilevate (opzionale)
        # if left_x_mean is not None:
        #     cv2.line(transformed_frame_display, (int(left_x_mean), 0), (int(left_x_mean), transformed_height), color_mean_lines, 1)
        # if right_x_mean is not None:
        #     cv2.line(transformed_frame_display, (int(right_x_mean), 0), (int(right_x_mean), transformed_height), color_mean_lines, 1)

        # Disegna il centro della corsia stimato
        if current_estimated_lane_center is not None:
            cv2.line(transformed_frame_display, (int(current_estimated_lane_center), 0), (int(current_estimated_lane_center), transformed_height), color_lane_center_estimated, 2)
            cv2.circle(transformed_frame_display, (int(current_estimated_lane_center), transformed_height - 25), 10, color_lane_center_estimated, -1)
            debug_texts.append(f"Est. Center: {current_estimated_lane_center:.1f}px")


        # Calcolo dello sterzo
        target_angle = 0.0
        error_px = 0.0

        if current_estimated_lane_center is not None:
            error_px = current_estimated_lane_center - frame_center_x
            if abs(error_px) < TOLERANCE_PX:
                target_angle = 0.0 
            else:
                target_angle = -error_px / STEERING_GAIN
        else:
            debug_texts.append("No Center -> Target Angle = 0")
            pass # target_angle rimane 0

        smoothed_angle = (STEERING_SMOOTHING_FACTOR * target_angle) + \
                         ((1.0 - STEERING_SMOOTHING_FACTOR) * smoothed_angle)
        final_angle_degrees = np.clip(smoothed_angle, -SERVO_ANGLE_LIMIT, SERVO_ANGLE_LIMIT)
        
        picarx.forward(5)
        picarx.set_dir_servo_angle(final_angle_degrees)

        # Aggiungi informazioni testuali di sterzata
        debug_texts.append(f"Error: {error_px:.1f} px")
        debug_texts.append(f"Target Angle: {target_angle:.1f} deg")
        debug_texts.append(f"Final Angle: {final_angle_degrees:.1f} deg")
        
        # Mostra i testi di debug sull'immagine trasformata
        y_offset = 20
        for i, text in enumerate(debug_texts):
            text_color = font_color_info
            if "Warn" in text: text_color = font_color_warning
            if "No Lines!" in text and consecutive_no_lines_frames > 0 : text_color = font_color_error
            if "No Center" in text : text_color = font_color_error

            cv2.putText(transformed_frame_display, text, (10, y_offset + i * 20), font, font_scale, text_color, thickness, line_type)
        
        # Disegna la linea di guida (direzione)
        guide_line_length = 70
        guide_line_end_x = int(frame_center_x - guide_line_length * np.sin(np.deg2rad(final_angle_degrees))) # X e Y invertiti per angolo sterzo
        guide_line_end_y = int(transformed_height - guide_line_length * np.cos(np.deg2rad(final_angle_degrees)))
        cv2.line(transformed_frame_display, (int(frame_center_x), transformed_height -5), (guide_line_end_x, guide_line_end_y), (50,150,255), 3)


        # Mostra immagini
        cv2.imshow("Original Frame (with ROI)", frame_with_roi)
        cv2.imshow("Bird's Eye View (Debug)", transformed_frame_display)
        cv2.imshow("Lane Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("Stopping PicarX and closing windows.")
    picarx.stop()
    cv2.destroyAllWindows()
    if picam2.started:
        picam2.stop()