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
time.sleep(2)  # attesa per stabilizzare la fotocamera

# Trackbar per calibrazione
def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing) # Valore iniziale per V (luminosità) alto per linee bianche
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing) # Tolleranza per la saturazione
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Parametri per smorzamento e controllo
STEERING_SMOOTHING_FACTOR = 0.4  # Sperimenta con questo valore (es. 0.3 - 0.7)
STEERING_GAIN = 5.0             # Divisore per l'errore, più alto = sterzo meno aggressivo (originale era 3)
TOLERANCE_PX = 15               # Tolleranza in pixel per considerare l'auto dritta (aumentata da 5)
EXPECTED_LANE_WIDTH_PX = 300    # Larghezza attesa della corsia in pixel nell'immagine trasformata (DA CALIBRARE!)
                                # Se nel tuo codice originale usavi +/-100, potrebbe essere 2*100=200
                                # o se 100 era una stima approssimativa della distanza dal centro alla linea,
                                # allora la larghezza totale potrebbe essere ~300-400px. Regola questo valore!
SERVO_ANGLE_LIMIT = 35          # Limite angolo servo (gradi)

# Variabili per stato/cronologia
smoothed_angle = 0.0
last_valid_lane_center = None
consecutive_no_lines_frames = 0
MAX_CONSECUTIVE_NO_LINES = 5

try:
    while True:
        frame = picam2.capture_array()
        # frame = cv2.resize(frame, (640, 480)) # Già configurato in Picamera2

        # ROI (Region of Interest)
        # Assicurati che questi punti definiscano un trapezio che copra bene la corsia di fronte all'auto
        # tl = (70,180) #precedente (70,220)
        # bl = (0,472)
        # tr = (570,180) #precedente (570,220)
        # br = (640,472)
        
        # ROI un po' più stretto in alto e più focalizzato in basso
        height, width = frame.shape[:2]
        roi_top_y = int(height * 0.55) # Inizia a circa metà altezza dell'immagine
        roi_bottom_y = height - 10 # Fino quasi in fondo
        
        roi_top_left_x = int(width * 0.1)
        roi_top_right_x = int(width * 0.9)
        roi_bottom_left_x = int(width * 0.0)
        roi_bottom_right_x = int(width * 1.0)

        tl = (roi_top_left_x, roi_top_y)
        bl = (roi_bottom_left_x, roi_bottom_y)
        tr = (roi_top_right_x, roi_top_y)
        br = (roi_bottom_right_x, roi_bottom_y)


        # Disegna i punti ROI per verifica (opzionale)
        # for pt in [tl, bl, tr, br]:
        #     cv2.circle(frame, pt, 5, (0,0,255), -1)

        pts1 = np.float32([tl, bl, tr, br])
        # pts2 deve corrispondere alle dimensioni della transformed_frame desiderate
        transformed_width, transformed_height = 640, 480
        pts2 = np.float32([[0,0], [0,transformed_height], [transformed_width,0], [transformed_width,transformed_height]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv2.warpPerspective(frame, matrix, (transformed_width, transformed_height))

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

        # Istogramma per trovare la base delle linee
        # Considera solo la metà inferiore della maschera
        histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0]/2)
        
        left_base_candidate = np.argmax(histogram[:midpoint])
        right_base_candidate = np.argmax(histogram[midpoint:]) + midpoint

        # Sliding window
        num_windows = 10
        window_height = int(transformed_frame.shape[0] / num_windows)
        # Minimo numero di pixel trovati per ricentrare la finestra
        minpix = 50 

        current_left_x_points = []
        current_right_x_points = []
        
        # Copia della maschera per disegnare le finestre (opzionale)
        # out_img = np.dstack((mask, mask, mask)) * 255 

        current_left_x_base = left_base_candidate
        current_right_x_base = right_base_candidate

        for window_idx in range(num_windows):
            win_y_low = transformed_frame.shape[0] - (window_idx + 1) * window_height
            win_y_high = transformed_frame.shape[0] - window_idx * window_height
            
            # Finestra sinistra
            win_xleft_low = current_left_x_base - 75  # Larghezza finestra +/- 75px
            win_xleft_high = current_left_x_base + 75
            # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            
            good_left_inds_y, good_left_inds_x = (mask[win_y_low:win_y_high, max(0,win_xleft_low):min(transformed_width,win_xleft_high)] == 255).nonzero()
            
            if len(good_left_inds_x) > minpix:
                current_left_x_base = max(0,win_xleft_low) + int(np.mean(good_left_inds_x))
                current_left_x_points.append(current_left_x_base)

            # Finestra destra
            win_xright_low = current_right_x_base - 75 # Larghezza finestra +/- 75px
            win_xright_high = current_right_x_base + 75
            # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            
            good_right_inds_y, good_right_inds_x = (mask[win_y_low:win_y_high, max(0, win_xright_low):min(transformed_width,win_xright_high)] == 255).nonzero()

            if len(good_right_inds_x) > minpix:
                current_right_x_base = max(0,win_xright_low) + int(np.mean(good_right_inds_x))
                current_right_x_points.append(current_right_x_base)
        
        # cv2.imshow("Sliding Windows", out_img) # Per visualizzare le finestre

        # Logica di rilevamento e calcolo del centro corsia
        left_line_detected = len(current_left_x_points) > 2 # Richiedi almeno qualche punto
        right_line_detected = len(current_right_x_points) > 2

        frame_center_x = transformed_frame.shape[1] / 2.0
        current_estimated_lane_center = None

        if left_line_detected and right_line_detected:
            left_x_mean = np.mean(current_left_x_points)
            right_x_mean = np.mean(current_right_x_points)
            
            # Controllo di sanità: la linea sinistra deve essere a sinistra della destra
            # e la larghezza della corsia deve essere ragionevole
            detected_width = right_x_mean - left_x_mean
            if left_x_mean < right_x_mean - (EXPECTED_LANE_WIDTH_PX * 0.3) and \
               detected_width > EXPECTED_LANE_WIDTH_PX * 0.5 and \
               detected_width < EXPECTED_LANE_WIDTH_PX * 1.5: # Tolleranza sulla larghezza
                current_estimated_lane_center = (left_x_mean + right_x_mean) / 2.0
                last_valid_lane_center = current_estimated_lane_center
                consecutive_no_lines_frames = 0
            else:
                # print("Warning: Left/Right lines crossed or inconsistent width.")
                if last_valid_lane_center is not None:
                    current_estimated_lane_center = last_valid_lane_center
        
        elif left_line_detected:
            left_x_mean = np.mean(current_left_x_points)
            # Controllo: la linea sinistra è davvero a sinistra?
            if left_x_mean < frame_center_x - (EXPECTED_LANE_WIDTH_PX * 0.1): # Un po' a sinistra del centro
                current_estimated_lane_center = left_x_mean + EXPECTED_LANE_WIDTH_PX / 2.0
                last_valid_lane_center = current_estimated_lane_center # Aggiorna con cautela
                consecutive_no_lines_frames = 0
            else:
                # print("Warning: Left line detected too far right.")
                if last_valid_lane_center is not None:
                    current_estimated_lane_center = last_valid_lane_center

        elif right_line_detected:
            right_x_mean = np.mean(current_right_x_points)
            # Controllo: la linea destra è davvero a destra?
            if right_x_mean > frame_center_x + (EXPECTED_LANE_WIDTH_PX * 0.1): # Un po' a destra del centro
                current_estimated_lane_center = right_x_mean - EXPECTED_LANE_WIDTH_PX / 2.0
                last_valid_lane_center = current_estimated_lane_center # Aggiorna con cautela
                consecutive_no_lines_frames = 0
            else:
                # print("Warning: Right line detected too far left.")
                if last_valid_lane_center is not None:
                    current_estimated_lane_center = last_valid_lane_center
        
        else: # Nessuna linea rilevata
            consecutive_no_lines_frames += 1
            if last_valid_lane_center is not None and consecutive_no_lines_frames < MAX_CONSECUTIVE_NO_LINES:
                current_estimated_lane_center = last_valid_lane_center
            # else: current_estimated_lane_center rimane None

        # Calcolo dello sterzo
        target_angle = 0.0

        if current_estimated_lane_center is not None:
            error = current_estimated_lane_center - frame_center_x
            if abs(error) < TOLERANCE_PX:
                target_angle = 0.0 # Punta dritto se l'errore è piccolo
            else:
                target_angle = -error / STEERING_GAIN
        else:
            # Se nessuna linea è rilevata per troppi frame e non c'è un centro valido precedente,
            # l'auto continuerà con l'ultimo smoothed_angle (che tenderà a 0 se target era 0)
            # o puoi decidere di fermare l'auto o usare un angolo di default.
            # Qui, target_angle rimane 0 se non c'è stima del centro.
            # print("Warning: No lane center estimated, defaulting to straight or last known.")
             pass


        # Applica smorzamento all'angolo di sterzata
        smoothed_angle = (STEERING_SMOOTHING_FACTOR * target_angle) + \
                         ((1.0 - STEERING_SMOOTHING_FACTOR) * smoothed_angle)
        
        final_angle_degrees = np.clip(smoothed_angle, -SERVO_ANGLE_LIMIT, SERVO_ANGLE_LIMIT)
        
        picarx.forward(5) # Mantieni velocità costante per ora
        picarx.set_dir_servo_angle(final_angle_degrees)

        # Mostra immagini (facoltativo ma utile per debug)
        # Disegna il centro della corsia stimato e la linea di guida
        # if current_estimated_lane_center is not None:
        #    cv2.circle(transformed_frame, (int(current_estimated_lane_center), transformed_height - 20), 10, (0, 0, 255), -1)
        # cv2.line(transformed_frame, (int(frame_center_x), transformed_height), 
        #           (int(frame_center_x + final_angle_degrees * STEERING_GAIN /2), transformed_height - 50), (0, 255, 0), 2) # Linea indicativa della sterzata


        cv2.imshow("Original Frame", frame)
        cv2.imshow("Bird's Eye View (Transformed)", transformed_frame)
        cv2.imshow("Lane Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("Stopping PicarX and closing windows.")
    picarx.stop()
    cv2.destroyAllWindows()
    picam2.stop()