import cv2
import numpy as np
from picarx import Picarx # DECOMMENTA
from picamera2 import Picamera2 # DECOMMENTA
import time # UTILE PER LA CAMERA E CONTROLLI

# Inizializzazione PicarX e Picamera2
picarx = Picarx() # DECOMMENTA
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(camera_config)
picam2.start()
time.sleep(1)

# Trackbar HSV
def nothing(x): pass
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Parametri globali
last_direction = 0
filtered_angle = 0 # Non sembra usato, forse retaggio?
valid_lane_center = 320
invalid_frame_count = 0
MAX_INVALID_FRAMES = 10 # Aumenta a 15-20 se serve piÃ¹ tolleranza

# --- PARAMETRI DI CONTROLLO DA FARE TUNING ---
K_DIVISOR = 2.8       # FASE 1: RIDUCI QUESTO VALORE (era 3.5). Inizia con 3.2, poi 3.0, 2.8 etc.
ALPHA_SMOOTHING = 0.4 # FASE 1: AUMENTA QUESTO VALORE (era 0.3). Prova 0.4, 0.5.
MAX_SERVO_ANGLE = 45  # Limite fisico del servo PicarX

# Parametri per controllo velocitÃ  (FASE 2)
ENABLE_DYNAMIC_SPEED = True # Imposta a True per attivare la Fase 2
MAX_SPEED = 25             # VelocitÃ  massima (0-100 per PicarX forward)
MIN_SPEED = 12             # VelocitÃ  minima in curva
SPEED_REDUCTION_THRESHOLD_ANGLE = 18 # Angolo (in gradi) oltre cui si inizia a ridurre velocitÃ 
# --- FINE PARAMETRI DI CONTROLLO ---


estimated_lane_width = 459 # Inizializza con un valore plausibile
MIN_PLAUSIBLE_LANE_WIDTH = 350 # Riduci un po' per maggiore flessibilitÃ 
MAX_PLAUSIBLE_LANE_WIDTH = 550 # Aumenta un po'
LANE_WIDTH_SMOOTH_ALPHA = 0.01

frame_center_x = 320

MIN_PEAK_STRENGTH_RATIO = 0.3
MIN_SEPARATION_FOR_TWO_LINES = 80
LANE_SIDE_CONFIDENCE_THRESHOLD = 50
WINDOW_SEARCH_MARGIN = 50 # Margine per le finestre di sliding
MIN_CONTOUR_AREA_IN_WINDOW = 20 # Riduci se le linee sono sottili

def smooth_angle_func(current, previous, alpha_val): # Rinominato per chiarezza
    return int(alpha_val * current + (1 - alpha_val) * previous)

def draw_windows(image, win_y_low, win_y_high, left_center_x, right_center_x,
                 color_left=(255,0,255), color_right=(0,255,0),
                 draw_left_flag=False, draw_right_flag=False):
    if draw_left_flag:
        cv2.rectangle(image, (left_center_x - WINDOW_SEARCH_MARGIN, win_y_low),
                      (left_center_x + WINDOW_SEARCH_MARGIN, win_y_high), color_left, 2)
    if draw_right_flag:
        cv2.rectangle(image, (right_center_x - WINDOW_SEARCH_MARGIN, win_y_low),
                      (right_center_x + WINDOW_SEARCH_MARGIN, win_y_high), color_right, 2)
    return image

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # CALIBRA QUESTI PUNTI CON PRECISIONE SULLA TUA PISTARX!
        # tl, bl, tr, br = (70,400), (0,480), (570,400), (640,480) # Vecchi
        # Esempio di punti per una vista piÃ¹ "bird-eye" se la camera Ã¨ angolata
        # Devi sperimentare! Crea uno script a parte per aiutarti a trovare questi punti.
        # tl, bl, tr, br = (200,300), (0,480), (440,300), (640,480) # Esempio
        # Immaginiamo che i tuoi (70,400), (0,480), (570,400), (640,480) siano ancora validi
        # Assicurati che l'area ROI sia sensata per la tua pista e la PicarX
        # Potrebbe essere necessario aggiustare l'altezza (secondo valore Y) dei punti superiori
        # tl=(x_top_left, y_top_horizon), bl=(x_bottom_left, y_bottom_image)
        # tr=(x_top_right, y_top_horizon), br=(x_bottom_right, y_bottom_image)
        # Se le curve sono molto strette, potresti voler un y_top_horizon piÃ¹ basso (es. 350 invece di 400)
        # per "vedere" la curva prima.
        ROI_TOP_Y = 380 # Prova a variare questo (es. 350, 380, 400, 420)
        ROI_BOTTOM_Y = 480
        ROI_TL_X_OFFSET = 70  # Distanza dal bordo sx per il punto top-left
        ROI_TR_X_OFFSET = 70  # Distanza dal bordo dx per il punto top-right
        
        tl = (ROI_TL_X_OFFSET, ROI_TOP_Y)
        bl = (0, ROI_BOTTOM_Y)
        tr = (640 - ROI_TR_X_OFFSET, ROI_TOP_Y)
        br = (640, ROI_BOTTOM_Y)

        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, matrix, (640, 480))

        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        l_h, l_s, l_v = cv2.getTrackbarPos("L - H", "Trackbars"), cv2.getTrackbarPos("L - S", "Trackbars"), cv2.getTrackbarPos("L - V", "Trackbars")
        u_h, u_s, u_v = cv2.getTrackbarPos("U - H", "Trackbars"), cv2.getTrackbarPos("U - S", "Trackbars"), cv2.getTrackbarPos("U - V", "Trackbars")
        lower, upper = np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower, upper)

        # --- LOGICA ISTOGRAMMA (invariata) ---
        histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
        midpoint_hist = histogram.shape[0] // 2
        left_half_hist = histogram[:midpoint_hist]
        left_base_candidate = np.argmax(left_half_hist) if left_half_hist.size > 0 and np.any(left_half_hist) else 0
        left_peak_value = np.max(left_half_hist) if left_half_hist.size > 0 and np.any(left_half_hist) else 0
        right_half_hist = histogram[midpoint_hist:]
        right_base_candidate = (np.argmax(right_half_hist) + midpoint_hist) if right_half_hist.size > 0 and np.any(right_half_hist) else midpoint_hist
        right_peak_value = np.max(right_half_hist) if right_half_hist.size > 0 and np.any(right_half_hist) else 0
        max_hist_value = np.max(histogram) if histogram.size > 0 and np.any(histogram) else 1
        search_for_left_line = False
        search_for_right_line = False
        current_left_lane_base = valid_lane_center - estimated_lane_width // 2
        current_right_lane_base = valid_lane_center + estimated_lane_width // 2
        strong_left_peak = left_peak_value > (max_hist_value * MIN_PEAK_STRENGTH_RATIO)
        strong_right_peak = right_peak_value > (max_hist_value * MIN_PEAK_STRENGTH_RATIO)

        if strong_left_peak and strong_right_peak and \
           abs(right_base_candidate - left_base_candidate) > MIN_SEPARATION_FOR_TWO_LINES:
            search_for_left_line = True
            search_for_right_line = True
            current_left_lane_base = left_base_candidate
            current_right_lane_base = right_base_candidate
        elif (strong_left_peak and not strong_right_peak) or \
             (not strong_left_peak and strong_right_peak) or \
             (strong_left_peak and strong_right_peak and \
              abs(right_base_candidate - left_base_candidate) <= MIN_SEPARATION_FOR_TWO_LINES):
            single_line_pos_candidate = 0
            if (strong_left_peak and not strong_right_peak): single_line_pos_candidate = left_base_candidate
            elif (not strong_left_peak and strong_right_peak): single_line_pos_candidate = right_base_candidate
            else: single_line_pos_candidate = (left_base_candidate + right_base_candidate) // 2
            
            reference_center = valid_lane_center
            if single_line_pos_candidate < reference_center - LANE_SIDE_CONFIDENCE_THRESHOLD:
                search_for_left_line = True
                current_left_lane_base = single_line_pos_candidate
                current_right_lane_base = current_left_lane_base + estimated_lane_width
            elif single_line_pos_candidate > reference_center + LANE_SIDE_CONFIDENCE_THRESHOLD:
                search_for_right_line = True
                current_right_lane_base = single_line_pos_candidate
                current_left_lane_base = current_right_lane_base - estimated_lane_width
            else: # Ambigua, usa la posizione rispetto al centro del frame
                if single_line_pos_candidate < frame_center_x:
                    search_for_left_line = True
                    current_left_lane_base = single_line_pos_candidate
                    current_right_lane_base = current_left_lane_base + estimated_lane_width
                else:
                    search_for_right_line = True
                    current_right_lane_base = single_line_pos_candidate
                    current_left_lane_base = current_right_lane_base - estimated_lane_width
        else: # Nessun picco chiaro o troppo deboli
            search_for_left_line = True # Cerca entrambe per default
            search_for_right_line = True
        
        current_left_lane_base = np.clip(current_left_lane_base, 0, warped.shape[1]-1-WINDOW_SEARCH_MARGIN) # Evita out of bounds
        current_right_lane_base = np.clip(current_right_lane_base, WINDOW_SEARCH_MARGIN, warped.shape[1]-1) # Evita out of bounds
        # --- FINE LOGICA ISTOGRAMMA ---

        y_sliding = warped.shape[0] - 1 
        window_height = 40
        left_x_points, right_x_points = [], []
        annotated_warped = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        iter_left_base = current_left_lane_base
        iter_right_base = current_right_lane_base
        num_windows = warped.shape[0] // window_height

        for window_idx in range(num_windows):
            win_y_high = warped.shape[0] - (window_idx * window_height)
            win_y_low = warped.shape[0] - ((window_idx + 1) * window_height)
            drew_left_in_this_window = False
            drew_right_in_this_window = False
            if search_for_left_line:
                win_xleft_low = max(0, iter_left_base - WINDOW_SEARCH_MARGIN)
                win_xleft_high = min(warped.shape[1], iter_left_base + WINDOW_SEARCH_MARGIN)
                left_window_roi = mask[win_y_low:win_y_high, win_xleft_low:win_xleft_high]
                contours_l, _ = cv2.findContours(left_window_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours_l:
                    largest_l = max(contours_l, key=cv2.contourArea)
                    if cv2.contourArea(largest_l) > MIN_CONTOUR_AREA_IN_WINDOW:
                        M_l = cv2.moments(largest_l)
                        if M_l["m00"] != 0:
                            cx_l = int(M_l["m10"] / M_l["m00"])
                            iter_left_base = win_xleft_low + cx_l
                            left_x_points.append(iter_left_base)
                            drew_left_in_this_window = True
            if search_for_right_line:
                win_xright_low = max(0, iter_right_base - WINDOW_SEARCH_MARGIN)
                win_xright_high = min(warped.shape[1], iter_right_base + WINDOW_SEARCH_MARGIN)
                right_window_roi = mask[win_y_low:win_y_high, win_xright_low:win_xright_high]
                contours_r, _ = cv2.findContours(right_window_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours_r:
                    largest_r = max(contours_r, key=cv2.contourArea)
                    if cv2.contourArea(largest_r) > MIN_CONTOUR_AREA_IN_WINDOW:
                        M_r = cv2.moments(largest_r)
                        if M_r["m00"] != 0:
                            cx_r = int(M_r["m10"] / M_r["m00"])
                            iter_right_base = win_xright_low + cx_r
                            right_x_points.append(iter_right_base)
                            drew_right_in_this_window = True
            annotated_warped = draw_windows(annotated_warped, win_y_low, win_y_high,
                                            iter_left_base, iter_right_base,
                                            draw_left_flag=drew_left_in_this_window and search_for_left_line,
                                            draw_right_flag=drew_right_in_this_window and search_for_right_line)

        new_lane_center_calculated_this_frame = False
        current_frame_lane_center = valid_lane_center # Default al precedente valido
        
        if left_x_points and right_x_points:
            left_avg = int(np.mean(left_x_points))
            right_avg = int(np.mean(right_x_points))
            current_detected_width = right_avg - left_avg
            if MIN_PLAUSIBLE_LANE_WIDTH < current_detected_width < MAX_PLAUSIBLE_LANE_WIDTH:
                current_frame_lane_center = (left_avg + right_avg) // 2
                estimated_lane_width = int(LANE_WIDTH_SMOOTH_ALPHA * current_detected_width + \
                                           (1 - LANE_WIDTH_SMOOTH_ALPHA) * estimated_lane_width)
                valid_lane_center = current_frame_lane_center
                invalid_frame_count = 0
                new_lane_center_calculated_this_frame = True
        elif left_x_points: # Solo linea sinistra trovata
            left_avg = int(np.mean(left_x_points))
            # Usa la larghezza stimata per calcolare il centro
            if MIN_PLAUSIBLE_LANE_WIDTH < estimated_lane_width < MAX_PLAUSIBLE_LANE_WIDTH:
                current_frame_lane_center = left_avg + estimated_lane_width // 2
                valid_lane_center = current_frame_lane_center # Aggiorna il centro valido
                invalid_frame_count = 0
                new_lane_center_calculated_this_frame = True
        elif right_x_points: # Solo linea destra trovata
            right_avg = int(np.mean(right_x_points))
            # Usa la larghezza stimata per calcolare il centro
            if MIN_PLAUSIBLE_LANE_WIDTH < estimated_lane_width < MAX_PLAUSIBLE_LANE_WIDTH:
                current_frame_lane_center = right_avg - estimated_lane_width // 2
                valid_lane_center = current_frame_lane_center # Aggiorna il centro valido
                invalid_frame_count = 0
                new_lane_center_calculated_this_frame = True

        if not new_lane_center_calculated_this_frame:
            invalid_frame_count += 1
        
        final_lane_center = valid_lane_center # Usiamo sempre l'ultimo centro valido noto

        angle = last_direction # Default all'ultimo angolo se non si fa nulla

        if invalid_frame_count > MAX_INVALID_FRAMES:
            picarx.stop() # DECOMMENTA per fermare la macchina
            print(f"STOP: {invalid_frame_count} frame consecutivi senza rilevamento affidabile.")
            # Potresti voler mantenere l'ultimo angolo di sterzata o raddrizzare le ruote
            # picarx.set_dir_servo_angle(0)
        else:
            error = final_lane_center - frame_center_x
            
            # --- FASE 1: Calcolo Angolo Migliorato ---
            raw_angle = -int(error / K_DIVISOR)
            # Non Ã¨ necessario un clip intermedio qui se K_DIVISOR Ã¨ ben tunato,
            # ma se lo vuoi, assicurati che sia ampio, es. MAX_SERVO_ANGLE * 1.2
            # raw_angle = np.clip(raw_angle, -int(MAX_SERVO_ANGLE*1.2), int(MAX_SERVO_ANGLE*1.2))

            angle = smooth_angle_func(raw_angle, last_direction, ALPHA_SMOOTHING)
            angle = np.clip(angle, -MAX_SERVO_ANGLE, MAX_SERVO_ANGLE)
            last_direction = angle
            # --- FINE FASE 1 ---

            current_speed_to_set = MAX_SPEED # Default speed

            if ENABLE_DYNAMIC_SPEED:
                # --- FASE 2: Controllo VelocitÃ  Dinamico ---
                abs_angle = abs(angle)
                if abs_angle > SPEED_REDUCTION_THRESHOLD_ANGLE:
                    reduction_range = MAX_SERVO_ANGLE - SPEED_REDUCTION_THRESHOLD_ANGLE
                    if reduction_range <=0: reduction_range = MAX_SERVO_ANGLE # Evita divisione per zero
                    
                    overshoot_angle = abs_angle - SPEED_REDUCTION_THRESHOLD_ANGLE
                    speed_factor = max(0, 1 - (overshoot_angle / reduction_range))
                    current_speed_to_set = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * speed_factor
                    current_speed_to_set = int(np.clip(current_speed_to_set, MIN_SPEED, MAX_SPEED))
                else:
                    current_speed_to_set = MAX_SPEED
                # --- FINE FASE 2 ---
            
            # print(f"L:{len(left_x_points)} R:{len(right_x_points)} Err:{error} RawAng:{raw_angle} SmAng:{angle} Spd:{current_speed_to_set}")
            
            # QUI implementeresti la FASE 3 (controllo differenziale) se volessi
            # al posto di picarx.forward()
            # Per ora usiamo Fase 1 e 2:
            picarx.forward(current_speed_to_set) # DECOMMENTA
            picarx.set_dir_servo_angle(-angle)   # DECOMMENTA

        # Annotazioni
        cv2.line(annotated_warped, (final_lane_center, 0), (final_lane_center, warped.shape[0]), (0,255,255), 2)
        cv2.line(annotated_warped, (frame_center_x, 0), (frame_center_x, warped.shape[0]), (255,255,255), 1)
        # Mostra l'angolo che viene effettivamente inviato al servo (con il segno invertito)
        cv2.putText(annotated_warped, f"Servo Angle: {-angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(annotated_warped, f"L_pts: {len(left_x_points)} R_pts: {len(right_x_points)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.putText(annotated_warped, f"Est.W: {estimated_lane_width:.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        if ENABLE_DYNAMIC_SPEED and not (invalid_frame_count > MAX_INVALID_FRAMES) :
             cv2.putText(annotated_warped, f"Speed: {current_speed_to_set}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


        cv2.imshow("Annotated Warped", annotated_warped)
        # cv2.imshow("Original Frame", frame) # Utile per calibrare la prospettiva
        # cv2.imshow("Mask", mask)

        if cv2.waitKey(1) == 27: # ESC per uscire
            break
        
        # time.sleep(0.01) # Riduci o rimuovi se il processing Ã¨ giÃ  abbastanza lento

finally:
    print("Uscita dal programma...")
    picarx.stop() # DECOMMENTA
    picam2.stop()
    cv2.destroyAllWindows()