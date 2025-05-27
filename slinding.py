import cv2
import numpy as np
from picarx import Picarx # DECOMMENTATO
from picamera2 import Picamera2 # DECOMMENTATO
import time

# --- INIZIALIZZAZIONE PICARX E PICAMERA2 ---
picar = Picarx() # DECOMMENTATO
picam2 = Picamera2() # DECOMMENTATO

# Configurazione Picamera2
# Nota: OpenCV usa BGR, Picamera2 per default cattura in RGB. La conversione è necessaria.
# Se la tua pipeline si aspetta BGR (es. per cvtColor(..., cv2.COLOR_BGR2HSV)), converti.
camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}, # Cattura in RGB
    controls={"FrameDurationLimits": (33333, 33333)} # ~30 FPS, (16666, 16666) per ~60FPS se supportato
)
picam2.configure(camera_config) # DECOMMENTATO
picam2.start() # DECOMMENTATO
time.sleep(1) # Dai tempo alla camera di avviarsi e stabilizzarsi
# --- FINE INIZIALIZZAZIONE ---


# Trackbar HSV (mantieni se hai un display collegato per tuning live, altrimenti puoi rimuoverle se i valori sono fissi)
# Se non hai un display sulla PicarX per vedere le finestre OpenCV, le trackbar non saranno utilizzabili
# e dovrai impostare i valori HSV direttamente nel codice.
# Per ora le lascio, ma considera questo aspetto.
def nothing(x): pass
cv2.namedWindow("Trackbars") # Richiede un ambiente grafico
# Se vuoi testare senza display, commenta namedWindow e createTrackbar, e imposta i valori HSV fissi:
# hsv_lower_fixed = np.array([0, 0, 200])
# hsv_upper_fixed = np.array([255, 50, 255])
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)


# Parametri globali (invariati dalla versione precedente)
last_direction = 0
valid_lane_center = 320 # Inizializza al centro del frame (640/2)
invalid_frame_count = 0
MAX_INVALID_FRAMES = 150 # Quanti frame "sbagliati" prima di fermarsi o usare l'ultimo angolo
alpha = 0.3

estimated_lane_width = 459 # CALIBRA basandoti sulla tua pista e trasformazione prospettica
MIN_PLAUSIBLE_LANE_WIDTH = 458 # CALIBRA
MAX_PLAUSIBLE_LANE_WIDTH = 460 # CALIBRA
LANE_WIDTH_SMOOTH_ALPHA = 0.05

frame_width = 640 # Larghezza del frame catturato
frame_height = 480 # Altezza del frame catturato
frame_center_x = frame_width // 2

# Parametri istogramma
MIN_PEAK_STRENGTH_RATIO = 0.3
MIN_SEPARATION_FOR_TWO_LINES = 80 # CALIBRA
LANE_SIDE_CONFIDENCE_THRESHOLD = 50 # CALIBRA

# Parametri Sliding Window
WINDOW_SEARCH_MARGIN = 75 # CALIBRA
MIN_CONTOUR_AREA_IN_WINDOW = 20
WINDOW_HEIGHT_PARAM = 20
MIN_POINTS_FOR_FIT = 3

# Parametri per la sterzata aggressiva in curva
Y_EVAL_POINT_FACTOR = 0.5 # CALIBRA (0.3-0.7 è un range comune)
AGGRESSIVE_STEERING_SLOPE_THRESHOLD = 0.3 # CALIBRA
AGGRESSIVE_OFFSET_FACTOR = 100 # CALIBRA
MAX_AGGRESSIVE_OFFSET_RATIO = 0.30 # CALIBRA

# Parametri per i filtri di coerenza
FIT_CROSS_INVALIDATION_MARGIN = MIN_SEPARATION_FOR_TWO_LINES / 3 # CALIBRA

# --- PARAMETRI DI CONTROLLO PICARX ---
PICARX_SPEED = 15  # CALIBRA: Velocità di avanzamento (0-50). Inizia basso!
STEERING_FACTOR = 3.0 # CALIBRA: (error / STEERING_FACTOR). Valore più piccolo = sterzata più reattiva.
# --- FINE PARAMETRI DI CONTROLLO ---


def smooth_angle(current, previous, alpha_val=0.3):
    return int(alpha_val * current + (1 - alpha_val) * previous)

def draw_windows(image, win_y_low, win_y_high, left_center_x, right_center_x,
                 color_left=(255,0,255), color_right=(0,255,0),
                 draw_left_flag=False, draw_right_flag=False):
    if draw_left_flag:
        cv2.rectangle(image, (left_center_x - WINDOW_SEARCH_MARGIN, win_y_low),
                      (left_center_x + WINDOW_SEARCH_MARGIN, win_y_high), color_left, 1)
    if draw_right_flag:
        cv2.rectangle(image, (right_center_x - WINDOW_SEARCH_MARGIN, win_y_low),
                      (right_center_x + WINDOW_SEARCH_MARGIN, win_y_high), color_right, 1)
    return image

# Variabile per controllare se mostrare le finestre OpenCV (utile se si esegue headless)
SHOW_CV_WINDOWS = True # Imposta a False se esegui senza display collegato

try:
    print("Avvio rilevamento corsia sulla PicarX...")
    picar.set_dir_servo_angle(0) # Assicurati che la camera guardi dritto o all'angolazione desiderata
    time.sleep(0.5)

    while True: # Loop principale per la PicarX
        frame_rgb = picam2.capture_array() # Cattura frame come array RGB
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # Converti in BGR per OpenCV

        # Trasformazione prospettica - FONDAMENTALE CALIBRARLA SULLA PICARX!
        # Questi punti sono solo un ESEMPIO e quasi certamente dovranno essere cambiati.
        # tl, bl, tr, br = (70, frame_height*0.52), (0, frame_height), (frame_width-70, frame_height*0.52), (frame_width, frame_height)
        # Esempio di punti che potrebbero funzionare, ma VANNO CALIBRATI:
        # tl=(200, 300) bl=(0, 460) tr=(440, 300) br=(640, 460) # Per una camera più alta
        # Per una camera bassa sulla PicarX, y_top potrebbe essere più basso (es. 200-250) e y_bottom più alto (es. 350-400)
        # Esempio usato nei test video:
        tl, bl, tr, br = (70,250), (0,480), (570,250), (640,480) # RICORDA DI CALIBRARE QUESTI PUNTI SULLA MACCHINA REALE!
        
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0,0], [0,frame_height], [frame_width,0], [frame_width,frame_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, matrix, (frame_width, frame_height))

        # Threshold HSV
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        if SHOW_CV_WINDOWS:
            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")
            lower_hsv, upper_hsv = np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])
        else: # Valori HSV fissi se non si usano le trackbar
            # CALIBRA QUESTI VALORI HSV DIRETTAMENTE SE NON USI LE TRACKBAR
            lower_hsv = np.array([0, 0, 180]) # Esempio, da calibrare
            upper_hsv = np.array([180, 60, 255]) # Esempio, da calibrare
        
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        kernel_dilate = np.ones((3,3),np.uint8) # Prova (3,3) o (5,5)
        mask = cv2.dilate(mask, kernel_dilate, iterations = 1)

        # --- LOGICA ISTOGRAMMA RAFFORZATA (invariata dalla versione precedente) ---
        histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
        midpoint_hist = histogram.shape[0] // 2
        left_half_hist = histogram[:midpoint_hist]
        left_base_candidate = np.argmax(left_half_hist) if left_half_hist.size > 0 else 0
        left_peak_value = np.max(left_half_hist) if left_half_hist.size > 0 else 0
        right_half_hist = histogram[midpoint_hist:]
        right_base_candidate = (np.argmax(right_half_hist) + midpoint_hist) if right_half_hist.size > 0 else midpoint_hist
        right_peak_value = np.max(right_half_hist) if right_half_hist.size > 0 else 0
        max_hist_value = np.max(histogram) if histogram.size > 0 else 1
        search_for_left_line = False
        search_for_right_line = False
        current_left_lane_base = valid_lane_center - estimated_lane_width // 2
        current_right_lane_base = valid_lane_center + estimated_lane_width // 2
        strong_left_peak = left_peak_value > (max_hist_value * MIN_PEAK_STRENGTH_RATIO)
        strong_right_peak = right_peak_value > (max_hist_value * MIN_PEAK_STRENGTH_RATIO)
        reference_center_for_hist = valid_lane_center
        hist_decision = "None"
        if strong_left_peak and strong_right_peak and abs(right_base_candidate - left_base_candidate) > MIN_SEPARATION_FOR_TWO_LINES:
            search_for_left_line, search_for_right_line = True, True
            current_left_lane_base, current_right_lane_base = left_base_candidate, right_base_candidate
            hist_decision = "LR_strong"
        elif (strong_left_peak and not strong_right_peak) or (not strong_left_peak and strong_right_peak) or \
             (strong_left_peak and strong_right_peak and abs(right_base_candidate - left_base_candidate) <= MIN_SEPARATION_FOR_TWO_LINES):
            single_line_pos_candidate = left_base_candidate if (strong_left_peak and not strong_right_peak) else \
                                     (right_base_candidate if (not strong_left_peak and strong_right_peak) else \
                                     (left_base_candidate + right_base_candidate) // 2)
            if single_line_pos_candidate < reference_center_for_hist - LANE_SIDE_CONFIDENCE_THRESHOLD:
                search_for_left_line, search_for_right_line = True, False
                current_left_lane_base = single_line_pos_candidate
                current_right_lane_base = current_left_lane_base + estimated_lane_width
                hist_decision = "L_only"
            elif single_line_pos_candidate > reference_center_for_hist + LANE_SIDE_CONFIDENCE_THRESHOLD:
                search_for_left_line, search_for_right_line = False, True
                current_right_lane_base = single_line_pos_candidate
                current_left_lane_base = current_right_lane_base - estimated_lane_width
                hist_decision = "R_only"
            else: # Ambiguo, usa frame_center_x
                if single_line_pos_candidate < frame_center_x:
                    search_for_left_line, search_for_right_line = True, False
                    current_left_lane_base = single_line_pos_candidate
                    current_right_lane_base = current_left_lane_base + estimated_lane_width
                    hist_decision = "L_amb_frame"
                else:
                    search_for_left_line, search_for_right_line = False, True
                    current_right_lane_base = single_line_pos_candidate
                    current_left_lane_base = current_right_lane_base - estimated_lane_width
                    hist_decision = "R_amb_frame"
        else: # Fallback
            search_for_left_line, search_for_right_line = True, True
            hist_decision = "LR_fallback"
        current_left_lane_base = np.clip(current_left_lane_base, 0, frame_width-1)
        current_right_lane_base = np.clip(current_right_lane_base, 0, frame_width-1)
        # --- FINE LOGICA ISTOGRAMMA ---

        # --- SLIDING WINDOW & RACCOLTA PUNTI (invariata) ---
        window_height = WINDOW_HEIGHT_PARAM
        num_windows = frame_height // window_height
        left_x_points, right_x_points = [], []
        left_y_points, right_y_points = [], []
        annotated_warped = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if SHOW_CV_WINDOWS else None # Crea solo se necessario
        iter_left_base = current_left_lane_base
        iter_right_base = current_right_lane_base
        plot_y_coords_for_poly = np.linspace(0, frame_height-1, num=frame_height)

        for window_idx in range(num_windows):
            win_y_high = frame_height - (window_idx * window_height)
            win_y_low = frame_height - ((window_idx + 1) * window_height)
            win_y_center = (win_y_high + win_y_low) // 2
            if search_for_left_line:
                win_xleft_low = max(0, iter_left_base - WINDOW_SEARCH_MARGIN)
                win_xleft_high = min(frame_width-1, iter_left_base + WINDOW_SEARCH_MARGIN)
                if search_for_right_line: win_xleft_high = min(win_xleft_high, iter_right_base - int(MIN_SEPARATION_FOR_TWO_LINES / 2))
                if win_xleft_low < win_xleft_high:
                    # ... (trova contorni e punti come prima) ...
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
                                left_y_points.append(win_y_center)
            if search_for_right_line:
                win_xright_low = max(0, iter_right_base - WINDOW_SEARCH_MARGIN)
                win_xright_high = min(frame_width-1, iter_right_base + WINDOW_SEARCH_MARGIN)
                if search_for_left_line: win_xright_low = max(win_xright_low, iter_left_base + int(MIN_SEPARATION_FOR_TWO_LINES / 2))
                if win_xright_low < win_xright_high:
                    # ... (trova contorni e punti come prima) ...
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
                                right_y_points.append(win_y_center)
            if SHOW_CV_WINDOWS and annotated_warped is not None:
                annotated_warped = draw_windows(annotated_warped, win_y_low, win_y_high,
                                                iter_left_base, iter_right_base,
                                                draw_left_flag=search_for_left_line,
                                                draw_right_flag=search_for_right_line)
        # --- FINE SLIDING WINDOW ---

        # --- FILTRO SUI PUNTI RACCOLTI (invariato) ---
        points_filter_debug_text = ""
        if search_for_left_line and len(left_x_points) > 0:
            max_x_boundary_for_left = valid_lane_center + (WINDOW_SEARCH_MARGIN if not search_for_right_line else -int(MIN_SEPARATION_FOR_TWO_LINES/3))
            original_l_pts_count = len(left_x_points)
            filtered_left_x, filtered_left_y = zip(*[(x,y) for x,y in zip(left_x_points, left_y_points) if x < max_x_boundary_for_left]) if any(x < max_x_boundary_for_left for x in left_x_points) else ([],[])
            if len(filtered_left_x) < original_l_pts_count: points_filter_debug_text += f"L_filt({original_l_pts_count}->{len(filtered_left_x)}) "
            left_x_points, left_y_points = list(filtered_left_x), list(filtered_left_y)
        if search_for_right_line and len(right_x_points) > 0:
            min_x_boundary_for_right = valid_lane_center - (WINDOW_SEARCH_MARGIN if not search_for_left_line else -int(MIN_SEPARATION_FOR_TWO_LINES/3))
            original_r_pts_count = len(right_x_points)
            filtered_right_x, filtered_right_y = zip(*[(x,y) for x,y in zip(right_x_points, right_y_points) if x > min_x_boundary_for_right]) if any(x > min_x_boundary_for_right for x in right_x_points) else ([],[])
            if len(filtered_right_x) < original_r_pts_count: points_filter_debug_text += f"R_filt({original_r_pts_count}->{len(filtered_right_x)}) "
            right_x_points, right_y_points = list(filtered_right_x), list(filtered_right_y)
        # --- FINE FILTRO PUNTI ---

        # --- FITTING POLINOMIALE E CALCOLO CENTRO CORSIA (logica di coerenza fit e sterzata aggressiva invariata) ---
        new_lane_center_calculated_this_frame = False
        current_frame_lane_center = valid_lane_center # Inizia con l'ultimo valido
        left_fit_coeffs, right_fit_coeffs = None, None
        if len(left_x_points) >= MIN_POINTS_FOR_FIT:
            try: left_fit_coeffs = np.polyfit(np.array(left_y_points), np.array(left_x_points), 2)
            except: left_fit_coeffs = None
        if len(right_x_points) >= MIN_POINTS_FOR_FIT:
            try: right_fit_coeffs = np.polyfit(np.array(right_y_points), np.array(right_x_points), 2)
            except: right_fit_coeffs = None

        y_eval_point = frame_height * Y_EVAL_POINT_FACTOR 
        aggressive_debug_text = ""
        fit_check_debug_text = ""

        if left_fit_coeffs is not None and right_fit_coeffs is not None:
            left_x_at_y_eval_check = left_fit_coeffs[0]
            y_eval_point*2 + left_fit_coeffs[1]*y_eval_point + left_fit_coeffs[2]
            right_x_at_y_eval_check = right_fit_coeffs[0]
            y_eval_point*2 + right_fit_coeffs[1]*y_eval_point + right_fit_coeffs[2]
            if left_x_at_y_eval_check >= right_x_at_y_eval_check - FIT_CROSS_INVALIDATION_MARGIN:
                fit_check_debug_text = f"FitX!L{left_x_at_y_eval_check:.0f}>R{right_x_at_y_eval_check:.0f}"
                # ... (logica di invalidazione fit come prima) ...
                if left_x_at_y_eval_check > valid_lane_center + WINDOW_SEARCH_MARGIN : 
                    left_fit_coeffs = None; fit_check_debug_text += "->L_inv(too_right)"
                elif right_x_at_y_eval_check < valid_lane_center - WINDOW_SEARCH_MARGIN :
                    right_fit_coeffs = None; fit_check_debug_text += "->R_inv(too_left)"
                elif len(left_x_points) < len(right_x_points) :
                    left_fit_coeffs = None; fit_check_debug_text += "->L_inv(pts)"
                elif len(right_x_points) < len(left_x_points) :
                    right_fit_coeffs = None; fit_check_debug_text += "->R_inv(pts)"
                else: left_fit_coeffs = None ; fit_check_debug_text += "->L_inv(def)"


        if left_fit_coeffs is not None and right_fit_coeffs is not None:
            left_x_at_y_eval = left_fit_coeffs[0]
            y_eval_point*2 + left_fit_coeffs[1]*y_eval_point + left_fit_coeffs[2]
            right_x_at_y_eval = right_fit_coeffs[0]
            y_eval_point*2 + right_fit_coeffs[1]*y_eval_point + right_fit_coeffs[2]
            current_detected_width = right_x_at_y_eval - left_x_at_y_eval
            if MIN_PLAUSIBLE_LANE_WIDTH < current_detected_width < MAX_PLAUSIBLE_LANE_WIDTH:
                current_frame_lane_center = int((left_x_at_y_eval + right_x_at_y_eval) / 2)
                estimated_lane_width = int(LANE_WIDTH_SMOOTH_ALPHA * current_detected_width + (1 - LANE_WIDTH_SMOOTH_ALPHA) * estimated_lane_width)
                new_lane_center_calculated_this_frame = True
            elif len(left_x_points) > 0 and len(right_x_points) > 0:
                left_avg, right_avg = int(np.mean(left_x_points)), int(np.mean(right_x_points))
                current_detected_width_raw = right_avg - left_avg
                if MIN_PLAUSIBLE_LANE_WIDTH < current_detected_width_raw < MAX_PLAUSIBLE_LANE_WIDTH:
                    current_frame_lane_center = (left_avg + right_avg) // 2
                    estimated_lane_width = int(LANE_WIDTH_SMOOTH_ALPHA * current_detected_width_raw + (1 - LANE_WIDTH_SMOOTH_ALPHA) * estimated_lane_width)
                    new_lane_center_calculated_this_frame = True
        elif left_fit_coeffs is not None:
            left_x_at_y_eval = left_fit_coeffs[0]
            y_eval_point*2 + left_fit_coeffs[1]*y_eval_point + left_fit_coeffs[2]
            slope_left = 2 * left_fit_coeffs[0] * y_eval_point + left_fit_coeffs[1]
            additional_offset = 0
            if slope_left < -AGGRESSIVE_STEERING_SLOPE_THRESHOLD:
                severity = abs(slope_left) - AGGRESSIVE_STEERING_SLOPE_THRESHOLD
                additional_offset = min(severity * AGGRESSIVE_OFFSET_FACTOR, estimated_lane_width * MAX_AGGRESSIVE_OFFSET_RATIO)
                aggressive_debug_text = f"AggroL:s{slope_left:.2f} off{additional_offset:.0f}"
            current_frame_lane_center = int(left_x_at_y_eval + (estimated_lane_width / 2) + additional_offset)
            new_lane_center_calculated_this_frame = True
        elif right_fit_coeffs is not None:
            right_x_at_y_eval = right_fit_coeffs[0]
            y_eval_point*2 + right_fit_coeffs[1]*y_eval_point + right_fit_coeffs[2]
            slope_right = 2 * right_fit_coeffs[0] * y_eval_point + right_fit_coeffs[1]
            additional_offset = 0
            if slope_right > AGGRESSIVE_STEERING_SLOPE_THRESHOLD:
                severity = abs(slope_right) - AGGRESSIVE_STEERING_SLOPE_THRESHOLD
                additional_offset = min(severity * AGGRESSIVE_OFFSET_FACTOR, estimated_lane_width * MAX_AGGRESSIVE_OFFSET_RATIO)
                aggressive_debug_text = f"AggroR:s{slope_right:.2f} off{additional_offset:.0f}"
            current_frame_lane_center = int(right_x_at_y_eval - (estimated_lane_width / 2) - additional_offset)
            new_lane_center_calculated_this_frame = True
        
        if not new_lane_center_calculated_this_frame: # Fallback finale ai punti grezzi
            if len(left_x_points) > 0 and len(right_x_points) > 0:
                left_avg, right_avg = int(np.mean(left_x_points)), int(np.mean(right_x_points))
                # ... (come prima) ...
                current_detected_width_raw = right_avg - left_avg
                if MIN_PLAUSIBLE_LANE_WIDTH < current_detected_width_raw < MAX_PLAUSIBLE_LANE_WIDTH:
                    current_frame_lane_center = (left_avg + right_avg) // 2
                    estimated_lane_width = int(LANE_WIDTH_SMOOTH_ALPHA * current_detected_width_raw + (1 - LANE_WIDTH_SMOOTH_ALPHA) * estimated_lane_width)
                    new_lane_center_calculated_this_frame = True
            elif len(left_x_points) >= MIN_POINTS_FOR_FIT : 
                current_frame_lane_center = int(np.mean(left_x_points)) + estimated_lane_width // 2
                new_lane_center_calculated_this_frame = True
            elif len(right_x_points) >= MIN_POINTS_FOR_FIT :
                current_frame_lane_center = int(np.mean(right_x_points)) - estimated_lane_width // 2
                new_lane_center_calculated_this_frame = True

        if new_lane_center_calculated_this_frame:
            invalid_frame_count = 0
            valid_lane_center = current_frame_lane_center
        else:
            invalid_frame_count += 1
        
        final_lane_center = np.clip(valid_lane_center, 0, frame_width-1)
        # --- FINE CALCOLO CENTRO ---

        # --- CONTROLLO VEICOLO ---
        angle = 0
        if invalid_frame_count > MAX_INVALID_FRAMES:
            print(f"STOP: {invalid_frame_count} frame consecutivi senza rilevamento affidabile.")
            picar.stop() # Ferma la macchina
            # Potresti voler mantenere l'ultimo angolo o raddrizzare le ruote
            angle = 0 # o last_direction
            picar.set_dir_servo_angle(-angle) # Angolo convertito per il servo PicarX
            # Per uscire dal loop o attendere:
            # break # Per fermare lo script
            time.sleep(0.1) # Pausa per evitare di intasare la CPU se si blocca qui
            # continue # Per riprovare al prossimo frame (se non si usa break)
        else:
            error = final_lane_center - frame_center_x
            raw_angle = -int(error / STEERING_FACTOR) # Usa il parametro calibrato
            raw_angle = np.clip(raw_angle, -40, 40) # Limiti grezzi del servo PicarX
            angle = smooth_angle(raw_angle, last_direction, alpha)
            angle = np.clip(angle, -45, 45) # Limiti finali (possono essere più stretti di quelli del servo)
            last_direction = angle

            picar.forward(PICARX_SPEED) # Muovi in avanti con velocità calibrata
            picar.set_dir_servo_angle(-angle) # Imposta l'angolo di sterzo (il segno '-' potrebbe dipendere da come è montato il servo)
        # --- FINE CONTROLLO VEICOLO ---


        # --- ANNOTAZIONI E VISUALIZZAZIONE (opzionale sulla PicarX, utile per debug se c'è display) ---
        if SHOW_CV_WINDOWS and annotated_warped is not None:
            fit_status = ""
            if left_fit_coeffs is not None: fit_status += "L"
            if right_fit_coeffs is not None: fit_status += "R"
            if not fit_status: fit_status = "None"

            if left_fit_coeffs is not None:
                left_fit_x = left_fit_coeffs[0]
                plot_y_coords_for_poly*2 + left_fit_coeffs[1]*plot_y_coords_for_poly + left_fit_coeffs[2]
                pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y_coords_for_poly]))])
                cv2.polylines(annotated_warped, [np.int32(pts_left)], isClosed=False, color=(0,0,255), thickness=2)
            if right_fit_coeffs is not None:
                right_fit_x = right_fit_coeffs[0]
                plot_y_coords_for_poly*2 + right_fit_coeffs[1]*plot_y_coords_for_poly + right_fit_coeffs[2]
                pts_right = np.array([np.transpose(np.vstack([right_fit_x, plot_y_coords_for_poly]))])
                cv2.polylines(annotated_warped, [np.int32(pts_right)], isClosed=False, color=(255,255,0), thickness=2)

            cv2.line(annotated_warped, (final_lane_center, 0), (final_lane_center, frame_height), (0,255,255), 2)
            cv2.line(annotated_warped, (frame_center_x, 0), (frame_center_x, frame_height), (255,255,255), 1)
            
            cv2.putText(annotated_warped, f"Angle: {angle}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            debug_text_line1 = f"L{len(left_x_points)} R{len(right_x_points)} Fit:{fit_status} Hist:{hist_decision}"
            if points_filter_debug_text: debug_text_line1 = points_filter_debug_text.strip() + " " + debug_text_line1[0:30] # Limita lunghezza
            cv2.putText(annotated_warped, debug_text_line1, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            debug_text_line2 = f"EstW{estimated_lane_width:.0f} InvF{invalid_frame_count}"
            if fit_check_debug_text: debug_text_line2 = fit_check_debug_text.strip() + " " + debug_text_line2[0:30] # Limita lunghezza
            cv2.putText(annotated_warped, debug_text_line2, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            if aggressive_debug_text:
                 cv2.putText(annotated_warped, aggressive_debug_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1)

            cv2.imshow("Annotated Warped", annotated_warped)
            # cv2.imshow("Original Frame", frame) # Utile per calibrare la prospettiva
            # cv2.imshow("Mask", mask) # Utile per calibrare HSV

            if cv2.waitKey(1) == 27: # ESC per uscire
                print("Uscita manuale richiesta.")
                break
        
        # Piccolo delay per non sovraccaricare la CPU, specialmente se non si mostrano finestre
        # time.sleep(0.001) 

except KeyboardInterrupt:
    print("Programma interrotto da tastiera (Ctrl+C).")
finally:
    print("Fermata PicarX e pulizia...")
    picar.stop()
    picar.set_dir_servo_angle(0) # Raddrizza le ruote
    picam2.stop() # Ferma la camera
    if SHOW_CV_WINDOWS:
        cv2.destroyAllWindows()
    print("Programma terminato.")