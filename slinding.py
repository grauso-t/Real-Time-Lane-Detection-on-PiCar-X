import cv2
import numpy as np
from picarx import Picarx # MODIFICA: Decommentato
from picamera2 import Picamera2 # MODIFICA: Decommentato
import time

# MODIFICA: Rimossa la lettura da video
# cap = cv2.VideoCapture("antiorarioBuono.mp4")

# MODIFICA: Inizializzazione Picarx e PiCamera2
picarx = Picarx()
picam2 = Picamera2()
# CONSIGLIO: Puoi abbassare la risoluzione se le performance non sono sufficienti,
# ma 640x480 è un buon compromesso. Assicurati che il formato sia uno che OpenCV legge facilmente.
# "BGR888" è comodo perché non richiede conversioni di colore extra se OpenCV si aspetta BGR.
# Se usi "RGB888", OpenCV lo interpreterà come BGR, quindi ok.
# Se usi "XBGR8888" o "XRGB8888" (con canale alpha), dovrai fare frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1) # Pausa per permettere alla camera di stabilizzarsi

# MODIFICA: Rimossa la scrittura su video per migliorare le performance live.
# Se vuoi registrare un output per debug, puoi decommentarlo, ma rallenterà il loop.
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("output_realtime.avi", fourcc, 10.0, (640, 480)) # Riduci il framerate se lo usi

# Trackbar HSV (invariato)
def nothing(x): pass
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing) # Valori di default per il bianco
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing) # Tipicamente 180 o 255 per H
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)  # Max 50-70 per bianco
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Parametri globali (invariati)
last_direction = 0
filtered_angle = 0
valid_lane_center = 320
invalid_frame_count = 0
MAX_INVALID_FRAMES = 10
alpha = 0.3

estimated_lane_width = 459
MIN_PLAUSIBLE_LANE_WIDTH = 400
MAX_PLAUSIBLE_LANE_WIDTH = 520
LANE_WIDTH_SMOOTH_ALPHA = 0.01

frame_center_x = 320 # Dato che il frame è 640x480, il centro x è 320

MIN_PEAK_STRENGTH_RATIO = 0.3
MIN_SEPARATION_FOR_TWO_LINES = 400
LANE_SIDE_CONFIDENCE_THRESHOLD = 50
WINDOW_SEARCH_MARGIN = 50
MIN_CONTOUR_AREA_IN_WINDOW = 30

MIN_POINTS_FOR_STABLE_LINE = 4
MIN_PHYSICAL_SEPARATION_IF_BOTH_DETECTED = int(0.5 * MIN_PLAUSIBLE_LANE_WIDTH)

last_frame_had_stable_left = False
last_frame_had_stable_right = False

MAX_CONSECUTIVE_EMPTY_WINDOWS = 3

MIN_POINTS_FOR_CURVE_ANALYSIS = 4
CURVE_X_CHANGE_THRESHOLD = 40


def smooth_angle(current, previous, alpha=0.3):
    return int(alpha * current + (1 - alpha) * previous)

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

def analyze_line_curvature(points_x, min_points=MIN_POINTS_FOR_CURVE_ANALYSIS, x_threshold=CURVE_X_CHANGE_THRESHOLD):
    if len(points_x) < min_points:
        return "unknown"
    x_bottom = points_x[0]
    x_top = points_x[-1]
    delta_x = x_bottom - x_top
    if delta_x < -x_threshold:
        return "left_curve"
    elif delta_x > x_threshold:
        return "right_curve"
    else:
        return "straight"

# CONSIGLIO: La velocità base della PicarX. Dovrai tararla!
FORWARD_SPEED = 1 # Inizia basso, poi aumenta

try: # MODIFICA: Aggiunto try...finally per gestire la chiusura corretta
    # MODIFICA: Cambiato il loop per usare PiCamera2
    # while cap.isOpened():
    while True:
        # MODIFICA: Cattura frame da PiCamera2
        # ret, frame = cap.read()
        # if not ret:
        #     break
        frame = picam2.capture_array()
        # Se Picamera2 è configurata con formato RGBA (es. XRGB8888), converti in BGR
        # if frame.shape[2] == 4:
        #    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        # frame = cv2.resize(frame, (640, 480)) # Non più necessario se la camera è configurata a 640x480

        # CONSIGLIO: I PUNTI PER LA TRASFORMAZIONE PROSPETTICA (tl, bl, tr, br)
        # DOVRANNO ESSERE RICALIBRATI SULLA PISTA REALE CON LA PICARX!
        # Questi valori dipendono dall'angolazione e posizione della camera.
        # Puoi creare uno script separato per aiutarti a trovare questi punti.
        tl, bl, tr, br = (70,260), (0,480), (570,260), (640,480) # VALORI DA RICALIBRARE!
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, matrix, (640, 480))

        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        # CONSIGLIO: I valori HSV per il bianco della pista andranno ritoccati!
        # Usa le trackbar per trovare i valori ottimali in laboratorio.
        l_h, l_s, l_v = cv2.getTrackbarPos("L - H", "Trackbars"), cv2.getTrackbarPos("L - S", "Trackbars"), cv2.getTrackbarPos("L - V", "Trackbars")
        u_h, u_s, u_v = cv2.getTrackbarPos("U - H", "Trackbars"), cv2.getTrackbarPos("U - S", "Trackbars"), cv2.getTrackbarPos("U - V", "Trackbars")
        lower, upper = np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower, upper)

        # --- INIZIO LOGICA ISTOGRAMMA MIGLIORATA ---
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
            if (strong_left_peak and not strong_right_peak):
                single_line_pos_candidate = left_base_candidate
            elif (not strong_left_peak and strong_right_peak):
                single_line_pos_candidate = right_base_candidate
            else:
                single_line_pos_candidate = (left_base_candidate + right_base_candidate) // 2
            reference_center = valid_lane_center
            if single_line_pos_candidate < reference_center - LANE_SIDE_CONFIDENCE_THRESHOLD:
                search_for_left_line = True
                current_left_lane_base = single_line_pos_candidate
                current_right_lane_base = current_left_lane_base + estimated_lane_width
            elif single_line_pos_candidate > reference_center + LANE_SIDE_CONFIDENCE_THRESHOLD:
                search_for_right_line = True
                current_right_lane_base = single_line_pos_candidate
                current_left_lane_base = current_right_lane_base - estimated_lane_width
            else:
                if single_line_pos_candidate < frame_center_x:
                    search_for_left_line = True
                    current_left_lane_base = single_line_pos_candidate
                    current_right_lane_base = current_left_lane_base + estimated_lane_width
                else:
                    search_for_right_line = True
                    current_right_lane_base = single_line_pos_candidate
                    current_left_lane_base = current_right_lane_base - estimated_lane_width
        else:
            search_for_left_line = True
            search_for_right_line = True
        
        initial_hist_search_left = search_for_left_line
        initial_hist_search_right = search_for_right_line
        initial_hist_left_base = current_left_lane_base
        initial_hist_right_base = current_right_lane_base

        if initial_hist_search_left and not initial_hist_search_right:
            if last_frame_had_stable_right and not last_frame_had_stable_left:
                # print("COERENZA: Istogramma -> Sinistra, Precedente -> Destra Stabile. Override a Destra.")
                search_for_left_line = False
                search_for_right_line = True
                current_right_lane_base = initial_hist_left_base
                current_left_lane_base = current_right_lane_base - estimated_lane_width
        elif not initial_hist_search_left and initial_hist_search_right:
            if last_frame_had_stable_left and not last_frame_had_stable_right:
                # print("COERENZA: Istogramma -> Destra, Precedente -> Sinistra Stabile. Override a Sinistra.")
                search_for_right_line = False
                search_for_left_line = True
                current_left_lane_base = initial_hist_right_base
                current_right_lane_base = current_left_lane_base + estimated_lane_width
        elif initial_hist_search_left and initial_hist_search_right:
            detected_base_separation = abs(initial_hist_right_base - initial_hist_left_base)
            if detected_base_separation > MIN_SEPARATION_FOR_TWO_LINES and \
               detected_base_separation < MIN_PHYSICAL_SEPARATION_IF_BOTH_DETECTED:
                # print(f"COERENZA: Istogramma -> Entrambe, ma troppo vicine ({detected_base_separation:.0f}px). Verifico stato precedente.")
                if last_frame_had_stable_right and not last_frame_had_stable_left:
                    # print("COERENZA: Precedente -> Destra Stabile. Forzo solo Destra.")
                    search_for_left_line = False
                    search_for_right_line = True
                    current_right_lane_base = initial_hist_right_base
                    current_left_lane_base = current_right_lane_base - estimated_lane_width
                elif last_frame_had_stable_left and not last_frame_had_stable_right:
                    # print("COERENZA: Precedente -> Sinistra Stabile. Forzo solo Sinistra.")
                    search_for_right_line = False
                    search_for_left_line = True
                    current_left_lane_base = initial_hist_left_base
                    current_right_lane_base = current_left_lane_base + estimated_lane_width
                else:
                    # print("COERENZA: Basi vicine, stato precedente ambiguo. Scelgo il picco più forte.")
                    if left_peak_value >= right_peak_value:
                        search_for_right_line = False
                        search_for_left_line = True
                        current_left_lane_base = initial_hist_left_base
                        current_right_lane_base = current_left_lane_base + estimated_lane_width
                    else:
                        search_for_left_line = False
                        search_for_right_line = True
                        current_right_lane_base = initial_hist_right_base
                        current_left_lane_base = current_right_lane_base - estimated_lane_width
        
        current_left_lane_base = np.clip(current_left_lane_base, 0, warped.shape[1]-1)
        current_right_lane_base = np.clip(current_right_lane_base, 0, warped.shape[1]-1)
        # --- FINE LOGICA ISTOGRAMMA MIGLIORATA E COERENZA ---

        # Sliding window
        y_sliding = warped.shape[0] - 1
        window_height = 40
        
        left_x_points, right_x_points = [], []
        annotated_warped = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        iter_left_base = current_left_lane_base
        iter_right_base = current_right_lane_base
        num_windows = warped.shape[0] // window_height

        left_consecutive_misses = 0
        right_consecutive_misses = 0
        actively_tracking_left = search_for_left_line
        actively_tracking_right = search_for_right_line

        for window_idx in range(num_windows):
            win_y_high = warped.shape[0] - (window_idx * window_height)
            win_y_low = warped.shape[0] - ((window_idx + 1) * window_height)
            
            if actively_tracking_left:
                win_xleft_low = max(0, iter_left_base - WINDOW_SEARCH_MARGIN)
                win_xleft_high = min(warped.shape[1], iter_left_base + WINDOW_SEARCH_MARGIN)
                left_window_roi = mask[win_y_low:win_y_high, win_xleft_low:win_xleft_high]
                
                contours_l, _ = cv2.findContours(left_window_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                found_left_pixels_this_window = False
                if contours_l:
                    largest_l = max(contours_l, key=cv2.contourArea)
                    if cv2.contourArea(largest_l) > MIN_CONTOUR_AREA_IN_WINDOW:
                        M_l = cv2.moments(largest_l)
                        if M_l["m00"] != 0:
                            cx_l = int(M_l["m10"] / M_l["m00"])
                            iter_left_base = win_xleft_low + cx_l
                            left_x_points.append(iter_left_base)
                            left_consecutive_misses = 0
                            found_left_pixels_this_window = True
                
                if not found_left_pixels_this_window:
                    left_consecutive_misses += 1
                    if left_consecutive_misses > MAX_CONSECUTIVE_EMPTY_WINDOWS:
                        actively_tracking_left = False

            if actively_tracking_right:
                win_xright_low = max(0, iter_right_base - WINDOW_SEARCH_MARGIN)
                win_xright_high = min(warped.shape[1], iter_right_base + WINDOW_SEARCH_MARGIN)
                right_window_roi = mask[win_y_low:win_y_high, win_xright_low:win_xright_high]

                contours_r, _ = cv2.findContours(right_window_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                found_right_pixels_this_window = False
                if contours_r:
                    largest_r = max(contours_r, key=cv2.contourArea)
                    if cv2.contourArea(largest_r) > MIN_CONTOUR_AREA_IN_WINDOW:
                        M_r = cv2.moments(largest_r)
                        if M_r["m00"] != 0:
                            cx_r = int(M_r["m10"] / M_r["m00"])
                            iter_right_base = win_xright_low + cx_r
                            right_x_points.append(iter_right_base)
                            right_consecutive_misses = 0
                            found_right_pixels_this_window = True

                if not found_right_pixels_this_window:
                    right_consecutive_misses += 1
                    if right_consecutive_misses > MAX_CONSECUTIVE_EMPTY_WINDOWS:
                        actively_tracking_right = False
            
            annotated_warped = draw_windows(annotated_warped, win_y_low, win_y_high,
                                            iter_left_base, iter_right_base,
                                            draw_left_flag=search_for_left_line and actively_tracking_left,
                                            draw_right_flag=search_for_right_line and actively_tracking_right)

        curve_analysis_L = "unknown"
        curve_analysis_R = "unknown"

        if search_for_left_line and len(left_x_points) > 0:
            curve_analysis_L = analyze_line_curvature(left_x_points)
        if search_for_right_line and len(right_x_points) > 0:
            curve_analysis_R = analyze_line_curvature(right_x_points)

        # print(f"Pre-Curv: HistSeesL={search_for_left_line}, HistSeesR={search_for_right_line}. CurveL: {curve_analysis_L}, CurveR: {curve_analysis_R}")

        final_search_left = False
        final_search_right = False
        final_left_x_points = []
        final_right_x_points = []

        if search_for_right_line and curve_analysis_R == "left_curve":
            # print("CURV_OVERRIDE: Rilevata DESTRA, ma forma CURVA SINISTRA -> classificata come SINISTRA.")
            final_search_left = True
            final_left_x_points = list(right_x_points)
            if search_for_left_line and curve_analysis_L != "right_curve":
                # print("CURV_OVERRIDE: Conflitto D->S e Sinistra originale valida. Mantenuta Sinistra originale.")
                final_left_x_points = list(left_x_points)
            else:
                final_search_right = False
        elif search_for_left_line and curve_analysis_L == "right_curve":
            # print("CURV_OVERRIDE: Rilevata SINISTRA, ma forma CURVA DESTRA -> classificata come DESTRA.")
            final_search_right = True
            final_right_x_points = list(left_x_points)
            if search_for_right_line and curve_analysis_R != "left_curve":
                # print("CURV_OVERRIDE: Conflitto S->D e Destra originale valida. Mantenuta Destra originale.")
                final_right_x_points = list(right_x_points)
            else:
                final_search_left = False
        else:
            # print("CURV_OVERRIDE: Nessun override forte da curvatura applicato o forme coerenti.")
            if search_for_left_line and (curve_analysis_L == "left_curve" or curve_analysis_L == "straight" or curve_analysis_L == "unknown"):
                final_search_left = True
                final_left_x_points = list(left_x_points)
            if search_for_right_line and (curve_analysis_R == "right_curve" or curve_analysis_R == "straight" or curve_analysis_R == "unknown"):
                final_search_right = True
                final_right_x_points = list(right_x_points)

        if final_search_left and not final_left_x_points and search_for_left_line and len(left_x_points) > 0 and curve_analysis_L != "right_curve":
            # print("CURV_POST_CHECK: Ripristino linea sinistra originale (non coinvolta in override S->D).")
            final_left_x_points = list(left_x_points)
        if final_search_right and not final_right_x_points and search_for_right_line and len(right_x_points) > 0 and curve_analysis_R != "left_curve":
            # print("CURV_POST_CHECK: Ripristino linea destra originale (non coinvolta in override D->S).")
            final_right_x_points = list(right_x_points)

        # print(f"Post-Curv: Final_S_L={final_search_left} ({len(final_left_x_points)}pts), Final_S_R={final_search_right} ({len(final_right_x_points)}pts)")
        # --- FINE ANALISI CURVATURA E OVERRIDE ---

        new_lane_center_calculated_this_frame = False
        current_frame_lane_center = valid_lane_center

        if final_left_x_points and final_right_x_points:
            left_avg = int(np.mean(final_left_x_points))
            right_avg = int(np.mean(final_right_x_points))
            current_detected_width = right_avg - left_avg
            EDGE_ZONE_WIDTH = 30
            left_is_at_edge = left_avg < EDGE_ZONE_WIDTH
            right_is_at_edge = right_avg > (warped.shape[1] - EDGE_ZONE_WIDTH)
            width_is_plausible = MIN_PLAUSIBLE_LANE_WIDTH < current_detected_width < MAX_PLAUSIBLE_LANE_WIDTH
            temp_calculated_center = -1

            if width_is_plausible:
                if not left_is_at_edge and not right_is_at_edge:
                    temp_calculated_center = (left_avg + right_avg) // 2
                    estimated_lane_width = int(LANE_WIDTH_SMOOTH_ALPHA * current_detected_width + \
                                               (1 - LANE_WIDTH_SMOOTH_ALPHA) * estimated_lane_width)
                elif not left_is_at_edge and right_is_at_edge:
                    temp_calculated_center = left_avg + estimated_lane_width // 2
                elif left_is_at_edge and not right_is_at_edge:
                    temp_calculated_center = right_avg - estimated_lane_width // 2
            else:
                if not left_is_at_edge and not right_is_at_edge:
                    if left_avg < frame_center_x:
                        temp_calculated_center = left_avg + estimated_lane_width // 2
                    elif right_avg > frame_center_x:
                        temp_calculated_center = right_avg - estimated_lane_width // 2
                elif not left_is_at_edge:
                    temp_calculated_center = left_avg + estimated_lane_width // 2
                elif not right_is_at_edge:
                    temp_calculated_center = right_avg - estimated_lane_width // 2

            if temp_calculated_center != -1:
                current_frame_lane_center = temp_calculated_center
                valid_lane_center = current_frame_lane_center
                invalid_frame_count = 0
                new_lane_center_calculated_this_frame = True
        
        if not new_lane_center_calculated_this_frame:
            if final_left_x_points:
                left_avg = int(np.mean(final_left_x_points))
                if MIN_PLAUSIBLE_LANE_WIDTH < estimated_lane_width < MAX_PLAUSIBLE_LANE_WIDTH:
                    current_frame_lane_center = left_avg + estimated_lane_width // 2
                    valid_lane_center = current_frame_lane_center
                    invalid_frame_count = 0
                    new_lane_center_calculated_this_frame = True
            elif final_right_x_points:
                right_avg = int(np.mean(final_right_x_points))
                if MIN_PLAUSIBLE_LANE_WIDTH < estimated_lane_width < MAX_PLAUSIBLE_LANE_WIDTH:
                    current_frame_lane_center = right_avg - estimated_lane_width // 2
                    valid_lane_center = current_frame_lane_center
                    invalid_frame_count = 0
                    new_lane_center_calculated_this_frame = True
        
        if not new_lane_center_calculated_this_frame:
            invalid_frame_count += 1
        
        final_lane_center = current_frame_lane_center

        if invalid_frame_count > MAX_INVALID_FRAMES:
            # MODIFICA: Azione motori quando si perdono troppi frame
            picarx.stop() 
            print(f"STOP: {invalid_frame_count} frame consecutivi senza rilevamento affidabile.")
            # CONSIGLIO: Potresti voler uscire dal loop o avere una strategia di recupero qui
            # Per ora, continuerà a provare, ma con i motori fermi finché MAX_INVALID_FRAMES è superato.
            # Se vuoi che si fermi e basta, aggiungi 'break' qui.
            # Se vuoi che continui ma con una strategia (es. vai dritto piano), implementala.
        else:
            # MODIFICA: Azione motori quando il rilevamento è OK
            error = final_lane_center - frame_center_x
            raw_angle = -int(error / 3.5) # Il divisore 3.5 è un fattore P, da tarare
            raw_angle = np.clip(raw_angle, -40, 40) # Angolo massimo servo PicarX
            angle = smooth_angle(raw_angle, last_direction, alpha)
            angle = np.clip(angle, -35, 35) # Limita l'angolo effettivo per stabilità
            last_direction = angle
            
            picarx.forward(FORWARD_SPEED) # Muovi la macchina in avanti
            picarx.set_dir_servo_angle(angle) # Imposta l'angolo di sterzata

        # Aggiorna lo stato di stabilità per il prossimo frame
        if len(final_left_x_points) >= MIN_POINTS_FOR_STABLE_LINE:
            last_frame_had_stable_left = True
        else:
            last_frame_had_stable_left = False

        if len(final_right_x_points) >= MIN_POINTS_FOR_STABLE_LINE:
            last_frame_had_stable_right = True
        else:
            last_frame_had_stable_right = False
            
        # ANNOTAZIONI
        cv2.line(annotated_warped, (final_lane_center, 0), (final_lane_center, warped.shape[0]), (0,255,255), 2)
        cv2.line(annotated_warped, (frame_center_x, 0), (frame_center_x, warped.shape[0]), (255,255,255), 1)
        cv2.putText(annotated_warped, f"Angle: {angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(annotated_warped, f"L_pts: {len(final_left_x_points)} R_pts: {len(final_right_x_points)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.putText(annotated_warped, f"Est.W: {estimated_lane_width:.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        if (search_for_right_line and curve_analysis_R == "left_curve" and final_search_left and not final_search_right) or \
           (search_for_left_line and curve_analysis_L == "right_curve" and final_search_right and not final_search_left):
            cv2.putText(annotated_warped, "CURV OVERRIDE!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

        # MODIFICA: Scrittura video opzionale
        # if 'out' in locals() and out is not None: # Controlla se out è definito
        #    out.write(annotated_warped)

        cv2.imshow("Annotated Warped", annotated_warped)
        # cv2.imshow("Original", frame) # Decommenta per vedere l'originale
        # cv2.imshow("Mask", mask) # Decommenta per vedere la maschera HSV

        if cv2.waitKey(1) == 27: # Premi ESC per uscire
            print("Uscita richiesta dall'utente.")
            break
finally: # MODIFICA: Blocco finally per assicurare lo stop dei motori e il rilascio delle risorse
    print("Fermando i motori e rilasciando le risorse...")
    picarx.set_dir_servo_angle(0) # Resetta l'angolo del servo
    picarx.stop()
    picam2.stop() # Ferma la PiCamera2
    # MODIFICA: Rimossa la chiusura del video writer se non usato
    # if 'out' in locals() and out is not None:
    #    out.release()
    cv2.destroyAllWindows()
    print("Risorse rilasciate. Programma terminato.")