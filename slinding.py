import cv2
import numpy as np
from picarx import Picarx # DECOMMENTA
from picamera2 import Picamera2 # DECOMMENTA
import time # UTILE PER LA CAMERA E CONTROLLI

# Lettura da video -> DA COMMENTARE O RIMUOVERE
# cap = cv2.VideoCapture("pista.MP4")

# Inizializzazione PicarX e Picamera2
picarx = Picarx() # DECOMMENTA
picam2 = Picamera2() # NUOVO: Inizializza oggetto camera
camera_config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}) # NUOVO: Configura la camera
# Nota: OpenCV usa BGR, Picamera2 cattura in RGB888. Potrebbe essere necessaria una conversione.
picam2.configure(camera_config) # NUOVO: Applica configurazione
picam2.start() # NUOVO: Avvia la camera
time.sleep(1) # NUOVO: Dai tempo alla camera di avviarsi

# Scrittura video in output -> DA COMMENTARE O RIMUOVERE
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("output_improved.avi", fourcc, 20.0, (640, 480))

# Trackbar HSV (invariato, utile per tuning live)
def nothing(x): pass
cv2.namedWindow("Trackbars")
# ... (resto delle trackbar invariato) ...
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
MIN_PLAUSIBLE_LANE_WIDTH = 458
MAX_PLAUSIBLE_LANE_WIDTH = 460 # Potresti dover allargare leggermente questo range per la realtà
LANE_WIDTH_SMOOTH_ALPHA = 0.01

frame_center_x = 320

MIN_PEAK_STRENGTH_RATIO = 0.3
MIN_SEPARATION_FOR_TWO_LINES = 80
LANE_SIDE_CONFIDENCE_THRESHOLD = 50
WINDOW_SEARCH_MARGIN = 50
MIN_CONTOUR_AREA_IN_WINDOW = 30

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

try: # NUOVO: Aggiungi un blocco try...finally per pulizia
    # while cap.isOpened(): # SOSTITUISCI CON:
    while True:
        # ret, frame = cap.read() # SOSTITUISCI CON:
        frame = picam2.capture_array() # NUOVO: Cattura frame dalla Picamera2
        # Picamera2 (con format RGB888) dà un array RGB. OpenCV spesso si aspetta BGR.
        # La tua trasformazione in HSV cv2.cvtColor(warped, cv2.COLOR_BGR2HSV) si aspetta BGR.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # NUOVO: Converti RGB in BGR

        # if not ret: # RIMUOVI (non necessario con capture_array se la camera funziona)
        #     break
        
        # frame = cv2.resize(frame, (640, 480)) # Già gestito dalla configurazione della camera

        # Trasformazione prospettica (invariata)
        # tl, bl, tr, br = (70,400), (0,480), (570,400), (640,480) # CONTROLLA QUESTI VALORI SULLA MACCHINA REALE!
        # Potrebbero necessitare di ricalibrazione per la camera della PicarX
        # Suggerimento: crea uno script separato per calibrare questi punti sulla PicarX
        # mostrando il frame originale e disegnandoci sopra i punti per regolarli.
        # PER ORA, USIAMO QUELLI DEL VIDEO, MA PREPARATI A CAMBIARLI:
        tl, bl, tr, br = (70,400), (0,480), (570,400), (640,480)
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, matrix, (640, 480))

        # Threshold HSV (invariato)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        l_h, l_s, l_v = cv2.getTrackbarPos("L - H", "Trackbars"), cv2.getTrackbarPos("L - S", "Trackbars"), cv2.getTrackbarPos("L - V", "Trackbars")
        u_h, u_s, u_v = cv2.getTrackbarPos("U - H", "Trackbars"), cv2.getTrackbarPos("U - S", "Trackbars"), cv2.getTrackbarPos("U - V", "Trackbars")
        lower, upper = np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower, upper)

        # --- LOGICA ISTOGRAMMA (INVARIATA DAL TUO ULTIMO CODICE FUNZIONANTE) ---
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
            # print("HIST: Due linee rilevate (da metà istogramma).")
        elif (strong_left_peak and not strong_right_peak) or \
             (not strong_left_peak and strong_right_peak) or \
             (strong_left_peak and strong_right_peak and \
              abs(right_base_candidate - left_base_candidate) <= MIN_SEPARATION_FOR_TWO_LINES):
            single_line_pos_candidate = 0
            # print_prefix = "HIST: "
            if (strong_left_peak and not strong_right_peak):
                single_line_pos_candidate = left_base_candidate
                # print_prefix += "Solo picco sx hist -> "
            elif (not strong_left_peak and strong_right_peak):
                single_line_pos_candidate = right_base_candidate
                # print_prefix += "Solo picco dx hist -> "
            else:
                single_line_pos_candidate = (left_base_candidate + right_base_candidate) // 2
                # print_prefix += "Picchi vicini (media) -> "
            reference_center = valid_lane_center
            if single_line_pos_candidate < reference_center - LANE_SIDE_CONFIDENCE_THRESHOLD:
                search_for_left_line = True
                current_left_lane_base = single_line_pos_candidate
                current_right_lane_base = current_left_lane_base + estimated_lane_width
                # print(f"{print_prefix}Classificata SINISTRA (rispetto a V_L_C={reference_center}).")
            elif single_line_pos_candidate > reference_center + LANE_SIDE_CONFIDENCE_THRESHOLD:
                search_for_right_line = True
                current_right_lane_base = single_line_pos_candidate
                current_left_lane_base = current_right_lane_base - estimated_lane_width
                # print(f"{print_prefix}Classificata DESTRA (rispetto a V_L_C={reference_center}).")
            else:
                if single_line_pos_candidate < frame_center_x:
                    search_for_left_line = True
                    current_left_lane_base = single_line_pos_candidate
                    current_right_lane_base = current_left_lane_base + estimated_lane_width
                    # print(f"{print_prefix}Classificata SINISTRA (ambigua V_L_C, fallback su frame_center_x).")
                else:
                    search_for_right_line = True
                    current_right_lane_base = single_line_pos_candidate
                    current_left_lane_base = current_right_lane_base - estimated_lane_width
                    # print(f"{print_prefix}Classificata DESTRA (ambigua V_L_C, fallback su frame_center_x).")
        else:
            search_for_left_line = True
            search_for_right_line = True
            # print(f"HIST: Nessun picco chiaro, uso stime V_L_C={valid_lane_center}, Est.W={estimated_lane_width:.0f}.")

        current_left_lane_base = np.clip(current_left_lane_base, 0, warped.shape[1]-1)
        current_right_lane_base = np.clip(current_right_lane_base, 0, warped.shape[1]-1)
        # --- FINE LOGICA ISTOGRAMMA ---

        # Sliding window (invariata)
        # ... (tutta la logica delle sliding windows, calcolo centro corsia, etc. rimane invariata) ...
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
        current_frame_lane_center = valid_lane_center
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
        elif left_x_points:
            left_avg = int(np.mean(left_x_points))
            if MIN_PLAUSIBLE_LANE_WIDTH < estimated_lane_width < MAX_PLAUSIBLE_LANE_WIDTH:
                current_frame_lane_center = left_avg + estimated_lane_width // 2
                valid_lane_center = current_frame_lane_center
                invalid_frame_count = 0
                new_lane_center_calculated_this_frame = True
        elif right_x_points:
            right_avg = int(np.mean(right_x_points))
            if MIN_PLAUSIBLE_LANE_WIDTH < estimated_lane_width < MAX_PLAUSIBLE_LANE_WIDTH:
                current_frame_lane_center = right_avg - estimated_lane_width // 2
                valid_lane_center = current_frame_lane_center
                invalid_frame_count = 0
                new_lane_center_calculated_this_frame = True

        if not new_lane_center_calculated_this_frame:
            invalid_frame_count += 1
        final_lane_center = current_frame_lane_center

        if invalid_frame_count > MAX_INVALID_FRAMES:
            picarx.stop() # DECOMMENTA per fermare la macchina
            print(f"STOP: {invalid_frame_count} frame consecutivi senza rilevamento affidabile.")
            # Qui potresti voler uscire dal loop o attendere un input per riprovare
            # Per ora, continuiamo a processare i frame ma la macchina è ferma.
            # Se vuoi fermare tutto lo script:
            # break
        else:
            # Calcolo angolo e movimento
            error = final_lane_center - frame_center_x
            raw_angle = -int(error / 3.5) # Potrebbe necessitare di tuning per la PicarX
            raw_angle = np.clip(raw_angle, -40, 40)
            angle = smooth_angle(raw_angle, last_direction, alpha)
            angle = np.clip(angle, -35, 35) # Limiti di sterzata PicarX
            last_direction = angle

            # print(f"L:{len(left_x_points)} R:{len(right_x_points)} EstWidth:{estimated_lane_width:.0f} LaneCenter:{final_lane_center} Err:{error} Angle:{angle}")
            picarx.forward(20) # DECOMMENTA: Imposta una velocità (es. 20). Regola se necessario!
            picarx.set_dir_servo_angle(angle) # DECOMMENTA: Imposta l'angolo di sterzo

        # Annotazioni (invariate)
        cv2.line(annotated_warped, (final_lane_center, 0), (final_lane_center, warped.shape[0]), (0,255,255), 2)
        cv2.line(annotated_warped, (frame_center_x, 0), (frame_center_x, warped.shape[0]), (255,255,255), 1)
        cv2.putText(annotated_warped, f"Angle: {angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(annotated_warped, f"L_pts: {len(left_x_points)} R_pts: {len(right_x_points)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.putText(annotated_warped, f"Est.W: {estimated_lane_width:.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        # out.write(annotated_warped) # RIMUOVI O COMMENTA
        cv2.imshow("Annotated Warped", annotated_warped)
        # cv2.imshow("Original", frame) # Utile per calibrare la prospettiva
        # cv2.imshow("Mask", mask) # Utile per calibrare HSV

        if cv2.waitKey(1) == 27: # ESC per uscire
            break
        
        # time.sleep(0.01) # NUOVO: Aggiungi un piccolo delay se il loop è troppo veloce o per dare respiro alla CPU

finally: # NUOVO: Blocco finally per assicurare lo stop
    # cap.release() # RIMUOVI O COMMENTA
    # out.release() # RIMUOVI O COMMENTA
    print("Uscita dal programma...")
    picarx.stop() # DECOMMENTA E ASSICURATI CHE SIA QUI
    picam2.stop() # NUOVO: Ferma la camera
    cv2.destroyAllWindows()