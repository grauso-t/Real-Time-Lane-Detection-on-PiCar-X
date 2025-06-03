import cv2
import numpy as np
import time
from picamera2 import Picamera2 # Per la fotocamera
from picarx import Picarx      # Per il controllo dei motori

# --- I TUOI PARAMETRI GLOBALI (INVARIATI) ---
# Trackbar HSV (Valori fissi se headless, altrimenti puoi decommentare e usare la GUI)
# def nothing(x): pass
# cv2.namedWindow("Trackbars")
# cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing) 
# cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing) 
# cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Valori HSV Fissi (DA IMPOSTARE CON I TUOI VALORI OTTIMALI TROVATI CON LE TRACKBAR)
L_H_FIX, L_S_FIX, L_V_FIX = 0, 0, 200
U_H_FIX, U_S_FIX, U_V_FIX = 255, 50, 255

# Parametri di Guida e Frame
last_direction = 0
valid_lane_center = 320 
invalid_frame_count = 0
MAX_INVALID_FRAMES = 100000 
alpha = 0.3 
frame_height, frame_width, frame_center_x = 480, 640, 320

# Parametri Larghezza Corsia
estimated_lane_width = 459 
MIN_PLAUSIBLE_LANE_WIDTH_ESTIMATION = 380 
MAX_PLAUSIBLE_LANE_WIDTH_ESTIMATION = 550 
LANE_WIDTH_SMOOTH_ALPHA = 0.01 

# Parametri Istogramma
MIN_PEAK_STRENGTH_RATIO = 0.3 
MIN_SEPARATION_FOR_TWO_LINES_HIST = 300 
NEAR_VLC_THRESHOLD = 60 

# Parametri Sliding Windows
WINDOW_SEARCH_MARGIN = 50
MIN_CONTOUR_AREA_IN_WINDOW = 30 
MIN_POINTS_FOR_STABLE_LINE = 2 
MAX_CONSECUTIVE_EMPTY_WINDOWS = 3 
MIN_POINTS_FOR_SLOPE_ESTIMATION = 2 

last_valid_line_side = None 

# Parametri Controllo Curvatura
MIN_POINTS_FOR_CURVE_ANALYSIS = 2 
CURVE_X_CHANGE_THRESHOLD = 30  
CURVE_ROI_Y_START_PERCENT = 0.40  
CURVE_ROI_Y_END_PERCENT = 0.98    
CURVE_ROI_X_CENTER_OFFSET_PERCENT = 0.015

# --- FUNZIONI HELPER (INVARIATE) ---
def smooth_angle(current, previous, alpha_val=0.3):
    return int(alpha_val * current + (1 - alpha_val) * previous)
def draw_windows_unified(image, win_y_low, win_y_high, current_x_base, color=(255,0,255), margin=WINDOW_SEARCH_MARGIN):
    cv2.rectangle(image, (max(0, current_x_base - margin), win_y_low),
                  (min(frame_width-1, current_x_base + margin), win_y_high), color, 2)
    return image
def analyze_line_curvature(points_x_roi, min_points=MIN_POINTS_FOR_CURVE_ANALYSIS, x_threshold=CURVE_X_CHANGE_THRESHOLD):
    if len(points_x_roi) < min_points: return "unknown_roi_few_pts"
    x_bottom_roi, x_top_roi = points_x_roi[0], points_x_roi[-1] 
    delta_x_roi = x_bottom_roi - x_top_roi 
    curve_type_candidate = "straight_roi"
    if delta_x_roi < -x_threshold: curve_type_candidate = "left_curve_roi"
    elif delta_x_roi > x_threshold: curve_type_candidate = "right_curve_roi"
    else: return "straight_roi" 
    return curve_type_candidate

# --- INIZIALIZZAZIONE E LOOP PRINCIPALE ---
try:
    # --- INIZIALIZZAZIONE FOTOCAMERA ---
    picam2 = Picamera2()
    # Configura la risoluzione e il formato (RGB888 è comune con Picamera2, OpenCV si aspetta BGR)
    camera_config = picam2.create_preview_configuration(main={"size": (frame_width, frame_height), "format": "RGB888"})
    picam2.configure(camera_config)
    picam2.start()
    print("Fotocamera avviata e configurata.")
    time.sleep(1) # Lascia tempo alla fotocamera di stabilizzarsi

    # --- INIZIALIZZAZIONE PICARX ---
    px = Picarx()
    print("PiCarX inizializzata.")
    # Se necessario, calibra lo sterzo: px.set_dir_servo_offset(TUO_OFFSET)
    # px.set_dir_servo_offset(-5) # Esempio di offset, rimuovi o tara

    while True: # Loop infinito fino a interruzione (Ctrl+C)
        current_vlc_for_histogram_check = valid_lane_center 
        
        # --- Cattura Frame dalla PiCamera ---
        frame_rgb = picam2.capture_array() # Cattura come array NumPy RGB
        if frame_rgb is None:
            print("Errore: Impossibile catturare il frame.")
            time.sleep(0.1)
            continue
        # Converte da RGB (formato di Picamera2) a BGR (formato di OpenCV)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 1. TRASFORMAZIONE PROSPETTICA e 2. MASCHERA HSV
        tl, bl, tr, br = (70,260), (0,480), (570,260), (640,480) 
        pts1, pts2 = np.float32([tl, bl, tr, br]), np.float32([[0,0], [0,frame_height], [frame_width,0], [frame_width,frame_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, matrix, (frame_width, frame_height))
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        
        # Usa i valori HSV fissi (o decommenta getTrackbarPos se usi la GUI)
        # l_h,l_s,l_v = cv2.getTrackbarPos("L - H", "Trackbars"), ..., ...
        # u_h,u_s,u_v = cv2.getTrackbarPos("U - H", "Trackbars"), ..., ...
        l_h,l_s,l_v = L_H_FIX, L_S_FIX, L_V_FIX
        u_h,u_s,u_v = U_H_FIX, U_S_FIX, U_V_FIX
        lower_white, upper_white = np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # --- LA TUA LOGICA DI LANE DETECTION (Livelli 1A, 1B, 1C, Livello 2, Calcolo Centro) ---
        # Questa parte è ESATTAMENTE come nel codice che mi hai mandato
        # ... (LIVELLO 1A: ISTOGRAMMA) ...
        histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
        midpoint_hist = histogram.shape[0] // 2
        left_half_hist, right_half_hist = histogram[:midpoint_hist], histogram[midpoint_hist:]
        left_base_candidate = np.argmax(left_half_hist) if np.any(left_half_hist) else 0
        left_peak_value = np.max(left_half_hist) if np.any(left_half_hist) else 0
        right_base_candidate = (np.argmax(right_half_hist) + midpoint_hist) if np.any(right_half_hist) else midpoint_hist
        right_peak_value = np.max(right_half_hist) if np.any(right_half_hist) else 0
        max_hist_value = np.max(histogram) if np.any(histogram) else 1
        initial_search_left, initial_search_right = False, False
        current_left_lane_base_hist = current_vlc_for_histogram_check - estimated_lane_width // 2 
        current_right_lane_base_hist = current_vlc_for_histogram_check + estimated_lane_width // 2 
        strong_left_peak = left_peak_value > (max_hist_value * MIN_PEAK_STRENGTH_RATIO)
        strong_right_peak = right_peak_value > (max_hist_value * MIN_PEAK_STRENGTH_RATIO)
        hist_debug_msg = ""
        if strong_left_peak and strong_right_peak and abs(right_base_candidate - left_base_candidate) > MIN_SEPARATION_FOR_TWO_LINES_HIST:
            initial_search_left, initial_search_right = True, True; current_left_lane_base_hist, current_right_lane_base_hist = left_base_candidate, right_base_candidate; hist_debug_msg = "HIST: Two strong peaks"
        elif strong_left_peak or strong_right_peak: 
            single_line_pos_candidate = left_base_candidate if strong_left_peak and not strong_right_peak else (right_base_candidate if not strong_left_peak and strong_right_peak else (left_base_candidate + right_base_candidate) // 2)
            is_near_or_intersecting_vlc_prev = abs(single_line_pos_candidate - current_vlc_for_histogram_check) < NEAR_VLC_THRESHOLD
            if is_near_or_intersecting_vlc_prev:
                hist_debug_msg = f"HIST: Peak@{single_line_pos_candidate} near VLC(prev)@{current_vlc_for_histogram_check}."
                if last_valid_line_side == 'right': initial_search_right, current_right_lane_base_hist, current_left_lane_base_hist, hist_debug_msg = True, single_line_pos_candidate, current_right_lane_base_hist - estimated_lane_width, hist_debug_msg + " Use LastValid=R"
                elif last_valid_line_side == 'left': initial_search_left, current_left_lane_base_hist, current_right_lane_base_hist, hist_debug_msg = True, single_line_pos_candidate, current_left_lane_base_hist + estimated_lane_width, hist_debug_msg + " Use LastValid=L"
                else: 
                    hist_debug_msg += " No LastValid -> Fallback FC"
                    if single_line_pos_candidate < frame_center_x : initial_search_left, current_left_lane_base_hist, current_right_lane_base_hist = True, single_line_pos_candidate, current_left_lane_base_hist + estimated_lane_width
                    else: initial_search_right, current_right_lane_base_hist, current_left_lane_base_hist = True, single_line_pos_candidate, current_right_lane_base_hist - estimated_lane_width
            else: 
                hist_debug_msg = f"HIST: Peak@{single_line_pos_candidate} vs VLC(prev)@{current_vlc_for_histogram_check}"
                ambiguous_narrow_threshold = NEAR_VLC_THRESHOLD / 2 
                if single_line_pos_candidate < current_vlc_for_histogram_check - ambiguous_narrow_threshold: initial_search_left, current_left_lane_base_hist, current_right_lane_base_hist = True, single_line_pos_candidate, current_left_lane_base_hist + estimated_lane_width
                elif single_line_pos_candidate > current_vlc_for_histogram_check + ambiguous_narrow_threshold: initial_search_right, current_right_lane_base_hist, current_left_lane_base_hist = True, single_line_pos_candidate, current_right_lane_base_hist - estimated_lane_width
                else: 
                    hist_debug_msg += " -> Ambiguous vs VLC(prev), Fallback FC"
                    if single_line_pos_candidate < frame_center_x : initial_search_left, current_left_lane_base_hist, current_right_lane_base_hist = True, single_line_pos_candidate, current_left_lane_base_hist + estimated_lane_width
                    else: initial_search_right, current_right_lane_base_hist, current_left_lane_base_hist = True, single_line_pos_candidate, current_right_lane_base_hist - estimated_lane_width
        else: initial_search_left, initial_search_right, hist_debug_msg = True, True , "HIST: No strong peaks, search both"
        current_left_lane_base_hist, current_right_lane_base_hist = np.clip(current_left_lane_base_hist,0,frame_width-1), np.clip(current_right_lane_base_hist,0,frame_width-1)
        
        # ... (LIVELLO 1B: SLIDING WINDOWS - identica al tuo codice) ...
        annotated_warped = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # annotated_warped deve essere definita prima di usarla
        window_height, num_windows = 40, frame_height // 40; all_left_x_pts, all_left_y_pts, all_right_x_pts, all_right_y_pts = [], [], [], []
        iter_left_base, iter_right_base = current_left_lane_base_hist, current_right_lane_base_hist
        left_consecutive_misses, right_consecutive_misses = 0, 0
        actively_tracking_left, actively_tracking_right = initial_search_left, initial_search_right
        for window_idx in range(num_windows):
            win_y_high, win_y_low, win_y_center = frame_height-(window_idx*window_height), frame_height-((window_idx+1)*window_height), (frame_height-(window_idx*window_height) + frame_height-((window_idx+1)*window_height))//2
            if actively_tracking_left: # Logica per linea sinistra (copiata dalla tua versione)
                current_search_x_L = iter_left_base 
                win_xL_low, win_xL_high = max(0, current_search_x_L - WINDOW_SEARCH_MARGIN), min(frame_width-1, current_search_x_L + WINDOW_SEARCH_MARGIN)
                contours_l, _ = cv2.findContours(mask[win_y_low:win_y_high, win_xL_low:win_xL_high], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                found_L = False
                if contours_l:
                    largest_l = max(contours_l, key=cv2.contourArea)
                    if cv2.contourArea(largest_l) > MIN_CONTOUR_AREA_IN_WINDOW and cv2.moments(largest_l)["m00"] != 0:
                        iter_left_base = win_xL_low + int(cv2.moments(largest_l)["m10"] / cv2.moments(largest_l)["m00"])
                        all_left_x_pts.append(iter_left_base); all_left_y_pts.append(win_y_center)
                        left_consecutive_misses, found_L = 0, True
                        if initial_search_left: draw_windows_unified(annotated_warped, win_y_low, win_y_high, iter_left_base, color=(255,0,255))
                if not found_L:
                    left_consecutive_misses += 1
                    x_to_draw_empty_L = current_search_x_L 
                    if left_consecutive_misses <= MAX_CONSECUTIVE_EMPTY_WINDOWS and len(all_left_x_pts) >= MIN_POINTS_FOR_SLOPE_ESTIMATION:
                        dx_L = all_left_x_pts[-1] - all_left_x_pts[-2] if len(all_left_x_pts) >= 2 else 0
                        x_to_draw_empty_L = np.clip(iter_left_base + dx_L, 0, frame_width-1) 
                        iter_left_base = x_to_draw_empty_L 
                        if initial_search_left: draw_windows_unified(annotated_warped, win_y_low, win_y_high, x_to_draw_empty_L, color=(180,0,180))
                    elif initial_search_left: draw_windows_unified(annotated_warped, win_y_low, win_y_high, x_to_draw_empty_L, color=(120,0,120))
                    if left_consecutive_misses > MAX_CONSECUTIVE_EMPTY_WINDOWS: actively_tracking_left = False
            if actively_tracking_right: # Logica per linea destra (copiata dalla tua versione)
                current_search_x_R = iter_right_base
                win_xR_low, win_xR_high = max(0, current_search_x_R - WINDOW_SEARCH_MARGIN), min(frame_width-1, current_search_x_R + WINDOW_SEARCH_MARGIN)
                contours_r, _ = cv2.findContours(mask[win_y_low:win_y_high, win_xR_low:win_xR_high], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                found_R = False
                if contours_r:
                    largest_r = max(contours_r, key=cv2.contourArea)
                    if cv2.contourArea(largest_r) > MIN_CONTOUR_AREA_IN_WINDOW and cv2.moments(largest_r)["m00"] != 0:
                        iter_right_base = win_xR_low + int(cv2.moments(largest_r)["m10"] / cv2.moments(largest_r)["m00"])
                        all_right_x_pts.append(iter_right_base); all_right_y_pts.append(win_y_center)
                        right_consecutive_misses, found_R = 0, True
                        if initial_search_right: draw_windows_unified(annotated_warped, win_y_low, win_y_high, iter_right_base, color=(0,255,0))
                if not found_R:
                    right_consecutive_misses += 1
                    x_to_draw_empty_R = current_search_x_R
                    if right_consecutive_misses <= MAX_CONSECUTIVE_EMPTY_WINDOWS and len(all_right_x_pts) >= MIN_POINTS_FOR_SLOPE_ESTIMATION:
                        dx_R = all_right_x_pts[-1] - all_right_x_pts[-2] if len(all_right_x_pts) >= 2 else 0
                        x_to_draw_empty_R = np.clip(iter_right_base + dx_R, 0, frame_width-1)
                        iter_right_base = x_to_draw_empty_R
                        if initial_search_right: draw_windows_unified(annotated_warped, win_y_low, win_y_high, x_to_draw_empty_R, color=(0,180,0))
                    elif initial_search_right: draw_windows_unified(annotated_warped, win_y_low, win_y_high, x_to_draw_empty_R, color=(0,120,0))
                    if right_consecutive_misses > MAX_CONSECUTIVE_EMPTY_WINDOWS: actively_tracking_right = False
        
        # ... (LIVELLO 1C: SELEZIONE MONOLINEA - identica al tuo codice) ...
        line_selection_msg = ""
        found_L_by_sw = len(all_left_x_pts) >= MIN_POINTS_FOR_STABLE_LINE 
        found_R_by_sw = len(all_right_x_pts) >= MIN_POINTS_FOR_STABLE_LINE
        current_search_L, current_search_R = False, False; current_L_x_pts, current_L_y_pts, current_R_x_pts, current_R_y_pts = [],[],[],[]
        if found_L_by_sw and found_R_by_sw:
            line_selection_msg = "SEL: Both SW valid."
            if all_left_y_pts[0] >= all_right_y_pts[0]: line_selection_msg += " Keep L (lower)"; current_search_L,current_L_x_pts,current_L_y_pts = True,list(all_left_x_pts),list(all_left_y_pts)
            else: line_selection_msg += " Keep R (lower)"; current_search_R,current_R_x_pts,current_R_y_pts = True,list(all_right_x_pts),list(all_right_y_pts)
        elif found_L_by_sw: line_selection_msg = "SEL: Only L SW valid"; current_search_L,current_L_x_pts,current_L_y_pts = True,list(all_left_x_pts),list(all_left_y_pts)
        elif found_R_by_sw: line_selection_msg = "SEL: Only R SW valid"; current_search_R,current_R_x_pts,current_R_y_pts = True,list(all_right_x_pts),list(all_right_y_pts)
        else: line_selection_msg = "SEL: No SW lines valid"

        # ... (LIVELLO 2: CONTROLLO CURVATURA - identico al tuo codice) ...
        curve_override_msg = ""
        roi_y_start_abs, roi_y_end_abs = int(frame_height*CURVE_ROI_Y_START_PERCENT), int(frame_height*CURVE_ROI_Y_END_PERCENT)
        curve_roi_x_offset_abs = int(frame_width*CURVE_ROI_X_CENTER_OFFSET_PERCENT)
        roi_x_start_abs_L, roi_x_end_abs_L = max(0,frame_center_x-curve_roi_x_offset_abs-int(frame_width*0.15)), min(frame_width-1,frame_center_x+curve_roi_x_offset_abs-int(frame_width*0.05))
        roi_x_start_abs_R, roi_x_end_abs_R = max(0,frame_center_x-curve_roi_x_offset_abs+int(frame_width*0.05)), min(frame_width-1,frame_center_x+curve_roi_x_offset_abs+int(frame_width*0.15))
        left_x_curve_roi, right_x_curve_roi = [], []
        if current_search_L and current_L_x_pts: left_x_curve_roi = [x for i,x in enumerate(current_L_x_pts) if i<len(current_L_y_pts) and roi_y_start_abs<current_L_y_pts[i]<roi_y_end_abs and roi_x_start_abs_L<x<roi_x_end_abs_L]
        if current_search_R and current_R_x_pts: right_x_curve_roi = [x for i,x in enumerate(current_R_x_pts) if i<len(current_R_y_pts) and roi_y_start_abs<current_R_y_pts[i]<roi_y_end_abs and roi_x_start_abs_R<x<roi_x_end_abs_R]
        analysis_L_curve_roi = analyze_line_curvature(left_x_curve_roi) if current_search_L and left_x_curve_roi else "unknown"
        analysis_R_curve_roi = analyze_line_curvature(right_x_curve_roi) if current_search_R and right_x_curve_roi else "unknown"
        if current_search_R and not current_search_L and analysis_R_curve_roi == "left_curve_roi": 
            curve_override_msg = f"CURV_OVR R->L (ROI_R: {analysis_R_curve_roi.replace('_roi','').replace('curve','')})"
            current_search_L,current_L_x_pts,current_L_y_pts,current_search_R,current_R_x_pts,current_R_y_pts = True,list(current_R_x_pts),list(current_R_y_pts),False,[],[]
        elif current_search_L and not current_search_R and analysis_L_curve_roi == "right_curve_roi": 
            curve_override_msg = f"CURV_OVR L->R (ROI_L: {analysis_L_curve_roi.replace('_roi','').replace('curve','')})"
            current_search_R,current_R_x_pts,current_R_y_pts,current_search_L,current_L_x_pts,current_L_y_pts = True,list(current_L_x_pts),list(current_L_y_pts),False,[],[]

        final_search_L, final_search_R = current_search_L, current_search_R
        final_L_x_pts, final_R_x_pts = current_L_x_pts, current_R_x_pts
            
        # ... (CALCOLO CENTRO CORSIA - identico al tuo codice) ...
        new_center_calc, center_to_use_for_steering = False, valid_lane_center 
        if final_search_L and final_L_x_pts and not final_search_R: center_to_use_for_steering,new_center_calc = int(np.mean(final_L_x_pts))+estimated_lane_width//2,True
        elif final_search_R and final_R_x_pts and not final_search_L: center_to_use_for_steering,new_center_calc = int(np.mean(final_R_x_pts))-estimated_lane_width//2,True
        elif final_search_L and final_L_x_pts and final_search_R and final_R_x_pts: 
            avg_L,avg_R,width = int(np.mean(final_L_x_pts)),int(np.mean(final_R_x_pts)),int(np.mean(final_R_x_pts))-int(np.mean(final_L_x_pts))
            if MIN_PLAUSIBLE_LANE_WIDTH_ESTIMATION<width<MAX_PLAUSIBLE_LANE_WIDTH_ESTIMATION:
                center_to_use_for_steering=(avg_L+avg_R)//2
                estimated_lane_width=int(LANE_WIDTH_SMOOTH_ALPHA*width+(1-LANE_WIDTH_SMOOTH_ALPHA)*estimated_lane_width)
            else: center_to_use_for_steering = avg_L+estimated_lane_width//2 if abs(avg_L-(valid_lane_center-estimated_lane_width//2)) < abs(avg_R-(valid_lane_center+estimated_lane_width//2)) else avg_R-estimated_lane_width//2
            new_center_calc=True
        if new_center_calc: valid_lane_center,invalid_frame_count = center_to_use_for_steering,0
        else: invalid_frame_count +=1
        if invalid_frame_count > MAX_INVALID_FRAMES: pass 
        error = center_to_use_for_steering - frame_center_x 
        raw_angle = np.clip(-int(error/3.5),-40,40); angle = np.clip(smooth_angle(raw_angle,last_direction,alpha),-35,35); last_direction=angle

        # --- AGGIORNAMENTO MEMORIA last_valid_line_side ---
        line_L_is_stable = final_search_L and len(final_L_x_pts) >= MIN_POINTS_FOR_STABLE_LINE
        line_R_is_stable = final_search_R and len(final_R_x_pts) >= MIN_POINTS_FOR_STABLE_LINE
        if line_L_is_stable and not line_R_is_stable : last_valid_line_side = 'left'
        elif line_R_is_stable and not line_L_is_stable: last_valid_line_side = 'right'
        # Se nessuna è stabile o entrambe sono stabili (raro), last_valid_line_side non cambia

        # --- ANNOTAZIONI (come nel tuo codice, assicurati che i nomi delle variabili siano corretti) ---
        cv2.line(annotated_warped,(center_to_use_for_steering,0),(center_to_use_for_steering,frame_height),(0,255,255),2)
        cv2.line(annotated_warped,(frame_center_x,0),(frame_center_x,frame_height),(255,255,255),1)
        cv2.putText(annotated_warped, f"Angle: {angle}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        lpts_display = len(final_L_x_pts) if final_search_L and final_L_x_pts else 0
        rpts_display = len(final_R_x_pts) if final_search_R and final_R_x_pts else 0
        cv2.putText(annotated_warped, f"L_pts: {lpts_display} R_pts: {rpts_display}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),1)
        cv2.putText(annotated_warped, f"Est.W: {estimated_lane_width:.0f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),1)
        roi_draw_x_start = min(roi_x_start_abs_L, roi_x_start_abs_R); roi_draw_x_end = max(roi_x_end_abs_L, roi_x_end_abs_R)
        cv2.rectangle(annotated_warped, (roi_draw_x_start, roi_y_start_abs), (roi_draw_x_end, roi_y_end_abs), (255, 255, 0), 1)
        y_pos_msg = 120
        if hist_debug_msg: cv2.putText(annotated_warped, hist_debug_msg, (10, y_pos_msg), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1); y_pos_msg += 15
        if line_selection_msg: cv2.putText(annotated_warped, line_selection_msg, (10, y_pos_msg), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,255), 1); y_pos_msg += 20
        if curve_override_msg: cv2.putText(annotated_warped, curve_override_msg, (10, y_pos_msg), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,100,0), 1)
        cv2.putText(annotated_warped,f"VLC(prev): {current_vlc_for_histogram_check}",(10,160),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,180,0),1)
        cv2.putText(annotated_warped,f"VLC(curr): {valid_lane_center}",(10,180),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,255,0),1)
        cv2.putText(annotated_warped, f"LastValid: {last_valid_line_side}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1) 
        if final_search_L and analysis_L_curve_roi!="unknown": cv2.putText(annotated_warped,f"L_ROI:{analysis_L_curve_roi.replace('_roi','').replace('curve','')}",(frame_width-150,70),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,100,255),1)
        if final_search_R and analysis_R_curve_roi!="unknown": cv2.putText(annotated_warped,f"R_ROI:{analysis_R_curve_roi.replace('_roi','').replace('curve','')}",(frame_width-150,90),cv2.FONT_HERSHEY_SIMPLEX,0.4,(100,255,100),1)


        # --- Controllo PiCarX ---
        px.set_dir_servo_angle(angle)
        velocita_avanzamento = 15 # DA TARARE (0-100), o rendila dinamica
        px.forward(velocita_avanzamento)

        # --- Visualizzazione ---
        cv2.imshow("Annotated Warped", annotated_warped) 
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC per uscire
            print("Tasto ESC premuto, uscita.")
            break
        time.sleep(0.01) # Rallenta il loop se necessario e se non usi waitKey

except KeyboardInterrupt:
    print("Programma interrotto dall'utente (Ctrl+C).")
except Exception as e:
    print(f"Errore imprevisto: {e}")
    import traceback
    traceback.print_exc() # Stampa lo stack trace completo per errori imprevisti
finally:
    print("Fermo i motori e rilascio le risorse...")
    if 'px' in locals() or 'px' in globals():
        px.stop()
        px.set_dir_servo_angle(0) 
        print("Motori PiCarX fermati.")
    if 'picam2' in locals() or 'picam2' in globals():
        if picam2.started:
            picam2.stop()
            print("Fotocamera PiCamera2 fermata.")
    cv2.destroyAllWindows()
    print("Finestre OpenCV chiuse. Uscita completata.")