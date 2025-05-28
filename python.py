import cv2
import time
import numpy as np
import math

from picarx import Picarx
from picamera2 import Picamera2

# Initialize PicarX
px = Picarx()
px.forward(0)  # Start with zero speed

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Wait for camera to initialize
time.sleep(2)

# Steering control parameters
last_steering_angle = 0.0
last_lane_width = 250.0  # Larghezza iniziale stimata
smoothing_factor = 0.3   # Fattore di smoothing per correzioni graduali

# Parametri di controllo adattivo
Kp_normal = 2.0          # Gain normale per carreggiata standard
Kp_narrow = 5.0          # Gain aggressivo per carreggiata stretta
narrow_threshold = 150   # Soglia per carreggiata stretta
wide_threshold = 300     # Soglia per carreggiata larga

# Offset dal centro (50 pixel a destra del centro)
center_offset = 50       # Positivo = sposta a destra, Negativo = sposta a sinistra

base_speed = 1  # Set a positive value to make the robot move

def calculate_average_line_coords(image_shape, lines_segments):
    img_height = image_shape[0]
    default_coords = (0, 0, 0, 0)

    if not lines_segments:
        return default_coords

    x_coords = []
    y_coords = []
    for x1_seg, y1_seg, x2_seg, y2_seg in lines_segments:
        x_coords.extend([x1_seg, x2_seg])
        y_coords.extend([y1_seg, y2_seg])

    if not x_coords or not y_coords or len(np.unique(y_coords)) < 2:
        return default_coords

    try:
        poly_coeffs = np.polyfit(y_coords, x_coords, deg=1)
        y_top_draw = 0
        y_bottom_draw = img_height
        x_top_calc = int(poly_coeffs[0] * y_top_draw + poly_coeffs[1])
        x_bottom_calc = int(poly_coeffs[0] * y_bottom_draw + poly_coeffs[1])
        return (x_top_calc, y_top_draw, x_bottom_calc, y_bottom_draw)
    except (np.polynomial.polyutils.RankWarning, np.linalg.LinAlgError, TypeError):
        return default_coords

def calculate_angle(coords):
    if coords == (0, 0, 0, 0):
        return None
    x1, y1, x2, y2 = coords
    angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
    return angle

def smooth_steering(current_angle, target_angle, smoothing_factor):
    """Applica smoothing per correzioni graduali"""
    return current_angle + smoothing_factor * (target_angle - current_angle)

def adaptive_gain(lane_width):
    """Calcola il gain adattivo basato sulla larghezza della carreggiata"""
    if lane_width < narrow_threshold:
        # Carreggiata stretta: gain più aggressivo
        return Kp_narrow
    elif lane_width > wide_threshold:
        # Carreggiata larga: gain ridotto
        return Kp_normal * 0.7
    else:
        # Carreggiata normale: interpolazione lineare
        ratio = (lane_width - narrow_threshold) / (wide_threshold - narrow_threshold)
        return Kp_narrow - (Kp_narrow - Kp_normal) * ratio

while True:
    frame = picam2.capture_array()
    h, w = frame.shape[:2]
    roi_top, roi_bottom = int(h * 0.6), h

    src = np.float32([[0, roi_top], [w, roi_top], [w, roi_bottom], [0, roi_bottom]])
    dst_w, dst_h = 300, 200
    dst = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    bird_eye = cv2.warpPerspective(frame, matrix, (dst_w, dst_h))

    gaussian = cv2.GaussianBlur(bird_eye, (5, 5), 0)
    hls = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HLS)

    yellow_mask = cv2.inRange(hls, np.array([15, 40, 100]), np.array([35, 255, 255]))
    white_mask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    edges = cv2.Canny(mask, 50, 150)

    detected_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=15, maxLineGap=20)
    left_segments, right_segments = [], []

    if detected_lines is not None:
        for seg in detected_lines:
            x1, y1, x2, y2 = seg[0]
            mid_x = (x1 + x2) / 2
            center_x = bird_eye.shape[1] / 2
            min_distance_from_center = 20

            if abs(mid_x - center_x) < min_distance_from_center:
                continue  # Ignore segment and keep last angle

            if x2 - x1 == 0:
                if mid_x < bird_eye.shape[1] / 2:
                    left_segments.append((x1, y1, x2, y2))
                else:
                    right_segments.append((x1, y1, x2, y2))
                continue

            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.6:
                continue

            if slope < 0 and mid_x < bird_eye.shape[1] / 2:
                left_segments.append((x1, y1, x2, y2))
            elif slope > 0 and mid_x >= bird_eye.shape[1] / 2:
                right_segments.append((x1, y1, x2, y2))

    H_be, W_be = bird_eye.shape[:2]
    left_coords = calculate_average_line_coords(bird_eye.shape, left_segments)
    right_coords = calculate_average_line_coords(bird_eye.shape, right_segments)

    L_pt1, L_pt2 = (left_coords[0], left_coords[1]), (left_coords[2], left_coords[3])
    R_pt1, R_pt2 = (right_coords[0], right_coords[1]), (right_coords[2], right_coords[3])
    left_angle = calculate_angle(left_coords)
    right_angle = calculate_angle(right_coords)

    # Calcolo migliorato della larghezza della carreggiata
    bottom_left_corner_x = 0
    bottom_right_corner_x = W_be

    L_bot_x_detected_val = left_coords[2] if left_coords != (0,0,0,0) else None
    R_bot_x_detected_val = right_coords[2] if right_coords != (0,0,0,0) else None

    # Gestione più intelligente della larghezza
    if L_bot_x_detected_val is not None and R_bot_x_detected_val is not None:
        # Entrambe le linee rilevate - calcolo diretto
        current_lane_width = R_bot_x_detected_val - L_bot_x_detected_val
        L_bot_x = L_bot_x_detected_val
        R_bot_x = R_bot_x_detected_val
    elif L_bot_x_detected_val is not None:
        # Solo linea sinistra - stima la destra usando la larghezza precedente
        current_lane_width = last_lane_width
        L_bot_x = L_bot_x_detected_val
        R_bot_x = L_bot_x_detected_val + current_lane_width
    elif R_bot_x_detected_val is not None:
        # Solo linea destra - stima la sinistra usando la larghezza precedente
        current_lane_width = last_lane_width
        R_bot_x = R_bot_x_detected_val
        L_bot_x = R_bot_x_detected_val - current_lane_width
    else:
        # Nessuna linea rilevata - usa valori precedenti
        current_lane_width = last_lane_width
        image_center = W_be / 2
        L_bot_x = image_center - current_lane_width / 2
        R_bot_x = image_center + current_lane_width / 2

    # Smooth della larghezza per evitare cambiamenti bruschi
    smoothed_lane_width = last_lane_width + 0.2 * (current_lane_width - last_lane_width)
    
    # Calcolo del centro di corsia con offset
    lane_center = (L_bot_x + R_bot_x) / 2
    image_center = W_be / 2
    target_center = image_center + center_offset  # Applica offset dal centro
    
    deviation = target_center - lane_center
    
    # Gain adattivo basato sulla larghezza della carreggiata
    current_Kp = adaptive_gain(smoothed_lane_width)
    
    intended_steering_angle = current_Kp * deviation
    
    # Applica smoothing per correzioni graduali
    raw_steering_angle = max(-45.0, min(45.0, intended_steering_angle))
    steering_angle = smooth_steering(last_steering_angle, raw_steering_angle, smoothing_factor)
    
    # Validazione della larghezza con tolleranza adattiva
    width_valid = True
    if L_bot_x_detected_val is not None and R_bot_x_detected_val is not None:
        if smoothed_lane_width < 100 or smoothed_lane_width > 400:
            width_valid = False
            
    if not width_valid:
        steering_angle = last_steering_angle
        print(f"Larghezza {smoothed_lane_width:.1f} non valida, mantengo angolo {last_steering_angle:.1f}")
    elif L_bot_x_detected_val is None and R_bot_x_detected_val is None:
        # Nessuna linea rilevata - correzione graduale verso centro
        steering_angle = smooth_steering(last_steering_angle, 0, 0.1)
        print("Nessuna linea rilevata, correzione graduale verso centro")
    else:
        detection_status = ""
        if L_bot_x_detected_val is None:
            detection_status = "Solo linea destra"
        elif R_bot_x_detected_val is None:
            detection_status = "Solo linea sinistra"
        else:
            detection_status = "Entrambe le linee"
        
        # Informazioni sulla modalità di controllo
        control_mode = ""
        if smoothed_lane_width < narrow_threshold:
            control_mode = " [AGGRESSIVO]"
        elif smoothed_lane_width > wide_threshold:
            control_mode = " [RIDOTTO]"
        else:
            control_mode = " [NORMALE]"
            
        print(f"{detection_status}. Larghezza: {smoothed_lane_width:.1f}, Kp: {current_Kp:.1f}, Angolo: {steering_angle:.1f}{control_mode}")

    # Aggiorna i valori per il prossimo ciclo
    last_lane_width = smoothed_lane_width
    last_steering_angle = steering_angle

    # Controllo del robot
    px.forward(base_speed)
    px.set_dir_servo_angle(steering_angle)

    # --- Visualization ---
    poly_L_top_x = left_coords[0] if left_coords != (0,0,0,0) else int(L_bot_x)
    poly_L_bot_x = left_coords[2] if left_coords != (0,0,0,0) else int(L_bot_x)
    poly_R_top_x = right_coords[0] if right_coords != (0,0,0,0) else int(R_bot_x)
    poly_R_bot_x = right_coords[2] if right_coords != (0,0,0,0) else int(R_bot_x)

    pg_L_top = (poly_L_top_x, 0)
    pg_L_bot = (poly_L_bot_x, H_be)
    pg_R_top = (poly_R_top_x, 0)
    pg_R_bot = (poly_R_bot_x, H_be)

    verts = np.array([pg_L_top, pg_R_top, pg_R_bot, pg_L_bot], dtype=np.int32)
    lanes_view = bird_eye.copy()
    
    if verts.size > 0:
        overlay = np.zeros_like(lanes_view)
        cv2.fillPoly(overlay, [verts], (0,255,0))
        cv2.addWeighted(overlay, 0.3, lanes_view, 0.7, 0, lanes_view)

    if left_coords != (0,0,0,0):
        cv2.line(lanes_view, L_pt1, L_pt2, (255,0,0), 3)
    if right_coords != (0,0,0,0):
        cv2.line(lanes_view, R_pt1, R_pt2, (0,0,255), 3)

    # Visualizza il centro target con offset
    target_center_int = int(target_center)
    cv2.line(lanes_view, (target_center_int, 0), (target_center_int, H_be), (255,255,0), 2)
    
    # Visualizza il centro attuale della corsia
    lane_center_int = int(lane_center)
    cv2.line(lanes_view, (lane_center_int, 0), (lane_center_int, H_be), (255,0,255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    thickness = 1

    # Testi informativi aggiornati
    left_text = f"Angolo Sinistro: {left_angle:.1f} deg" if left_angle is not None else "Angolo Sinistro: N/A"
    cv2.putText(lanes_view, left_text, (10, 20), font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    right_text = f"Angolo Destro: {right_angle:.1f} deg" if right_angle is not None else "Angolo Destro: N/A"
    cv2.putText(lanes_view, right_text, (10, 40), font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    width_text = f"Larghezza: {smoothed_lane_width:.1f}"
    cv2.putText(lanes_view, width_text, (10, 60), font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    steer_text = f"Sterzata: {steering_angle:.1f} deg"
    cv2.putText(lanes_view, steer_text, (10, 80), font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    kp_text = f"Kp: {current_Kp:.1f}"
    cv2.putText(lanes_view, kp_text, (10, 100), font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    offset_text = f"Offset: {center_offset} px"
    cv2.putText(lanes_view, offset_text, (10, 120), font, font_scale, font_color, thickness, cv2.LINE_AA)

    cv2.imshow("Corsie Rilevate", lanes_view)
    cv2.imshow("Maschera", mask)
    cv2.rectangle(frame, (0, roi_top), (w, roi_bottom), (0,255,0), 2)
    cv2.imshow("Video Originale con ROI", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Cleanup
picam2.stop()
picam2.close()
px.forward(0)
px.set_dir_servo_angle(0)
time.sleep(0.5)
cv2.destroyAllWindows()