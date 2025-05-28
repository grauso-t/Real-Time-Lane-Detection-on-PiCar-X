import cv2
import time
import numpy as np
import math
from picamera2 import Picamera2
from picarx import Picarx

px = Picarx()
px.forward(0)  # Inizia con velocità zero
px.set_dir_servo_angle(0)  # Fixed: added parentheses for method call

# Inizializza la camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")

# Avvia la camera
picam2.start()
time.sleep(1)  # tempo per inizializzare la camera

px.forward(1)  # Imposta velocità di avanzamento

# Variabili globali per il controllo dell'angolo
previous_angle = 0.0
smooth_factor = 0.3  # Fattore di smoothing (0.1 = molto smooth, 0.9 = poco smooth)

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

def calculate_lane_width(left_coords, right_coords, image_width):
    """
    Calcola la larghezza della carreggiata e verifica se le linee sono troppo vicine al centro
    """
    # Se una delle due linee non è presente, non possiamo calcolare la larghezza
    if left_coords == (0,0,0,0) or right_coords == (0,0,0,0):
        return None, False, False
    
    # Calcola la posizione media delle linee sinistra e destra
    left_center_x = (left_coords[0] + left_coords[2]) / 2
    right_center_x = (right_coords[0] + right_coords[2]) / 2
    
    # Calcola la larghezza della carreggiata
    lane_width = abs(right_center_x - left_center_x)
    
    # Controlla se la larghezza è troppo piccola (< 180px)
    width_too_small = lane_width < 180
    
    # Controlla se le linee sono troppo vicine al centro dell'immagine
    center_x = image_width / 2
    left_too_close = abs(left_center_x - center_x) < 30
    right_too_close = abs(right_center_x - center_x) < 30
    lines_too_close_to_center = left_too_close or right_too_close
    
    return lane_width, width_too_small, lines_too_close_to_center

def calculate_angle(right_segments, left_segments):
    def get_average_vector(segments):
        if not segments:
            return None
        dx_total, dy_total = 0, 0
        for x1, y1, x2, y2 in segments:
            dx_total += (x2 - x1)
            dy_total += (y2 - y1)
        return (dx_total, dy_total)
    
    left_vec = get_average_vector(left_segments)
    right_vec = get_average_vector(right_segments)
    
    if left_vec and right_vec:
        angle = 0.0
    else:
        vec = left_vec if left_vec else right_vec
        if not vec:
            angle = 0.0
        else:
            dx, dy = vec
            if dx == 0:
                angle = 0
            else:
                angle_rad = math.atan2(dy, dx)
                angle = math.degrees(angle_rad)
    
    # Normalizzazione tra -45 e 45
    min_real, max_real = -80, 80
    new_min, new_max = -45, 45
    
    # Clamp dell'angolo ai limiti noti per evitare extrapolazione
    angle = max(min(angle, max_real), min_real)
    normalized_angle = new_min + ((angle - min_real) * (new_max - new_min) / (max_real - min_real))
    
    # OPZIONE 3: Mappatura personalizzata con punti di controllo
    def custom_mapping(value):
        # Definisci coppie (input, output) per la mappatura
        mapping_points = [
            (-45, -45),  # Estremo negativo resta uguale
            (-25, -35),  # Valori negativi medi amplificati
            (0, 0),      # Zero resta zero
            (25, 35),    # Valori positivi medi amplificati
            (45, 45)     # Estremo positivo resta uguale
        ]
        
        # Interpolazione lineare tra i punti
        for i in range(len(mapping_points) - 1):
            x1, y1 = mapping_points[i]
            x2, y2 = mapping_points[i + 1]
            
            if x1 <= value <= x2:
                # Interpolazione lineare
                t = (value - x1) / (x2 - x1) if x2 != x1 else 0
                return y1 + t * (y2 - y1)
        
        return value  # Fallback
    
    amplified_angle = custom_mapping(normalized_angle)
    return amplified_angle

def apply_smoothing(new_angle, previous_angle, smooth_factor):
    """
    Applica lo smoothing all'angolo per evitare sterzate brusche
    """
    return previous_angle * (1 - smooth_factor) + new_angle * smooth_factor

try:
    # Main loop - removed video file handling, using only camera
    while True:
        frame = picam2.capture_array()
        if frame is None:
            break

        h, w = frame.shape[:2]
        roi_top, roi_bottom = int(h * 0.6), h

        # Perspective transform to bird-eye view
        src = np.float32([[0, roi_top], [w, roi_top], [w, roi_bottom], [0, roi_bottom]])
        dst_w, dst_h = 300, 200
        dst = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        bird_eye = cv2.warpPerspective(frame, matrix, (dst_w, dst_h))

        # Pre-processing
        gaussian = cv2.GaussianBlur(bird_eye, (5, 5), 0)
        hls = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HLS)
        yellow_mask = cv2.inRange(hls, np.array([15, 40, 100]), np.array([35, 255, 255]))
        white_mask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(yellow_mask, white_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Edge detection and Hough transform
        edges = cv2.Canny(mask, 50, 150)
        detected_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=15, maxLineGap=20)

        # Separate segments by slope and side
        left_segments, right_segments = [], []
        if detected_lines is not None:
            for seg in detected_lines:
                x1, y1, x2, y2 = seg[0]
                if x2 - x1 == 0:
                    if x1 < bird_eye.shape[1] / 2:
                        left_segments.append((x1, y1, x2, y2))
                    else:
                        right_segments.append((x1, y1, x2, y2))
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.4:
                    continue
                if slope < 0:
                    left_segments.append((x1, y1, x2, y2))
                else:
                    right_segments.append((x1, y1, x2, y2))

        # Average line coords
        H_be, W_be = bird_eye.shape[:2]
        left_coords = calculate_average_line_coords(bird_eye.shape, left_segments)
        right_coords = calculate_average_line_coords(bird_eye.shape, right_segments)
        
        # Calcola la larghezza della carreggiata e verifica le condizioni
        lane_width, width_too_small, lines_too_close = calculate_lane_width(left_coords, right_coords, W_be)
        
        # Calcola il nuovo angolo
        new_angle = calculate_angle(right_segments, left_segments)
        
        # Determina se usare il nuovo angolo o mantenere quello precedente
        if lane_width is not None:
            if width_too_small or lines_too_close:
                # Mantieni l'angolo precedente se la carreggiata è troppo stretta o le linee troppo vicine al centro
                final_angle = previous_angle
                status = f"MANTAIN PREV ANGLE - Width: {lane_width:.1f}px, TooSmall: {width_too_small}, TooClose: {lines_too_close}"
            else:
                # Applica smoothing al nuovo angolo
                final_angle = apply_smoothing(new_angle, previous_angle, smooth_factor)
                status = f"NEW ANGLE (smoothed) - Width: {lane_width:.1f}px"
        else:
            # Se non possiamo calcolare la larghezza, usa il nuovo angolo con smoothing
            final_angle = apply_smoothing(new_angle, previous_angle, smooth_factor)
            status = "NEW ANGLE (no width calc)"
        
        # Aggiorna l'angolo precedente per il prossimo frame
        previous_angle = final_angle
        
        print(f"Angle: {final_angle:.2f}° | Raw: {new_angle:.2f}° | {status}")

        L_pt1, L_pt2 = (left_coords[0], left_coords[1]), (left_coords[2], left_coords[3])
        R_pt1, R_pt2 = (right_coords[0], right_coords[1]), (right_coords[2], right_coords[3])

        # Polygon base x-coords
        poly_L_top_x = left_coords[0] if left_coords != (0,0,0,0) else 0
        poly_L_bot_x = left_coords[2] if left_coords != (0,0,0,0) else 0
        poly_R_top_x = right_coords[0] if right_coords != (0,0,0,0) else W_be
        poly_R_bot_x = right_coords[2] if right_coords != (0,0,0,0) else W_be

        pg_L_top = (poly_L_top_x, 0)
        pg_L_bot = (poly_L_bot_x, H_be)
        pg_R_top = (poly_R_top_x, 0)
        pg_R_bot = (poly_R_bot_x, H_be)

        # Build vertices: triangle if one missing, quad if both, none if none
        if left_coords == (0,0,0,0) and right_coords != (0,0,0,0):
            pg_L_bot = (0, H_be)
            verts = np.array([pg_R_top, pg_R_bot, pg_L_bot], dtype=np.int32)
        elif right_coords == (0,0,0,0) and left_coords != (0,0,0,0):
            pg_R_bot = (W_be, H_be)
            verts = np.array([pg_L_top, pg_L_bot, pg_R_bot], dtype=np.int32)
        elif left_coords != (0,0,0,0) and right_coords != (0,0,0,0):
            verts = np.array([pg_L_top, pg_R_top, pg_R_bot, pg_L_bot], dtype=np.int32)
        else:
            verts = np.array([], dtype=np.int32)

        lanes_view = bird_eye.copy()
        if verts.size > 0:
            overlay = np.zeros_like(lanes_view)
            cv2.fillPoly(overlay, [verts], (0,255,0))
            cv2.addWeighted(overlay, 0.5, lanes_view, 0.5, 0, lanes_view)

        # Draw lines and angle
        if left_coords != (0,0,0,0): 
            cv2.line(lanes_view, L_pt1, L_pt2, (255,0,0), 2)
        if right_coords != (0,0,0,0): 
            cv2.line(lanes_view, R_pt1, R_pt2, (0,0,255), 2)

        # Aggiungi testo con informazioni di debug
        if lane_width is not None:
            cv2.putText(lanes_view, f"Width: {lane_width:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(lanes_view, f"Angle: {final_angle:.1f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Fixed: correct method call for setting servo angle
        px.set_dir_servo_angle(final_angle)

        # Display
        cv2.imshow("Detected Lanes", lanes_view)
        cv2.imshow("Mask", mask)
        cv2.rectangle(frame, (0, roi_top), (w, roi_bottom), (0,255,0), 2)
        cv2.imshow("Video", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        time.sleep(0.04)

except KeyboardInterrupt:
    print("Program interrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Cleanup
    print("Cleaning up...")
    px.forward(0)  # Stop the car
    px.set_dir_servo_angle(0)  # Center the steering
    cv2.destroyAllWindows()
    picam2.stop()
    print("Cleanup complete")