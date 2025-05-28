import cv2
import time
import numpy as np
import math
from picamera2 import Picamera2
from picarx import Picarx

px = Picarx()
px.forward(0)
px.set_dir_servo_angle(0)

# Inizializza la camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")

# Avvia la camera
picam2.start()
time.sleep(1)

px.forward(1)

# Variabili globali per il controllo dell'angolo
previous_angle = 0.0
smooth_factor = 0.3
# NUOVA SOGLIA: Larghezza minima della carreggiata per considerare affidabile il calcolo dell'angolo.
# Regola questo valore in base ai tuoi test. Deve essere > 180 (valore di width_too_small).
MIN_WIDTH_THRESHOLD = 220 # Esempio: 220 pixel

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
    if left_coords == (0,0,0,0) or right_coords == (0,0,0,0):
        return None, False, False

    left_center_x = (left_coords[0] + left_coords[2]) / 2
    right_center_x = (right_coords[0] + right_coords[2]) / 2
    lane_width = abs(right_center_x - left_center_x)
    width_too_small = lane_width < 180

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
        angle = 0.0 # Se vede entrambe le linee, idealmente dovrebbe andare dritto (0) o mediare? Per ora 0.
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

    min_real, max_real = -80, 80
    new_min, new_max = -45, 45
    angle = max(min(angle, max_real), min_real)
    normalized_angle = new_min + ((angle - min_real) * (new_max - new_min) / (max_real - min_real))

    def custom_mapping(value):
        mapping_points = [
            (-45, -45), (-25, -35), (0, 0), (25, 35), (45, 45)
        ]
        for i in range(len(mapping_points) - 1):
            x1, y1 = mapping_points[i]
            x2, y2 = mapping_points[i + 1]
            if x1 <= value <= x2:
                t = (value - x1) / (x2 - x1) if x2 != x1 else 0
                return y1 + t * (y2 - y1)
        return value

    amplified_angle = custom_mapping(normalized_angle)
    return amplified_angle

def apply_smoothing(new_angle, previous_angle, smooth_factor):
    return previous_angle * (1 - smooth_factor) + new_angle * smooth_factor

try:
    while True:
        frame = picam2.capture_array()
        if frame is None:
            break

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
                if x2 - x1 == 0:
                    if x1 < bird_eye.shape[1] / 2: left_segments.append((x1, y1, x2, y2))
                    else: right_segments.append((x1, y1, x2, y2))
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.4: continue
                if slope < 0: left_segments.append((x1, y1, x2, y2))
                else: right_segments.append((x1, y1, x2, y2))

        H_be, W_be = bird_eye.shape[:2]
        left_coords = calculate_average_line_coords(bird_eye.shape, left_segments)
        right_coords = calculate_average_line_coords(bird_eye.shape, right_segments)

        lane_width, width_too_small, lines_too_close = calculate_lane_width(left_coords, right_coords, W_be)
        new_angle = calculate_angle(right_segments, left_segments)

        # --- MODIFICA LOGICA ANGOLO ---
        if lane_width is not None:
            # Verifica se la larghezza è troppo piccola, le linee troppo vicine
            # O se la larghezza è SOTTO la nostra nuova soglia minima.
            is_below_min_threshold = lane_width < MIN_WIDTH_THRESHOLD
            if width_too_small or lines_too_close or is_below_min_threshold:
                # Se una di queste condizioni è vera, mantieni l'angolo precedente.
                final_angle = previous_angle
                status = f"MANTAIN PREV - W: {lane_width:.1f} (Small:{width_too_small}, Close:{lines_too_close}, BelowMin:{is_below_min_threshold})"
            else:
                # Altrimenti, se la larghezza è OK, usa il nuovo angolo (con smoothing).
                final_angle = apply_smoothing(new_angle, previous_angle, smooth_factor)
                status = f"NEW ANGLE (smoothed) - W: {lane_width:.1f}"
        else:
            # Se non possiamo calcolare la larghezza, mantieni l'angolo precedente per sicurezza.
            final_angle = previous_angle
            status = "MANTAIN PREV (no width calc)"
        # --- FINE MODIFICA ---

        previous_angle = final_angle
        print(f"Angle: {final_angle:.2f}° | Raw: {new_angle:.2f}° | {status}")

        L_pt1, L_pt2 = (left_coords[0], left_coords[1]), (left_coords[2], left_coords[3])
        R_pt1, R_pt2 = (right_coords[0], right_coords[1]), (right_coords[2], right_coords[3])

        poly_L_top_x = left_coords[0] if left_coords != (0,0,0,0) else 0
        poly_L_bot_x = left_coords[2] if left_coords != (0,0,0,0) else 0
        poly_R_top_x = right_coords[0] if right_coords != (0,0,0,0) else W_be
        poly_R_bot_x = right_coords[2] if right_coords != (0,0,0,0) else W_be

        pg_L_top = (poly_L_top_x, 0)
        pg_L_bot = (poly_L_bot_x, H_be)
        pg_R_top = (poly_R_top_x, 0)
        pg_R_bot = (poly_R_bot_x, H_be)

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

        if left_coords != (0,0,0,0): cv2.line(lanes_view, L_pt1, L_pt2, (255,0,0), 2)
        if right_coords != (0,0,0,0): cv2.line(lanes_view, R_pt1, R_pt2, (0,0,255), 2)

        if lane_width is not None:
            cv2.putText(lanes_view, f"Width: {lane_width:.1f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(lanes_view, f"Angle: {final_angle:.1f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        px.set_dir_servo_angle(final_angle)

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
    print("Cleaning up...")
    px.forward(0)
    px.set_dir_servo_angle(0)
    cv2.destroyAllWindows()
    picam2.stop()
    print("Cleanup complete")