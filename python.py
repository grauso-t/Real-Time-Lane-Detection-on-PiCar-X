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
Kp = 0.5  # Proportional gain for steering (adjust as needed)
base_speed = 0 # Base forward speed (adjust as needed) - SET THIS TO A POSITIVE VALUE TO MAKE THE ROBOT MOVE!

# Define a threshold for how close a single line can be to the center
# before we ignore its steering suggestion.
# This value will need tuning based on your bird's-eye view.
SINGLE_LINE_CENTER_THRESHOLD = 50 # pixels from image center

def calculate_average_line_coords(image_shape, lines_segments):
    """
    Calcola le coordinate medie di una linea a partire dai segmenti rilevati.

    Args:
        image_shape: Le dimensioni dell'immagine (altezza, larghezza).
        lines_segments: Una lista di segmenti di linea [(x1, y1, x2, y2), ...].

    Returns:
        Una tupla (x1, y1, x2, y2) con le coordinate della linea media,
        o (0, 0, 0, 0) se non è possibile calcolarla.
    """
    img_height = image_shape[0]
    default_coords = (0, 0, 0, 0)

    if not lines_segments:
        return default_coords

    x_coords = []
    y_coords = []
    for x1_seg, y1_seg, x2_seg, y2_seg in lines_segments:
        x_coords.extend([x1_seg, x2_seg])
        y_coords.extend([y1_seg, y2_seg])

    # Controlla se ci sono abbastanza punti unici per polyfit
    if not x_coords or not y_coords or len(np.unique(y_coords)) < 2:
        return default_coords

    try:
        # Esegue un fit polinomiale di primo grado (retta) scambiando x e y
        # per gestire meglio le linee quasi verticali.
        poly_coeffs = np.polyfit(y_coords, x_coords, deg=1)
        y_top_draw = 0  # Y superiore dell'immagine
        y_bottom_draw = img_height # Y inferiore dell'immagine
        # Calcola le coordinate X corrispondenti
        x_top_calc = int(poly_coeffs[0] * y_top_draw + poly_coeffs[1])
        x_bottom_calc = int(poly_coeffs[0] * y_bottom_draw + poly_coeffs[1])
        return (x_top_calc, y_top_draw, x_bottom_calc, y_bottom_draw)
    except (np.polynomial.polyutils.RankWarning, np.linalg.LinAlgError, TypeError):
        # Gestisce errori durante il polyfit
        return default_coords

def calculate_angle(coords):
    """
    Calcola l'angolo di una linea rispetto all'asse verticale.

    Args:
        coords: Una tupla (x1, y1, x2, y2) con le coordinate della linea.

    Returns:
        L'angolo in gradi, o None se le coordinate sono (0, 0, 0, 0).
    """
    if coords == (0, 0, 0, 0):
        return None
    x1, y1, x2, y2 = coords
    # Calcola l'angolo usando atan2 e converte in gradi.
    # Usiamo (x2 - x1) e (y2 - y1) per ottenere l'angolo
    # rispetto all'asse verticale (positivo verso destra).
    angle = math.degrees(math.atan2(x2 - x1, y2 - y1))

    # Normalizzazione dell'angolo tra -45 e 45 gradi
    min_real, max_real = -50, 50
    new_min, new_max = -45, 45

    # Clamp per evitare extrapolazioni
    normalized_angle = max(new_min, min(new_max, new_min + ((angle - min_real) * (new_max - new_min) / (max_real - min_real))))

    return normalized_angle

while True:
    frame = picam2.capture_array()
    h, w = frame.shape[:2]
    roi_top, roi_bottom = int(h * 0.6), h

    # Perspective Transform (Bird's Eye View)
    src = np.float32([[0, roi_top], [w, roi_top], [w, roi_bottom], [0, roi_bottom]])
    dst_w, dst_h = 300, 200 # Standard size for bird's-eye view
    dst = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    bird_eye = cv2.warpPerspective(frame, matrix, (dst_w, dst_h))

    # Image Preprocessing for Lane Detection
    gaussian = cv2.GaussianBlur(bird_eye, (5, 5), 0)
    hls = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HLS)

    # Define color ranges for yellow and white lanes
    yellow_mask = cv2.inRange(hls, np.array([15, 40, 100]), np.array([35, 255, 255]))
    white_mask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(yellow_mask, white_mask)

    # Morphological operations to clean up the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Canny Edge Detection
    edges = cv2.Canny(mask, 50, 150)

    # Hough Line Transform to detect line segments
    detected_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=15, maxLineGap=20)

    # Classify detected lines into left and right segments
    left_segments, right_segments = [], []
    if detected_lines is not None:
        for seg in detected_lines:
            x1, y1, x2, y2 = seg[0]
            # Calculate the midpoint of the segment for positional classification
            mid_x = (x1 + x2) / 2

            # Handle perfectly vertical lines to avoid division by zero for slope
            if x2 - x1 == 0:
                if mid_x < bird_eye.shape[1] / 2: # Vertical line in the left half
                    left_segments.append((x1, y1, x2, y2))
                else: # Vertical line in the right half
                    right_segments.append((x1, y1, x2, y2))
                continue # Skip to the next segment

            slope = (y2 - y1) / (x2 - x1)

            # Filter out near-horizontal lines as they are typically not lane lines
            if abs(slope) < 0.4:
                continue

            # Classify based on slope AND position
            # A negative slope (/) usually indicates a left lane line
            # A positive slope (\) usually indicates a right lane line
            # Ensure the line's midpoint is in its respective half of the image
            if slope < 0 and mid_x < bird_eye.shape[1] / 2: # Left line: negative slope and in left half
                left_segments.append((x1, y1, x2, y2))
            elif slope > 0 and mid_x >= bird_eye.shape[1] / 2: # Right line: positive slope and in right half
                right_segments.append((x1, y1, x2, y2))

    H_be, W_be = bird_eye.shape[:2]
    left_coords = calculate_average_line_coords(bird_eye.shape, left_segments)
    right_coords = calculate_average_line_coords(bird_eye.shape, right_segments)

    # L_pt1, L_pt2, R_pt1, R_pt2 are for visualization later, derived from final coords
    L_pt1, L_pt2 = (left_coords[0], left_coords[1]), (left_coords[2], left_coords[3])
    R_pt1, R_pt2 = (right_coords[0], right_coords[1]), (right_coords[2], right_coords[3])

    left_angle = calculate_angle(left_coords)
    right_angle = calculate_angle(right_coords)

    # --- Steering Control Logic ---
    intended_steering_angle = 0.0
    lane_width = 0 # Will be calculated based on detected/inferred lines

    # Define the bottom-left and bottom-right corners of the bird's-eye view
    # for inferring missing lines
    bottom_left_corner_x = 0
    bottom_right_corner_x = W_be

    # Get the bottom X-coordinates of the detected lines
    # Store whether they were actually detected or are (0,0,0,0) (meaning not detected)
    L_bot_x_detected_val = left_coords[2] if left_coords != (0,0,0,0) else None
    R_bot_x_detected_val = right_coords[2] if right_coords != (0,0,0,0) else None

    # Determine the actual X-coordinates for the lane calculation
    # If a line is not detected, assume its bottom point is at the corresponding image corner
    L_bot_x = L_bot_x_detected_val if L_bot_x_detected_val is not None else bottom_left_corner_x
    R_bot_x = R_bot_x_detected_val if R_bot_x_detected_val is not None else bottom_right_corner_x

    # Calculate lane_width and lane_center based on the determined L_bot_x and R_bot_x
    lane_width = R_bot_x - L_bot_x
    lane_center = (L_bot_x + R_bot_x) / 2
    image_center = W_be / 2
    deviation = image_center - lane_center

    # Calculate intended steering angle based on deviation
    intended_steering_angle = -Kp * deviation # Steer proportionally to deviation

    # Check if both lines were truly detected
    both_lines_truly_detected = (L_bot_x_detected_val is not None) and (R_bot_x_detected_val is not None)

    # Limit the steering angle to prevent extreme turns (pre-emptively, will be overridden by logic below)
    steering_angle = max(-45.0, min(45.0, intended_steering_angle))

    # --- Lane Width Control and No Line Detected / Single Line Too Close to Center Fallback ---
    if both_lines_truly_detected:
        if lane_width < 180: # Adjust 180 as needed for your track
            steering_angle = last_steering_angle
            print(f"Larghezza {lane_width:.1f} < 180 (entrambe le linee reali), mantengo angolo {last_steering_angle:.1f}")
        else:
            # Both lines detected and width is acceptable, use calculated angle
            print(f"Entrambe le linee rilevate. Larghezza: {lane_width:.1f}, Angolo calcolato: {steering_angle:.1f}")
    elif L_bot_x_detected_val is None and R_bot_x_detected_val is None:
        # If no lines were detected at all (both are inferred from corners)
        steering_angle = last_steering_angle # Or set to 0 to go straight/stop
        print("Nessuna linea rilevata, mantenendo l'ultimo angolo.")
    elif L_bot_x_detected_val is not None: # Only left line is detected
        # Check if the detected left line is too close to the center
        if abs(L_bot_x_detected_val - image_center) < SINGLE_LINE_CENTER_THRESHOLD:
            steering_angle = last_steering_angle
            print(f"Solo linea sinistra rilevata ({L_bot_x_detected_val:.1f}), ma troppo vicina al centro. Mantengo angolo {last_steering_angle:.1f}")
        else:
            # Left line is detected and not too close to center, use calculated angle
            print(f"Solo linea sinistra rilevata. Larghezza stimata: {lane_width:.1f}, Angolo calcolato: {steering_angle:.1f}")
    elif R_bot_x_detected_val is not None: # Only right line is detected
        # Check if the detected right line is too close to the center
        if abs(R_bot_x_detected_val - image_center) < SINGLE_LINE_CENTER_THRESHOLD:
            steering_angle = last_steering_angle
            print(f"Solo linea destra rilevata ({R_bot_x_detected_val:.1f}), ma troppo vicina al centro. Mantengo angolo {last_steering_angle:.1f}")
        else:
            # Right line is detected and not too close to center, use calculated angle
            print(f"Solo linea destra rilevata. Larghezza stimata: {lane_width:.1f}, Angolo calcolato: {steering_angle:.1f}")


    # Apply the steering angle and speed to the PicarX
    px.set_dir_servo_angle(steering_angle)
    px.forward(base_speed)

    # Update the last steering angle for the next iteration
    last_steering_angle = steering_angle
    # --- End Steering Control ---

    # --- Visualization ---
    # Prepare coordinates for drawing the lane polygon.
    # Note: These poly_X_top/bot_x variables are for visualization *only*.
    # The actual steering logic uses L_bot_x and R_bot_x which are inferred/clamped.

    # If left_coords is (0,0,0,0), assume its bottom is at 0 and top is at 0 for drawing.
    poly_L_top_x = left_coords[0] if left_coords != (0,0,0,0) else 0
    poly_L_bot_x = left_coords[2] if left_coords != (0,0,0,0) else 0

    # If right_coords is (0,0,0,0), assume its bottom is at W_be and top is at W_be for drawing.
    poly_R_top_x = right_coords[0] if right_coords != (0,0,0,0) else W_be
    poly_R_bot_x = right_coords[2] if right_coords != (0,0,0,0) else W_be

    pg_L_top = (poly_L_top_x, 0)
    pg_L_bot = (poly_L_bot_x, H_be)
    pg_R_top = (poly_R_top_x, 0)
    pg_R_bot = (poly_R_bot_x, H_be)

    # Define vertices for the lane polygon based on detected lines.
    # This logic now aligns with the assumption that missing lines are at image edges for visualization.
    verts = np.array([pg_L_top, pg_R_top, pg_R_bot, pg_L_bot], dtype=np.int32)


    # Draw the lane polygon (green overlay)
    lanes_view = bird_eye.copy()
    if verts.size > 0:
        overlay = np.zeros_like(lanes_view)
        # Ensure the polygon is a closed shape with 4 vertices.
        # If L or R coords are (0,0,0,0), their corresponding poly_X values default to 0 or W_be.
        cv2.fillPoly(overlay, [verts], (0,255,0)) # Green color for the lane
        cv2.addWeighted(overlay, 0.3, lanes_view, 0.7, 0, lanes_view) # Blend overlay

    # Draw the detected left and right lane lines (only if they were actually detected)
    if left_coords != (0,0,0,0):
        cv2.line(lanes_view, L_pt1, L_pt2, (255,0,0), 3) # Blue for left line
    if right_coords != (0,0,0,0):
        cv2.line(lanes_view, R_pt1, R_pt2, (0,0,255), 3) # Red for right line

    # Display text information (angles, width, steering)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255) # White color for text
    thickness = 1

    left_text = f"Angolo Sinistro: {left_angle:.1f} deg" if left_angle is not None else "Angolo Sinistro: N/A"
    cv2.putText(lanes_view, left_text, (10, 20), font, font_scale, font_color, thickness, cv2.LINE_AA)

    right_text = f"Angolo Destro: {right_angle:.1f} deg" if right_angle is not None else "Angolo Destro: N/A"
    cv2.putText(lanes_view, right_text, (10, 40), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Display lane_width which is now always calculated
    width_text = f"Larghezza: {lane_width:.1f}"
    cv2.putText(lanes_view, width_text, (10, 60), font, font_scale, font_color, thickness, cv2.LINE_AA)

    steer_text = f"Sterzata: {steering_angle:.1f} deg"
    cv2.putText(lanes_view, steer_text, (10, 80), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Display the processed frames
    cv2.imshow("Corsie Rilevate", lanes_view)
    cv2.imshow("Maschera", mask)
    # Draw the ROI rectangle on the original frame
    cv2.rectangle(frame, (0, roi_top), (w, roi_bottom), (0,255,0), 2)
    cv2.imshow("Video Originale con ROI", frame)

    # Exit condition: press 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Cleanup: release camera, stop robot, close windows
picam2.stop()
picam2.close()
px.forward(0) # Stop the robot
px.set_dir_servo_angle(0) # Straighten the wheels
time.sleep(0.5)
cv2.destroyAllWindows()