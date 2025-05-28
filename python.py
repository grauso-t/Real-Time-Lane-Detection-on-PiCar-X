import cv2
import numpy as np
import time
from picarx import Picarx  # Import PiCar-X control library
from picamera2 import Picamera2  # Import PiCamera2 library

# Initialize PiCar-X
px = Picarx()
px.forward(0)  # Start with zero speed

# Constants for PiCar control
FORWARD_SPEED = 10       # Forward speed (constant)
MAX_STEERING_ANGLE = 45  # Maximum steering angle
LANE_OFFSET = 50         # Offset desiderato dalla linea (in pixel)
SLOPE_SENSITIVITY = 30   # Sensibilità alla pendenza (gradi per unità di slope)

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Wait for camera to initialize
time.sleep(2)

# Initialize previous steering angle
previous_steering_angle = 0  # Start with neutral steering

def calculate_line_slope_and_intercept(line):
    """
    Calcola pendenza e intercetta di una linea
    Returns: (slope, intercept) o (None, None) se la linea è verticale
    """
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:  # Linea verticale
        return None, None
    
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def get_parallel_line_with_offset(slope, intercept, offset_distance, frame_width):
    """
    Calcola una linea parallela con offset specificato
    offset_distance: distanza in pixel (positiva = verso destra)
    """
    if slope is None:  # Linea verticale
        return None, None
    
    # Per una linea y = mx + b, la linea parallela è y = mx + (b + offset_adjusted)
    # L'offset deve essere adjusted per la pendenza
    if slope == 0:  # Linea orizzontale
        offset_adjusted = offset_distance
    else:
        # Calcola l'offset perpendicolare alla linea
        angle = np.arctan(slope)
        offset_adjusted = offset_distance / np.cos(angle)
    
    new_intercept = intercept + offset_adjusted
    return slope, new_intercept

def calculate_steering_from_slope(target_slope, current_position_x, frame_width):
    """
    Calcola l'angolo di sterzata basato sulla pendenza della linea target
    e sulla posizione corrente del veicolo
    """
    if target_slope is None:
        return 0
    
    # Converti la pendenza in angolo (in radianti)
    target_angle_rad = np.arctan(target_slope)
    # Converti in gradi
    target_angle_deg = np.degrees(target_angle_rad)
    
    # Calcola l'errore di posizione laterale
    center_x = frame_width // 2
    lateral_error = current_position_x - center_x
    
    # Combina l'angolo della linea con la correzione per l'errore laterale
    lateral_correction = (lateral_error / (frame_width // 4)) * 15  # Max 15 gradi per correzione laterale
    
    # L'angolo di sterzata finale
    steering_angle = target_angle_deg * SLOPE_SENSITIVITY / 100 + lateral_correction
    
    return max(min(steering_angle, MAX_STEERING_ANGLE), -MAX_STEERING_ANGLE)

def average_line_with_slope(lines, side, width, height):
    """
    Calculate average line and return slope information.
    """
    if len(lines) == 0:
        x_pos = 20 if side == 'left' else width - 20
        y1, y2 = height, int(height * 0.1)
        return (x_pos, y1, x_pos, y2), None  # Vertical line has undefined slope
   
    x_coords = []
    y_coords = []
    for x1, y1, x2, y2 in lines:
        x_coords += [x1, x2]
        y_coords += [y1, y2]
    
    poly = np.polyfit(y_coords, x_coords, 1)
    slope = poly[0]  # Coefficiente angolare
    
    y1, y2 = height, int(height * 0.2)
    x1 = int(poly[0] * y1 + poly[1])
    x2 = int(poly[0] * y2 + poly[1])
    
    return (x1, y1, x2, y2), slope

def follow_line_with_offset(reference_line, reference_slope, offset_pixels, frame_width, frame_height):
    """
    Calcola la traiettoria da seguire basata sulla linea di riferimento con offset
    """
    if reference_slope is None:
        # Linea verticale - mantieni offset orizzontale
        x1, y1, x2, y2 = reference_line
        offset_x1 = x1 + offset_pixels
        offset_x2 = x2 + offset_pixels
        return (offset_x1, y1, offset_x2, y2), None
    
    # Calcola intercetta della linea di riferimento
    x1, y1, x2, y2 = reference_line
    intercept = y1 - reference_slope * x1
    
    # Calcola la linea parallela con offset
    offset_slope, offset_intercept = get_parallel_line_with_offset(
        reference_slope, intercept, offset_pixels, frame_width
    )
    
    if offset_slope is None:
        return reference_line, None
    
    # Calcola i punti della linea offset
    y1_new, y2_new = frame_height, int(frame_height * 0.2)
    x1_new = int((y1_new - offset_intercept) / offset_slope) if offset_slope != 0 else int(offset_intercept)
    x2_new = int((y2_new - offset_intercept) / offset_slope) if offset_slope != 0 else int(offset_intercept)
    
    return (x1_new, y1_new, x2_new, y2_new), offset_slope

try:
    px.forward(FORWARD_SPEED)
    
    while True:
        frame = picam2.capture_array()
            
        height, width = frame.shape[:2]
        roi_top = int(height * 0.8)
        roi_bottom = height
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, roi_top), (width, roi_bottom), (0, 0, 0), -1)
        alpha = 0.5
        frame_overlay = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        src_pts = np.float32([
            [0, roi_top],
            [width, roi_top],
            [width, roi_bottom],
            [0, roi_bottom]
        ])
        dst_height = 200
        dst_width = 300
        dst_pts = np.float32([
            [0, 0],
            [dst_width, 0],
            [dst_width, dst_height],
            [0, dst_height]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        bird_eye_view = cv2.warpPerspective(frame, matrix, (dst_width, dst_height))
        
        hls = cv2.cvtColor(bird_eye_view, cv2.COLOR_BGR2HLS)
        
        lower_yellow = np.array([15, 40, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)
        
        lower_white = np.array([0, 200, 0])
        upper_white = np.array([180, 255, 255])
        mask_white = cv2.inRange(hls, lower_white, upper_white)
        
        mask_combined = cv2.bitwise_or(mask_white, mask_yellow)
        
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
        
        edges = cv2.Canny(mask_cleaned, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=120)
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:  # Skip vertical lines in classification
                    continue
                slope = (y2 - y1) / (x2 - x1)
                
                if abs(slope) < 0.6:
                    continue
                    
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))
        
        left_line, left_slope = average_line_with_slope(left_lines, 'left', dst_width, dst_height)
        right_line, right_slope = average_line_with_slope(right_lines, 'right', dst_width, dst_height)
        
        lane_overlay = np.zeros_like(bird_eye_view)
        
        # Determina quale linea usare come riferimento (preferisci quella più vicina al centro)
        reference_line = None
        reference_slope = None
        target_line = None
        target_slope = None
        
        if left_line and right_line:
            # Usa la linea destra come riferimento e mantieni offset verso sinistra
            reference_line = right_line
            reference_slope = right_slope
            target_line, target_slope = follow_line_with_offset(
                reference_line, reference_slope, -LANE_OFFSET, dst_width, dst_height
            )
        elif right_line:
            # Solo linea destra disponibile
            reference_line = right_line
            reference_slope = right_slope
            target_line, target_slope = follow_line_with_offset(
                reference_line, reference_slope, -LANE_OFFSET, dst_width, dst_height
            )
        elif left_line:
            # Solo linea sinistra disponibile
            reference_line = left_line
            reference_slope = left_slope
            target_line, target_slope = follow_line_with_offset(
                reference_line, reference_slope, LANE_OFFSET, dst_width, dst_height
            )
        
        if target_line and target_slope is not None:
            # Disegna la linea di riferimento in blu
            if reference_line:
                x1, y1, x2, y2 = reference_line
                cv2.line(lane_overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blu
            
            # Disegna la linea target in verde
            tx1, ty1, tx2, ty2 = target_line
            cv2.line(lane_overlay, (tx1, ty1), (tx2, ty2), (0, 255, 0), 4)  # Verde
            
            # Calcola steering basato sulla pendenza della linea target
            vehicle_x = dst_width // 2  # Posizione del veicolo al centro
            steering_angle = calculate_steering_from_slope(target_slope, tx1, dst_width)
            
            # Update the previous steering angle
            previous_steering_angle = steering_angle
            
            px.set_dir_servo_angle(steering_angle)
            
            cv2.putText(frame_overlay, f"Steering: {steering_angle:.1f} deg", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(frame_overlay, f"Target Slope: {target_slope:.3f}", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(frame_overlay, f"Offset: {LANE_OFFSET}px", 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            bird_eye_lane = cv2.addWeighted(bird_eye_view, 1.0, lane_overlay, 0.8, 0)
        else:
            # Mantieni l'ultimo angolo di sterzata se non ci sono linee
            px.set_dir_servo_angle(previous_steering_angle)
            cv2.putText(frame_overlay, "No reference line - Keeping previous angle", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            bird_eye_lane = bird_eye_view.copy()
        
        cv2.imshow("Carreggiata Rilevata", bird_eye_lane)
        cv2.imshow("Camera View", frame_overlay)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            px.forward(0)
            px.set_dir_servo_angle(0)
            time.sleep(30)
            px.forward(FORWARD_SPEED)
        elif key == ord('+'): 
            LANE_OFFSET = min(LANE_OFFSET + 5, 100)
        elif key == ord('-'):
            LANE_OFFSET = max(LANE_OFFSET - 5, 10)
        elif key == ord('u'):  # Increase slope sensitivity
            SLOPE_SENSITIVITY = min(SLOPE_SENSITIVITY + 5, 100)
        elif key == ord('d'):  # Decrease slope sensitivity
            SLOPE_SENSITIVITY = max(SLOPE_SENSITIVITY - 5, 10)

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    px.forward(0)
    picam2.stop()
    cv2.destroyAllWindows()