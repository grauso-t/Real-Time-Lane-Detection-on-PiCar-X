import cv2
import numpy as np
import time
from picarx import Picarx  # Import PiCar-X control library
from picamera2 import Picamera2  # Import PiCamera2 library

# Initialize PiCar-X
px = Picarx()
px.forward(0)  # Start with zero speed

# Constants for PiCar control
FORWARD_SPEED = 1       # Forward speed (constant)
MAX_STEERING_ANGLE = 45  # Maximum steering angle
STEERING_AGGRESSION = 1.5  # Amplification factor for steeper turns

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Wait for camera to initialize
time.sleep(2)

# Initialize previous steering angle
previous_steering_angle = 0  # Start with neutral steering

def average_line(lines, side, width, height):
    """
    Calculate average line.
    Se non ci sono linee, crea una linea dall'angolo corrispondente
    verso il centro dell'immagine per creare una carreggiata triangolare.
    """
    if len(lines) == 0:
        if side == 'left':
            # Punto nell'angolo in basso a sinistra
            x1, y1 = 0, height
            # Punto verso il centro dell'immagine nella parte alta
            x2, y2 = width // 3, int(height * 0.2)
        else:  # side == 'right'
            # Punto nell'angolo in basso a destra
            x1, y1 = width, height
            # Punto verso il centro dell'immagine nella parte alta
            x2, y2 = (2 * width) // 3, int(height * 0.2)
        
        return (x1, y1, x2, y2)
   
    # Calcolo normale se le linee sono presenti
    x_coords = []
    y_coords = []
    for x1, y1, x2, y2 in lines:
        x_coords += [x1, x2]
        y_coords += [y1, y2]
    poly = np.polyfit(y_coords, x_coords, 1)
    y1, y2 = height, int(height * 0.2)
    x1 = int(poly[0] * y1 + poly[1])
    x2 = int(poly[0] * y2 + poly[1])
    return (x1, y1, x2, y2)

def calculate_steering_angle_from_lane_angle(left_line, right_line, frame_width, frame_height):
    """
    Calcola l'angolo di sterzata basandosi sull'angolo della carreggiata.
    Questo approccio è più naturale quando una delle linee manca.
    """
    lx1, ly1, lx2, ly2 = left_line
    rx1, ry1, rx2, ry2 = right_line
    
    # Calcola i vettori delle due linee
    left_vector = np.array([lx2 - lx1, ly2 - ly1])
    right_vector = np.array([rx2 - rx1, ry2 - ry1])
    
    # Calcola gli angoli delle linee rispetto all'orizzontale
    left_angle = np.arctan2(left_vector[1], left_vector[0])
    right_angle = np.arctan2(right_vector[1], right_vector[0])
    
    # Angolo medio della carreggiata (direzione della strada)
    lane_angle = (left_angle + right_angle) / 2
    
    # Converti in gradi per debug
    lane_angle_deg = np.degrees(lane_angle)
    
    # L'angolo di sterzata dovrebbe essere opposto all'inclinazione della carreggiata
    # Se la carreggiata punta verso destra (positivo), sterziamo a sinistra (negativo)
    steering_response = -np.tan(lane_angle) * STEERING_AGGRESSION
    
    # Normalizza la risposta
    steering_response = max(min(steering_response, 1.0), -1.0)
    
    # Calcola l'angolo di sterzata finale
    steering_angle = steering_response * MAX_STEERING_ANGLE
    
    return steering_angle, lane_angle_deg

def draw_lane_angle_indicator(image, lane_angle_deg, steering_angle):
    """
    Disegna un indicatore visuale dell'angolo della carreggiata e della sterzata
    """
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height - 30
    
    # Disegna l'angolo della carreggiata
    angle_rad = np.radians(lane_angle_deg)
    line_length = 60
    end_x = int(center_x + line_length * np.cos(angle_rad))
    end_y = int(center_y + line_length * np.sin(angle_rad))
    
    # Linea che rappresenta la direzione della carreggiata
    cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), (255, 255, 0), 3)
    cv2.putText(image, f"Lane Angle: {lane_angle_deg:.1f}°", 
                (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Linea di riferimento orizzontale
    cv2.line(image, (center_x - 40, center_y), (center_x + 40, center_y), (128, 128, 128), 1)
    
    # Indicatore della sterzata
    steering_color = (0, 255, 0) if abs(steering_angle) < 15 else (0, 165, 255) if abs(steering_angle) < 30 else (0, 0, 255)
    cv2.putText(image, f"Steering: {steering_angle:.1f}°", 
                (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, steering_color, 2)

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
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                
                if abs(slope) < 0.6:
                    continue
                    
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))
        
        left_line = average_line(left_lines, 'left', dst_width, dst_height)
        right_line = average_line(right_lines, 'right', dst_width, dst_height)
        
        lane_overlay = np.zeros_like(bird_eye_view)
        
        if left_line and right_line:
            lx1, ly1, lx2, ly2 = left_line
            rx1, ry1, rx2, ry2 = right_line
            
            # Disegna le linee individuate
            cv2.line(lane_overlay, (lx1, ly1), (lx2, ly2), (255, 0, 0), 3)  # Linea sinistra in blu
            cv2.line(lane_overlay, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)  # Linea destra in rosso
            
            # Crea il poligono della carreggiata
            points = np.array([
                [lx1, ly1],
                [rx1, ry1],
                [rx2, ry2],
                [lx2, ly2]
            ])
            cv2.fillPoly(lane_overlay, [points], (0, 255, 0))
            
            # Calcola l'angolo di sterzata basato sull'angolo della carreggiata
            steering_angle, lane_angle_deg = calculate_steering_angle_from_lane_angle(
                left_line, right_line, dst_width, dst_height
            )
            
            # Applica un filtro per evitare oscillazioni brusche
            if abs(steering_angle - previous_steering_angle) > 10:
                steering_angle = previous_steering_angle + np.sign(steering_angle - previous_steering_angle) * 10
            
            # Aggiorna l'angolo precedente
            previous_steering_angle = steering_angle
            
            # Applica l'angolo di sterzata
            px.set_dir_servo_angle(steering_angle)
            
            # Disegna gli indicatori di debug sulla vista bird's eye
            center_x, center_y = dst_width // 2, dst_height - 20
            
            # Calcola e disegna la direzione media della carreggiata
            angle_rad = np.radians(lane_angle_deg)
            line_length = 40
            end_x = int(center_x + line_length * np.cos(angle_rad))
            end_y = int(center_y + line_length * np.sin(angle_rad))
            
            cv2.arrowedLine(lane_overlay, (center_x, center_y), (end_x, end_y), (255, 255, 0), 2)
            cv2.circle(lane_overlay, (center_x, center_y), 3, (255, 255, 255), -1)
            
            # Aggiungi testo informativo
            cv2.putText(lane_overlay, f"Lane: {lane_angle_deg:.1f}°", 
                      (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(lane_overlay, f"Steer: {steering_angle:.1f}°", 
                      (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(lane_overlay, f"Aggr: {STEERING_AGGRESSION:.1f}x", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Disegna l'indicatore dell'angolo sulla vista principale
            draw_lane_angle_indicator(frame_overlay, lane_angle_deg, steering_angle)
            
            bird_eye_lane = cv2.addWeighted(bird_eye_view, 0.7, lane_overlay, 0.3, 0)
        else:
            px.set_dir_servo_angle(0)
            cv2.putText(frame_overlay, "Lane not detected - Centered steering", 
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
            STEERING_AGGRESSION = min(STEERING_AGGRESSION + 0.1, 3.0)
        elif key == ord('-'):
            STEERING_AGGRESSION = max(STEERING_AGGRESSION - 0.1, 1.0)

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    px.forward(0)
    picam2.stop()
    cv2.destroyAllWindows()