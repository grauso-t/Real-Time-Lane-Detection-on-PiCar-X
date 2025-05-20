import cv2
import numpy as np
import time
from picarx import Picarx  # Import PiCar-X control library
from picamera2 import Picamera2  # Import PiCamera2 library

# Initialize PiCar-X
px = Picarx()
px.forward(0)  # Start with zero speed

# Constants for PiCar control
FORWARD_SPEED = 5       # Forward speed (constant)
MAX_STEERING_ANGLE = 40  # Maximum steering angle
CENTER_OFFSET = 0       # Offset from center (adjust if needed)

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Wait for camera to initialize
time.sleep(2)

def average_line(lines, side, width, height):
    """
    Calcola linea media.
    Se non ci sono linee, ritorna una linea verticale "forzata"
    side: 'left' o 'right'
    """
    if len(lines) == 0:
        # Forza linea verticale vicino al bordo sinistro o destro
        x_pos = 10 if side == 'left' else width - 10
        y1, y2 = height, int(height * 0.6)  # da basso a 60% altezza
        return (x_pos, y1, x_pos, y2)
   
    x_coords = []
    y_coords = []
    for x1, y1, x2, y2 in lines:
        x_coords += [x1, x2]
        y_coords += [y1, y2]
    poly = np.polyfit(y_coords, x_coords, 1)
    y1, y2 = height, int(height * 0.6)
    x1 = int(poly[0] * y1 + poly[1])
    x2 = int(poly[0] * y2 + poly[1])
    return (x1, y1, x2, y2)

def calculate_steering_angle(left_x, right_x, frame_width):
    """
    Calculate steering angle based on lane position
    Positive angle: turn right
    Negative angle: turn left
    """
    # Calculate center of lane
    lane_center = (left_x + right_x) // 2
    
    # Calculate offset from center of frame
    center_offset = lane_center - (frame_width // 2)
    
    # Calculate steering angle proportionally
    # Map the offset to an angle between -MAX_STEERING_ANGLE and MAX_STEERING_ANGLE
    max_offset = frame_width // 3  # Define maximum reasonable offset
    steering_angle = (center_offset / max_offset) * MAX_STEERING_ANGLE
    
    # Limit to maximum steering angle
    return max(min(steering_angle, MAX_STEERING_ANGLE), -MAX_STEERING_ANGLE)

try:
    # Set constant forward speed
    px.forward(FORWARD_SPEED)
    
    while True:
        # Capture frame from PiCamera2
        frame = picam2.capture_array()
            
        height, width = frame.shape[:2]
        roi_top = int(height * 0.6)  # Adjust ROI region for camera
        roi_bottom = height
        
        # Create ROI overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, roi_top), (width, roi_bottom), (0, 0, 0), -1)
        alpha = 0.5
        frame_overlay = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Bird's eye view transformation
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
        
        # Color filtering (HLS)
        hls = cv2.cvtColor(bird_eye_view, cv2.COLOR_BGR2HLS)
        
        # Yellow lane detection
        lower_yellow = np.array([15, 40, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)
        
        # White lane detection
        lower_white = np.array([0, 200, 0])
        upper_white = np.array([180, 255, 255])
        mask_white = cv2.inRange(hls, lower_white, upper_white)
        
        # Combine masks
        mask_combined = cv2.bitwise_or(mask_white, mask_yellow)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
        
        # Edge detection
        edges = cv2.Canny(mask_cleaned, 50, 150)
        
        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=120)
        
        # Separate left and right lines
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
                
                # Skip horizontal lines
                if abs(slope) < 0.5:
                    continue
                    
                # Categorize lines by slope
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                else:
                    right_lines.append((x1, y1, x2, y2))
        
        # Average lines
        left_line = average_line(left_lines, 'left', dst_width, dst_height)
        right_line = average_line(right_lines, 'right', dst_width, dst_height)
        
        # Create lane overlay
        lane_overlay = np.zeros_like(bird_eye_view)
        
        # Control PiCar-X based on lane detection
        if left_line and right_line:
            lx1, ly1, lx2, ly2 = left_line
            rx1, ry1, rx2, ry2 = right_line
            
            # Create lane polygon
            points = np.array([
                [lx1, ly1],
                [rx1, ry1],
                [rx2, ry2],
                [lx2, ly2]
            ])
            cv2.fillPoly(lane_overlay, [points], (0, 255, 0))
            
            # Calculate bottom points of lane lines (closest to car)
            left_bottom_x = lx1
            right_bottom_x = rx1
            
            # Calculate steering angle
            steering_angle = calculate_steering_angle(left_bottom_x, right_bottom_x, dst_width)
            
            # Apply steering to PiCar-X (negate angle because positive is right in our calculation)
            px.set_dir_servo_angle(-steering_angle)
            
            # Display steering information
            cv2.putText(frame_overlay, f"Steering: {steering_angle:.1f} deg", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Combine bird eye view with lane overlay
            bird_eye_lane = cv2.addWeighted(bird_eye_view, 1.0, lane_overlay, 0.5, 0)
        else:
            # If no lane detected, keep moving at constant speed but center steering
            px.set_dir_servo_angle(0)
            cv2.putText(frame_overlay, "Lane not detected - Centered steering", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            bird_eye_lane = bird_eye_view.copy()
        
        # Display results
        cv2.imshow("Video con ROI evidenziata", frame_overlay)
        # cv2.imshow("Vista Bird Eye sulla ROI", bird_eye_view)
        # cv2.imshow("Filtrati Bianco e Giallo (HLS)", cv2.bitwise_and(bird_eye_view, bird_eye_view, mask=mask_combined))
        cv2.imshow("Carreggiata Rilevata", bird_eye_lane)
        
        # Check for key press (q to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('s'):
            px.forward(0)
            px.set_dir_servo_angle(0)
            time.sleep(30)
            px.forward(FORWARD_SPEED)

except KeyboardInterrupt:
    print("Programma interrotto dall'utente")

finally:
    # Clean up
    px.forward(0)
    picam2.stop()
    cv2.destroyAllWindows()