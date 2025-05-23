import cv2
import numpy as np
import time
from picarx import Picarx  # Import PiCar-X control library
from picamera2 import Picamera2  # Import PiCamera2 library

# Initialize PiCar-X
px = Picarx()
px.forward(0)  # Start with zero speed

# Constants for PiCar control
FORWARD_SPEED = 0       # Forward speed (constant)
MAX_STEERING_ANGLE = 45  # Maximum steering angle
CENTER_OFFSET = 75 # Reduced for faster response
STEERING_AGGRESSION = 4  # Amplification factor for steeper turns
MIN_LANE_WIDTH = 100    # Minimum distance from center to consider valid lane line
MIN_TOTAL_LANE_WIDTH = 180  # Minimum total lane width to update steering

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
    If no lines are detected, return a forced vertical line near the left or right edge.
    """
    if len(lines) == 0:
        x_pos = 20 if side == 'left' else width - 20
        y1, y2 = height, int(height * 0.2)  # from bottom to 20% height
        return (x_pos, y1, x_pos, y2)
   
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

def calculate_steering_angle(left_x, right_x, frame_width):
    """
    Calculate the steering angle based on lane position.
    Adjusted for steeper turns with non-linear response.
    """
    lane_center = (left_x + right_x) // 2
    center_offset = lane_center - (frame_width // 2)
    
    max_offset = frame_width // 4  # Reduced for more sensitivity
    normalized_offset = center_offset / max_offset
    
    if normalized_offset != 0:
        sign = np.sign(normalized_offset)
        nonlinear_response = sign * (abs(normalized_offset) ** 0.8) * STEERING_AGGRESSION
        if abs(nonlinear_response) > 1:
            nonlinear_response = sign * 1.0
    else:
        nonlinear_response = 0
    
    steering_angle = nonlinear_response * MAX_STEERING_ANGLE
    return max(min(steering_angle, MAX_STEERING_ANGLE), -MAX_STEERING_ANGLE)

def draw_lane_width_visualization(img, left_x, right_x, lane_width, y_pos, is_valid_width):
    """
    Draw lane width visualization with color coding
    """
    # Determine color based on lane width and validity
    if not is_valid_width:
        color = (0, 0, 255)  # Red for invalid/narrow lane
        status = "INVALID"
    else:
        color = (0, 255, 0)  # Green for acceptable lane width
        status = "VALID"
    
    # Draw horizontal line showing lane width
    cv2.line(img, (left_x, y_pos), (right_x, y_pos), color, 3)
    
    # Draw vertical markers at lane edges
    cv2.line(img, (left_x, y_pos - 10), (left_x, y_pos + 10), color, 2)
    cv2.line(img, (right_x, y_pos - 10), (right_x, y_pos + 10), color, 2)
    
    # Draw lane width text
    mid_x = (left_x + right_x) // 2
    cv2.putText(img, f"{lane_width:.0f}px [{status}]", 
                (mid_x - 50, y_pos - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return status

def draw_steering_angle_visualization(img, steering_angle, center_x, center_y, is_angle_updated):
    """
    Draw steering angle visualization with arrow and arc
    """
    # Calculate arrow endpoint based on steering angle
    angle_rad = np.deg2rad(-steering_angle)  # Negative for correct direction
    arrow_length = 60
    end_x = int(center_x + arrow_length * np.sin(angle_rad))
    end_y = int(center_y - arrow_length * np.cos(angle_rad))
    
    # Determine color based on whether angle was updated
    if is_angle_updated:
        # Color based on steering intensity
        abs_angle = abs(steering_angle)
        if abs_angle < 10:
            color = (0, 255, 0)  # Green for small angles
        elif abs_angle < 25:
            color = (0, 255, 255)  # Yellow for medium angles
        else:
            color = (0, 0, 255)  # Red for large angles
    else:
        color = (128, 128, 128)  # Gray for maintained previous angle
    
    # Draw center point
    cv2.circle(img, (center_x, center_y), 5, (255, 255, 255), -1)
    
    # Draw steering direction arrow
    cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), color, 3, tipLength=0.3)
    
    # Draw arc showing angle range
    if abs(steering_angle) > 2:  # Only draw arc for significant angles
        start_angle = -90 - abs(steering_angle) if steering_angle > 0 else -90
        end_angle = -90 + abs(steering_angle) if steering_angle > 0 else -90 + abs(steering_angle)
        
        cv2.ellipse(img, (center_x, center_y), (40, 40), 0, 
                   start_angle, end_angle, color, 2)

def filter_lines_by_distance_from_center(lines, frame_center, min_distance):
    """
    Filter out lines that are too close to the center of the frame
    """
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]  # Extract coordinates from nested array
        # Calculate average x position of the line
        avg_x = (x1 + x2) / 2
        # Check if line is far enough from center
        if abs(avg_x - frame_center) > min_distance:
            filtered_lines.append(line)
    return filtered_lines

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
            # Filter lines that are too close to center
            frame_center = dst_width // 2
            filtered_lines = filter_lines_by_distance_from_center(lines, frame_center, MIN_LANE_WIDTH)
            
            for line in filtered_lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                
                if abs(slope) < 0.6:
                    continue
                
                # Check which side of center the line is on
                avg_x = (x1 + x2) / 2
                if avg_x < frame_center and slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                elif avg_x > frame_center and slope > 0:
                    right_lines.append((x1, y1, x2, y2))
        
        # Calculate average lines or use default positions if not detected
        left_line = average_line(left_lines, 'left', dst_width, dst_height)
        right_line = average_line(right_lines, 'right', dst_width, dst_height)
        
        lane_overlay = np.zeros_like(bird_eye_view)
        
        # Initialize variables for steering calculation
        steering_angle = previous_steering_angle
        is_angle_updated = False
        
        if left_line and right_line:
            lx1, ly1, lx2, ly2 = left_line
            rx1, ry1, rx2, ry2 = right_line
            
            # Draw lane lines
            cv2.line(bird_eye_view, (lx1, ly1), (lx2, ly2), (255, 0, 0), 3)  # Blue for left
            cv2.line(bird_eye_view, (rx1, ry1), (rx2, ry2), (0, 255, 255), 3)  # Cyan for right
            
            # Fill lane area
            points = np.array([
                [lx1, ly1],
                [rx1, ry1],
                [rx2, ry2],
                [lx2, ly2]
            ])
            cv2.fillPoly(lane_overlay, [points], (0, 255, 0))
            
            left_bottom_x = lx1
            right_bottom_x = rx1
            
            # Calculate lane width
            lane_width = abs(left_bottom_x - right_bottom_x)
            
            # Check if lane width is acceptable and lines are not too close to center
            left_distance_from_center = abs(left_bottom_x - dst_width // 2)
            right_distance_from_center = abs(right_bottom_x - dst_width // 2)
            
            is_valid_width = (lane_width >= MIN_TOTAL_LANE_WIDTH and 
                            left_distance_from_center >= MIN_LANE_WIDTH and 
                            right_distance_from_center >= MIN_LANE_WIDTH)
            
            # Update steering angle only if conditions are met
            if is_valid_width:
                steering_angle = calculate_steering_angle(left_bottom_x, right_bottom_x, dst_width)
                previous_steering_angle = steering_angle
                is_angle_updated = True
            
            # Set servo angle
            px.set_dir_servo_angle(steering_angle)
            
            # Draw visualizations
            lane_width_status = draw_lane_width_visualization(
                bird_eye_view, left_bottom_x, right_bottom_x, lane_width, 
                dst_height - 30, is_valid_width
            )
            
            draw_steering_angle_visualization(
                bird_eye_view, steering_angle, dst_width // 2, dst_height - 60, is_angle_updated
            )
            
            # Display information on main frame
            status_text = "UPDATED" if is_angle_updated else "MAINTAINED"
            color = (0, 255, 0) if is_angle_updated else (0, 165, 255)
            
            cv2.putText(frame_overlay, f"Steering: {steering_angle:.1f}Â° [{status_text}]", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(frame_overlay, f"Lane Width: {lane_width:.0f}px [{lane_width_status}]", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(frame_overlay, f"Aggression: {STEERING_AGGRESSION:.1f}x", 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            bird_eye_lane = cv2.addWeighted(bird_eye_view, 1.0, lane_overlay, 0.3, 0)
        else:
            # No lanes detected - maintain previous angle
            px.set_dir_servo_angle(steering_angle)
            cv2.putText(frame_overlay, f"No lanes - Maintaining: {steering_angle:.1f}Â°", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            bird_eye_lane = bird_eye_view.copy()
        
        # Draw center line for reference
        cv2.line(bird_eye_lane, (dst_width // 2, 0), (dst_width // 2, dst_height), (255, 255, 255), 1)
        
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