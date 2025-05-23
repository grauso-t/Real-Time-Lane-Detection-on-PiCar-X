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
STEERING_SMOOTHING = 0.7  # Smoothing factor for steering (0-1, higher = more smoothing)
MIN_LANE_WIDTH = 180    # Minimum lane width to accept steering changes

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
        y1, y2 = height, int(height * 0.6)  # from bottom to 60% height
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

def calculate_lane_width(left_line, right_line, left_detected, right_detected, frame_width):
    """
    Calculate lane width at the bottom of the frame.
    If only one line is detected, estimate the width based on typical lane dimensions.
    """
    if left_detected and right_detected:
        # Both lines detected - calculate actual width
        lx1, ly1, lx2, ly2 = left_line
        rx1, ry1, rx2, ry2 = right_line
        width = abs(rx1 - lx1)  # Width at bottom
        return width, "actual"
    
    elif left_detected and not right_detected:
        # Only left line - estimate right boundary
        lx1, ly1, lx2, ly2 = left_line
        estimated_right_x = lx1 + 150  # Typical lane width estimate
        # Clamp to frame boundaries
        estimated_right_x = min(estimated_right_x, frame_width - 10)
        width = abs(estimated_right_x - lx1)
        return width, "estimated_from_left"
    
    elif not left_detected and right_detected:
        # Only right line - estimate left boundary
        rx1, ry1, rx2, ry2 = right_line
        estimated_left_x = rx1 - 150  # Typical lane width estimate
        # Clamp to frame boundaries
        estimated_left_x = max(estimated_left_x, 10)
        width = abs(rx1 - estimated_left_x)
        return width, "estimated_from_right"
    
    else:
        # No lines detected
        return 0, "no_detection"

def calculate_lane_direction_angle(left_line, right_line, left_lines_detected, right_lines_detected, frame_width, frame_height):
    """
    Calculate the steering angle based on the direction of the lane.
    Now uses the actual angle of the lane direction for more accurate steering.
    """
    lx1, ly1, lx2, ly2 = left_line if left_line else (0, 0, 0, 0)
    rx1, ry1, rx2, ry2 = right_line if right_line else (0, 0, 0, 0)
    
    # Case 1: Both lines detected - use lane center direction and its angle
    if left_lines_detected and right_lines_detected:
        # Calculate center points
        center_bottom_x = (lx1 + rx1) / 2
        center_top_x = (lx2 + rx2) / 2
        
        # Calculate the angle of the center line relative to vertical
        dx = center_top_x - center_bottom_x
        dy = ly1 - ly2  # Note: y increases downward in image coordinates
        
        # Calculate angle in degrees (0 = straight, positive = right turn, negative = left turn)
        lane_angle_rad = np.arctan2(dx, dy)
        lane_angle_deg = np.degrees(lane_angle_rad)
        
        # Apply proportional steering based on the lane angle
        steering_angle = lane_angle_deg * 1.2  # Amplify response
        
        return np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE), lane_angle_deg, "both_lines"
    
    # Case 2: Only left line detected - use left line angle as reference
    elif left_lines_detected and not right_lines_detected:
        dx = lx2 - lx1
        dy = ly1 - ly2
        left_angle_rad = np.arctan2(dx, dy)
        left_angle_deg = np.degrees(left_angle_rad)
        
        # For single line, we need to interpret the line angle differently
        # If the left line is angled right (positive), we should turn left to center
        steering_angle = -left_angle_deg * 0.8  # Invert and reduce sensitivity
        
        return np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE), left_angle_deg, "left_only"
    
    # Case 3: Only right line detected - use right line angle as reference
    elif not left_lines_detected and right_lines_detected:
        dx = rx2 - rx1
        dy = ry1 - ry2
        right_angle_rad = np.arctan2(dx, dy)
        right_angle_deg = np.degrees(right_angle_rad)
        
        # For single line, if the right line is angled left (negative), we should turn right to center
        steering_angle = -right_angle_deg * 0.8  # Invert and reduce sensitivity
        
        return np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE), right_angle_deg, "right_only"
    
    # Case 4: No lines detected
    else:
        return 0, 0, "no_lines"

def smooth_steering(current_angle, previous_angle, smoothing_factor):
    """
    Apply smoothing to steering angle to reduce jittery movements.
    """
    return smoothing_factor * previous_angle + (1 - smoothing_factor) * current_angle

def draw_angle_visualization(image, center_x, center_y, angle_deg, length=80):
    """
    Draw a line showing the detected angle for steering calculation.
    """
    # Convert angle to radians for calculation
    angle_rad = np.radians(angle_deg)
    
    # Calculate end point of the angle line
    end_x = int(center_x + length * np.sin(angle_rad))
    end_y = int(center_y - length * np.cos(angle_rad))  # Negative because y increases downward
    
    # Draw the angle line
    cv2.line(image, (center_x, center_y), (end_x, end_y), (255, 0, 255), 3)  # Magenta line
    
    # Draw a small circle at the center
    cv2.circle(image, (center_x, center_y), 5, (255, 0, 255), -1)
    
    # Add angle text
    cv2.putText(image, f"Angle: {angle_deg:.1f}°", 
               (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

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
        
        # Track which lines were actually detected
        left_lines_detected = len(left_lines) > 0
        right_lines_detected = len(right_lines) > 0
        
        left_line = average_line(left_lines, 'left', dst_width, dst_height) if left_lines_detected else None
        right_line = average_line(right_lines, 'right', dst_width, dst_height) if right_lines_detected else None
        
        lane_overlay = np.zeros_like(bird_eye_view)
        
        if left_line or right_line:
            # Calculate lane width
            lane_width, width_method = calculate_lane_width(
                left_line, right_line, left_lines_detected, right_lines_detected, dst_width
            )
            
            # Calculate steering angle based on available lines
            raw_steering_angle, detected_angle, detection_status = calculate_lane_direction_angle(
                left_line, right_line, left_lines_detected, right_lines_detected, dst_width, dst_height
            )
            
            # Check if lane width is sufficient for steering adjustment
            if lane_width >= MIN_LANE_WIDTH:
                # Apply smoothing
                steering_angle = smooth_steering(raw_steering_angle, previous_steering_angle, STEERING_SMOOTHING)
                # Update the previous steering angle
                previous_steering_angle = steering_angle
                width_status = "OK"
            else:
                # Lane too narrow, maintain previous steering angle
                steering_angle = previous_steering_angle
                width_status = "TOO_NARROW"
            
            px.set_dir_servo_angle(steering_angle)
            
            # Draw detected lines and lane area
            if left_lines_detected and right_lines_detected:
                # Both lines detected - draw full lane
                lx1, ly1, lx2, ly2 = left_line
                rx1, ry1, rx2, ry2 = right_line
                points = np.array([
                    [lx1, ly1],
                    [rx1, ry1],
                    [rx2, ry2],
                    [lx2, ly2]
                ])
                cv2.fillPoly(lane_overlay, [points], (0, 255, 0))
                
                # Draw center line
                center_bottom_x = int((lx1 + rx1) / 2)
                center_top_x = int((lx2 + rx2) / 2)
                cv2.line(lane_overlay, (center_bottom_x, ly1), (center_top_x, ly2), (255, 0, 0), 3)
                
                # Draw width measurement line at bottom
                cv2.line(lane_overlay, (lx1, ly1), (rx1, ry1), (0, 255, 255), 2)  # Cyan line for width
                
                # Draw angle visualization at center
                center_y = int((ly1 + ly2) / 2)
                draw_angle_visualization(lane_overlay, center_bottom_x, center_y, detected_angle)
                
            elif left_lines_detected:
                # Only left line detected
                lx1, ly1, lx2, ly2 = left_line
                cv2.line(lane_overlay, (lx1, ly1), (lx2, ly2), (0, 255, 255), 4)  # Yellow line
                
                # Draw estimated lane area
                estimated_right_x1 = min(lx1 + 150, dst_width - 10)
                estimated_right_x2 = min(lx2 + 120, dst_width - 10)
                points = np.array([
                    [lx1, ly1],
                    [estimated_right_x1, ly1],
                    [estimated_right_x2, ly2],
                    [lx2, ly2]
                ])
                cv2.fillPoly(lane_overlay, [points], (0, 150, 150))  # Darker green for estimated area
                
                # Draw width measurement line
                cv2.line(lane_overlay, (lx1, ly1), (estimated_right_x1, ly1), (0, 255, 255), 2)
                
                # Draw angle visualization
                center_x = int((lx1 + estimated_right_x1) / 2)
                center_y = int((ly1 + ly2) / 2)
                draw_angle_visualization(lane_overlay, center_x, center_y, detected_angle)
                
            elif right_lines_detected:
                # Only right line detected
                rx1, ry1, rx2, ry2 = right_line
                cv2.line(lane_overlay, (rx1, ry1), (rx2, ry2), (0, 255, 255), 4)  # Yellow line
                
                # Draw estimated lane area
                estimated_left_x1 = max(rx1 - 150, 10)
                estimated_left_x2 = max(rx2 - 120, 10)
                points = np.array([
                    [estimated_left_x1, ry1],
                    [rx1, ry1],
                    [rx2, ry2],
                    [estimated_left_x2, ry2]
                ])
                cv2.fillPoly(lane_overlay, [points], (0, 150, 150))  # Darker green for estimated area
                
                # Draw width measurement line
                cv2.line(lane_overlay, (estimated_left_x1, ry1), (rx1, ry1), (0, 255, 255), 2)
                
                # Draw angle visualization
                center_x = int((estimated_left_x1 + rx1) / 2)
                center_y = int((ry1 + ry2) / 2)
                draw_angle_visualization(lane_overlay, center_x, center_y, detected_angle)
            
            # Display information
            cv2.putText(frame_overlay, f"Steering: {steering_angle:.1f}° ({width_status})", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(frame_overlay, f"Lane Width: {lane_width:.0f}px ({width_method})", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(frame_overlay, f"Detected Angle: {detected_angle:.1f}°", 
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(frame_overlay, f"Detection: {detection_status}", 
                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            bird_eye_lane = cv2.addWeighted(bird_eye_view, 1.0, lane_overlay, 0.5, 0)
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
            STEERING_SMOOTHING = min(STEERING_SMOOTHING + 0.05, 0.95)
            print(f"Smoothing: {STEERING_SMOOTHING:.2f}")
        elif key == ord('-'):
            STEERING_SMOOTHING = max(STEERING_SMOOTHING - 0.05, 0.05)
            print(f"Smoothing: {STEERING_SMOOTHING:.2f}")

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    px.forward(0)
    picam2.stop()
    cv2.destroyAllWindows()