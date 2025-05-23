#!/usr/bin/env python3
import cv2
import numpy as np
from picamera2 import Picamera2
from picarx import PiCarX
import time
import math

class LineFollower:
    def __init__(self):
        # Inizializza PiCarX
        self.car = PiCarX()
        
        # Inizializza camera
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"size": (640, 480)})
        self.picam2.configure(config)
        self.picam2.start()
        
        # Parametri di controllo
        self.speed = 1
        self.max_angle = 45
        self.last_steering_angle = 0
        self.min_lane_width = 50
        
        # Parametri per bird's eye view
        self.dst_height = 200
        self.dst_width = 300
        
        # Parametri Hough
        self.hough_threshold = 50
        self.min_line_length = 50
        self.max_line_gap = 150
        
    def get_bird_eye_transform(self, height, width):
        """Calcola la matrice di trasformazione per bird's eye view"""
        roi_top = int(height * 0.8)
        roi_bottom = height
        
        src_pts = np.float32([
            [0, roi_top],
            [width, roi_top],
            [width, roi_bottom],
            [0, roi_bottom]
        ])
        
        dst_pts = np.float32([
            [0, 0],
            [self.dst_width, 0],
            [self.dst_width, self.dst_height],
            [0, self.dst_height]
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        
        return M, M_inv
    
    def detect_white_yellow_lines(self, frame):
        """Rileva linee bianche e gialle usando filtro HSV"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Maschera per il bianco
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Maschera per il giallo
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combina le maschere
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        return combined_mask
    
    def apply_gaussian_blur(self, image, kernel_size=5):
        """Applica filtro gaussiano"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def detect_lines_hough(self, edge_image):
        """Rileva linee usando trasformata di Hough"""
        lines = cv2.HoughLinesP(
            edge_image,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        return lines
    
    def separate_left_right_lines(self, lines, width):
        """Separa le linee in sinistra e destra"""
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calcola la pendenza
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                # Linee con pendenza negativa sono a sinistra
                if slope < 0 and x1 < width/2 and x2 < width/2:
                    left_lines.append(line)
                # Linee con pendenza positiva sono a destra
                elif slope > 0 and x1 > width/2 and x2 > width/2:
                    right_lines.append(line)
        
        return left_lines, right_lines
    
    def average_lines(self, lines):
        """Calcola la linea media da un gruppo di linee"""
        if not lines:
            return None
        
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        if len(x_coords) < 2:
            return None
        
        # Fit lineare
        poly = np.polyfit(y_coords, x_coords, 1)
        
        # Calcola punti estremi
        y1 = self.dst_height
        y2 = 0
        x1 = int(poly[0] * y1 + poly[1])
        x2 = int(poly[0] * y2 + poly[1])
        
        return [[x1, y1, x2, y2]]
    
    def create_lane_with_triangle(self, left_line, right_line, width, height):
        """Crea la carreggiata con triangoli se manca una linea"""
        lane_points = []
        
        if left_line and right_line:
            # Entrambe le linee presenti
            x1_l, y1_l, x2_l, y2_l = left_line[0]
            x1_r, y1_r, x2_r, y2_r = right_line[0]
            
            lane_points = np.array([
                [x1_l, y1_l], [x2_l, y2_l],
                [x2_r, y2_r], [x1_r, y1_r]
            ], np.int32)
            
        elif left_line and not right_line:
            # Solo linea sinistra, crea triangolo a destra
            x1_l, y1_l, x2_l, y2_l = left_line[0]
            
            lane_points = np.array([
                [x1_l, y1_l], [x2_l, y2_l],
                [width, 0], [width, height]
            ], np.int32)
            
        elif right_line and not left_line:
            # Solo linea destra, crea triangolo a sinistra
            x1_r, y1_r, x2_r, y2_r = right_line[0]
            
            lane_points = np.array([
                [0, height], [0, 0],
                [x2_r, y2_r], [x1_r, y1_r]
            ], np.int32)
        
        return lane_points
    
    def calculate_steering_angle(self, left_line, right_line, width):
        """Calcola l'angolo di sterzata basato sulla carreggiata"""
        if left_line and right_line:
            # Entrambe le linee presenti
            x1_l, y1_l, x2_l, y2_l = left_line[0]
            x1_r, y1_r, x2_r, y2_r = right_line[0]
            
            # Centro della carreggiata
            center_x = (x1_l + x1_r) / 2
            lane_width = abs(x1_r - x1_l)
            
            # Se la carreggiata è troppo stretta, mantieni angolo precedente
            if lane_width < self.min_lane_width:
                return self.last_steering_angle
            
            # Calcola deviazione dal centro
            deviation = center_x - width/2
            angle = math.degrees(math.atan(deviation / (width/2))) * 2
            
        elif left_line and not right_line:
            # Solo linea sinistra
            x1_l, y1_l, x2_l, y2_l = left_line[0]
            # Angolo basato sulla pendenza della linea
            slope = (y2_l - y1_l) / (x2_l - x1_l) if x2_l != x1_l else 0
            angle = math.degrees(math.atan(slope)) + 15  # Bias verso destra
            
        elif right_line and not left_line:
            # Solo linea destra
            x1_r, y1_r, x2_r, y2_r = right_line[0]
            # Angolo basato sulla pendenza della linea
            slope = (y2_r - y1_r) / (x2_r - x1_r) if x2_r != x1_r else 0
            angle = math.degrees(math.atan(slope)) - 15  # Bias verso sinistra
            
        else:
            # Nessuna linea, mantieni angolo precedente
            return self.last_steering_angle
        
        # Limita l'angolo
        angle = max(-self.max_angle, min(self.max_angle, angle))
        return angle
    
    def draw_debug_info(self, frame, lines, left_line, right_line, lane_points):
        """Disegna informazioni di debug"""
        debug_frame = frame.copy()
        
        # Disegna tutte le linee Hough in blu
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Disegna linee medie in verde
        if left_line:
            x1, y1, x2, y2 = left_line[0]
            cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        if right_line:
            x1, y1, x2, y2 = right_line[0]
            cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Disegna overlay carreggiata con trasparenza
        if len(lane_points) > 0:
            overlay = debug_frame.copy()
            cv2.fillPoly(overlay, [lane_points], (0, 255, 255))
            debug_frame = cv2.addWeighted(debug_frame, 0.5, overlay, 0.5, 0)
        
        return debug_frame
    
    def run(self):
        """Loop principale"""
        print("Avvio del line follower...")
        
        try:
            while True:
                # Cattura frame
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                height, width = frame.shape[:2]
                
                # Trasformazione bird's eye
                M, M_inv = self.get_bird_eye_transform(height, width)
                bird_eye = cv2.warpPerspective(frame, M, (self.dst_width, self.dst_height))
                
                # Rileva linee bianche e gialle
                line_mask = self.detect_white_yellow_lines(bird_eye)
                
                # Applica filtro gaussiano
                blurred = self.apply_gaussian_blur(line_mask)
                
                # Rileva linee con Hough
                lines = self.detect_lines_hough(blurred)
                
                # Separa linee sinistra/destra
                left_lines, right_lines = self.separate_left_right_lines(lines, self.dst_width)
                
                # Calcola linee medie
                left_line = self.average_lines(left_lines)
                right_line = self.average_lines(right_lines)
                
                # Crea carreggiata con triangoli
                lane_points = self.create_lane_with_triangle(
                    left_line, right_line, self.dst_width, self.dst_height
                )
                
                # Calcola angolo di sterzata
                steering_angle = self.calculate_steering_angle(
                    left_line, right_line, self.dst_width
                )
                self.last_steering_angle = steering_angle
                
                # Controlla il robot
                self.car.forward(self.speed)
                self.car.set_dir_servo_angle(steering_angle)
                
                # Visualizzazione debug
                debug_frame = self.draw_debug_info(
                    bird_eye, lines, left_line, right_line, lane_points
                )
                
                # Mostra i frame
                cv2.imshow('Original', frame)
                cv2.imshow('Line Mask', line_mask)
                cv2.imshow('Hough Lines', blurred)
                cv2.imshow('Lane Detection', debug_frame)
                
                # Stampa info
                print(f"Steering angle: {steering_angle:.1f}°, Speed: {self.speed}")
                
                # Esci con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Interruzione da tastiera")
        
        finally:
            # Cleanup
            self.car.stop()
            self.picam2.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    follower = LineFollower()
    follower.run()