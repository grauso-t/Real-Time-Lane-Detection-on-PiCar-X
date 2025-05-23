import cv2
import numpy as np
import math
from picamera2 import Picamera2
from picarx import Picarx

class LineFollower:
    def __init__(self):
        # Inizializza PicarX
        self.px = Picarx()
        
        # Inizializza PiCamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (640, 480)})
        self.picam2.configure(config)
        self.picam2.start()
        
        # Parametri di controllo
        self.speed = 1
        self.max_angle = 45
        
        # Parametri per bird's eye view
        self.dst_height = 200
        self.dst_width = 300
        
        # Parametri per filtro HSV
        # Range per linee bianche
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 25, 255])
        
        # Range per linee gialle
        self.yellow_lower = np.array([15, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])
        
    def get_bird_eye_transform(self, width, height):
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
        
        return cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    def detect_lines_hsv(self, frame):
        """Rileva linee bianche e gialle usando filtro HSV"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Maschera per linee bianche
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # Maschera per linee gialle
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # Combina le maschere
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        return combined_mask, white_mask, yellow_mask
    
    def apply_gaussian_hough(self, mask):
        """Applica filtro gaussiano e Hough Transform"""
        # Filtro gaussiano
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Hough Transform per rilevare linee
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=20)
        
        return edges, lines
    
    def classify_lines(self, lines, width):
        """Classifica le linee in sinistra e destra"""
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calcola la pendenza
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Classifica in base alla posizione e pendenza
                    center_x = (x1 + x2) / 2
                    
                    if center_x < width / 2 and slope < 0:  # Linea sinistra
                        left_lines.append(line[0])
                    elif center_x > width / 2 and slope > 0:  # Linea destra
                        right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def extrapolate_line(self, lines, height, width):
        """Estrapolazione di una linea dai segmenti rilevati"""
        if not lines:
            return None
            
        # Calcola la linea media
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        if len(x_coords) < 2:
            return None
            
        # Fit lineare
        coeffs = np.polyfit(y_coords, x_coords, 1)
        
        # Calcola punti estremi
        y_top = 0
        y_bottom = height
        x_top = int(coeffs[0] * y_top + coeffs[1])
        x_bottom = int(coeffs[0] * y_bottom + coeffs[1])
        
        return [(x_top, y_top), (x_bottom, y_bottom)]
    
    def create_lane_overlay(self, frame, left_line, right_line):
        """Crea overlay della carreggiata con triangolo se manca una linea"""
        overlay = np.zeros_like(frame)
        height, width = frame.shape[:2]
        
        # Punti del poligono della carreggiata
        if left_line and right_line:
            # Entrambe le linee presenti - trapezio normale
            points = np.array([
                left_line[0], left_line[1],
                right_line[1], right_line[0]
            ], np.int32)
        elif left_line and not right_line:
            # Solo linea sinistra - triangolo a destra
            points = np.array([
                left_line[0], left_line[1],
                (width, height), (width, 0)
            ], np.int32)
        elif right_line and not left_line:
            # Solo linea destra - triangolo a sinistra
            points = np.array([
                (0, 0), (0, height),
                right_line[1], right_line[0]
            ], np.int32)
        else:
            # Nessuna linea - nessun overlay
            return overlay
        
        # Riempi il poligono
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        
        return overlay
    
    def calculate_steering_angle(self, left_line, right_line, width):
        """Calcola l'angolo di sterzata"""
        height = self.dst_height
        
        if left_line and right_line:
            # Entrambe le linee presenti - segui il centro
            left_bottom = left_line[1][0]
            right_bottom = right_line[1][0]
            lane_center = (left_bottom + right_bottom) / 2
            frame_center = width / 2
            
        elif left_line and not right_line:
            # Solo linea sinistra - calcola angolo del triangolo
            x1, y1 = left_line[0]
            x2, y2 = left_line[1]
            
            # Angolo della linea rispetto alla verticale
            angle_rad = math.atan2(x2 - x1, y2 - y1)
            steering_angle = math.degrees(angle_rad)
            
            return max(-self.max_angle, min(self.max_angle, steering_angle))
            
        elif right_line and not left_line:
            # Solo linea destra - calcola angolo del triangolo
            x1, y1 = right_line[0]
            x2, y2 = right_line[1]
            
            # Angolo della linea rispetto alla verticale
            angle_rad = math.atan2(x2 - x1, y2 - y1)
            steering_angle = -math.degrees(angle_rad)
            
            return max(-self.max_angle, min(self.max_angle, steering_angle))
        else:
            # Nessuna linea - vai dritto
            return 0
        
        # Calcola errore dal centro
        error = lane_center - frame_center
        steering_angle = error * 0.3  # Fattore di conversione
        
        return max(-self.max_angle, min(self.max_angle, steering_angle))
    
    def run(self):
        """Loop principale"""
        try:
            while True:
                # Cattura frame
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                height, width = frame.shape[:2]
                
                # Trasformazione bird's eye view
                M = self.get_bird_eye_transform(width, height)
                bird_eye = cv2.warpPerspective(frame, M, (self.dst_width, self.dst_height))
                
                # Rilevamento linee con HSV
                combined_mask, white_mask, yellow_mask = self.detect_lines_hsv(bird_eye)
                
                # Applicazione Gaussian e Hough
                edges, lines = self.apply_gaussian_hough(combined_mask)
                
                # Classificazione linee
                left_lines, right_lines = self.classify_lines(lines, self.dst_width)
                
                # Estrapolazione linee
                left_line = self.extrapolate_line(left_lines, self.dst_height, self.dst_width)
                right_line = self.extrapolate_line(right_lines, self.dst_height, self.dst_width)
                
                # Calcola angolo di sterzata
                steering_angle = self.calculate_steering_angle(left_line, right_line, self.dst_width)
                
                # Controllo del robot
                self.px.forward(self.speed)
                self.px.set_dir_servo_angle(steering_angle)
                
                # Visualizzazione
                # Overlay della carreggiata
                overlay = self.create_lane_overlay(bird_eye, left_line, right_line)
                result = cv2.addWeighted(bird_eye, 1.0, overlay, 0.5, 0)
                
                # Disegna le linee rilevate
                if left_line:
                    cv2.line(result, left_line[0], left_line[1], (255, 0, 0), 3)
                if right_line:
                    cv2.line(result, right_line[0], right_line[1], (0, 0, 255), 3)
                
                # Combina le visualizzazioni
                display_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
                
                # Frame originale (ridimensionato)
                original_resized = cv2.resize(frame, (width, height))
                display_frame[:, :width] = original_resized
                
                # Risultati processing (ridimensionato)
                result_resized = cv2.resize(result, (width, height))
                display_frame[:, width:] = result_resized
                
                # Informazioni di debug
                cv2.putText(display_frame, f"Steering: {steering_angle:.1f}°", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Speed: {self.speed}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Mostra il frame
                cv2.imshow('Line Following', display_frame)
                cv2.imshow('Hough Edges', edges)
                cv2.imshow('HSV Mask', combined_mask)
                
                # Controlli da tastiera
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Salva screenshot
                    cv2.imwrite('screenshot.jpg', display_frame)
                    print("Screenshot salvato!")
                    
        except KeyboardInterrupt:
            print("Interruzione da tastiera")
        finally:
            # Cleanup
            self.px.stop()
            self.picam2.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    line_follower = LineFollower()
    line_follower.run()
