import cv2
import numpy as np
from picamera2 import Picamera2
import time
from picarx import Picarx

class LaneDetector:
    def __init__(self):
        # Inizializza PiCar-X
        self.px = Picarx()
        
        # Inizializza la camera
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(
            main={"size": (640, 480)}
        )
        self.picam2.configure(camera_config)
        self.picam2.start()
        
        # Parametri di velocità e sterzo
        self.speed = 1
        self.max_angle = 45
        self.previous_steering_angle = 0  # Memorizza l'angolo precedente
        
        # Parametri per bird's eye view
        self.dst_height = 200
        self.dst_width = 300
        
        # Parametri per Hough Transform
        self.hough_threshold = 50
        self.min_line_length = 50
        self.max_line_gap = 20
        
        # Parametri per validazione carreggiata
        self.min_lane_width = 100  # Larghezza minima carreggiata (px)
        self.max_lane_width = 300  # Larghezza massima carreggiata (px)
        self.min_distance_from_center = 50  # Distanza minima dal centro (px)
        
        # Buffer per stabilizzare le linee
        self.left_line_buffer = []
        self.right_line_buffer = []
        self.buffer_size = 5
        
    def create_bird_eye_transform(self, width, height):
        """Crea la matrice di trasformazione per bird's eye view"""
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
    
    def detect_lane_colors(self, frame):
        """Rileva linee bianche e gialle usando filtro HSV"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Maschere per colori bianchi
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Maschere per colori gialli
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combina le maschere
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        return combined_mask, white_mask, yellow_mask
    
    def apply_gaussian_blur(self, image, kernel_size=5):
        """Applica filtro gaussiano"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def detect_lines_hough(self, edges):
        """Rileva linee usando Hough Transform"""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        return lines
    
    def classify_lines(self, lines, width):
        """Classifica le linee in sinistra e destra"""
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                
                # Filtra per pendenza ragionevole
                if abs(slope) < 0.3:
                    continue
                
                # Classifica in base alla posizione e pendenza
                if slope < 0 and x1 < width // 2 and x2 < width // 2:
                    left_lines.append(line[0])
                elif slope > 0 and x1 > width // 2 and x2 > width // 2:
                    right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def average_lines(self, lines):
        """Calcola la linea media da un set di linee"""
        if not lines:
            return None
        
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        if len(x_coords) < 2:
            return None
        
        # Calcola regressione lineare
        poly = np.polyfit(y_coords, x_coords, 1)
        
        # Calcola punti della linea
        y1 = self.dst_height
        y2 = 0
        x1 = int(poly[0] * y1 + poly[1])
        x2 = int(poly[0] * y2 + poly[1])
        
        return [x1, y1, x2, y2]
    
    def create_lane_triangle(self, left_line, right_line, width, height):
        """Crea triangolo della carreggiata gestendo linee mancanti"""
        lane_points = []
        
        if left_line is not None:
            x1, y1, x2, y2 = left_line
            lane_points.extend([[x1, y1], [x2, y2]])
        else:
            # Assume linea sinistra in basso a sinistra
            lane_points.extend([[0, height], [width//4, 0]])
        
        if right_line is not None:
            x1, y1, x2, y2 = right_line
            lane_points.extend([[x2, y2], [x1, y1]])
        else:
            # Assume linea destra in basso a destra
            lane_points.extend([[3*width//4, 0], [width, height]])
        
        return np.array(lane_points, dtype=np.int32)
    
    def calculate_lane_width(self, left_line, right_line):
        """Calcola la larghezza della carreggiata"""
        if left_line is None or right_line is None:
            return None
        
        # Calcola la distanza media tra le linee
        left_center_x = (left_line[0] + left_line[2]) // 2
        right_center_x = (right_line[0] + right_line[2]) // 2
        
        lane_width = abs(right_center_x - left_center_x)
        return lane_width
    
    def is_line_too_close_to_center(self, line, center_x):
        """Verifica se una linea è troppo vicina al centro"""
        if line is None:
            return False
        
        line_center_x = (line[0] + line[2]) // 2
        distance_from_center = abs(line_center_x - center_x)
        
        return distance_from_center < self.min_distance_from_center
    
    def validate_lane_detection(self, left_line, right_line, width):
        """Valida il rilevamento della carreggiata"""
        center_x = width // 2
        
        # Caso 1: Entrambe le linee presenti
        if left_line is not None and right_line is not None:
            lane_width = self.calculate_lane_width(left_line, right_line)
            
            # Verifica larghezza carreggiata
            if lane_width < self.min_lane_width:
                return False, f"Carreggiata troppo stretta: {lane_width}px"
            
            # Verifica se le linee sono troppo vicine al centro
            if (self.is_line_too_close_to_center(left_line, center_x) or 
                self.is_line_too_close_to_center(right_line, center_x)):
                return False, "Linee troppo vicine al centro"
        
        # Caso 2: Solo una linea presente
        elif left_line is not None:
            if self.is_line_too_close_to_center(left_line, center_x):
                return False, "Linea sinistra troppo vicina al centro"
        
        elif right_line is not None:
            if self.is_line_too_close_to_center(right_line, center_x):
                return False, "Linea destra troppo vicina al centro"
        
        return True, "Rilevamento valido"
        """Calcola l'angolo di sterzo basato sulle linee rilevate"""
        center_x = width // 2
        
        if left_line is not None and right_line is not None:
            # Entrambe le linee presenti
            left_x = (left_line[0] + left_line[2]) // 2
            right_x = (right_line[0] + right_line[2]) // 2
            lane_center = (left_x + right_x) // 2
        elif left_line is not None:
            # Solo linea sinistra
            left_x = (left_line[0] + left_line[2]) // 2
            lane_center = left_x + width // 4
        elif right_line is not None:
            # Solo linea destra
            right_x = (right_line[0] + right_line[2]) // 2
            lane_center = right_x - width // 4
        else:
            # Nessuna linea rilevata
            lane_center = center_x
        
        # Calcola offset dal centro
        offset = lane_center - center_x
        
        # Converte offset in angolo di sterzo
        steering_angle = np.arctan(offset / (width // 2)) * 180 / np.pi
        
        # Limita l'angolo massimo
        steering_angle = np.clip(steering_angle, -self.max_angle, self.max_angle)
        
        return steering_angle
    
    def draw_lane_overlay(self, frame, lane_points, alpha=0.5):
        """Disegna overlay della carreggiata con trasparenza"""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [lane_points], (0, 255, 0))
        return cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
    
    def process_frame(self, frame):
        """Processa un singolo frame"""
        height, width = frame.shape[:2]
        
        # Crea bird's eye view
        M = self.create_bird_eye_transform(width, height)
        bird_eye = cv2.warpPerspective(frame, M, (self.dst_width, self.dst_height))
        
        # Rileva colori delle linee
        lane_mask, white_mask, yellow_mask = self.detect_lane_colors(bird_eye)
        
        # Applica filtro gaussiano
        blurred = self.apply_gaussian_blur(lane_mask)
        
        # Rileva bordi
        edges = cv2.Canny(blurred, 50, 150)
        
        # Rileva linee con Hough
        lines = self.detect_lines_hough(edges)
        
        # Classifica linee
        left_lines, right_lines = self.classify_lines(lines, self.dst_width)
        
        # Calcola linee medie
        left_line = self.average_lines(left_lines)
        right_line = self.average_lines(right_lines)
        
        # Crea triangolo della carreggiata
        lane_points = self.create_lane_triangle(left_line, right_line, 
                                               self.dst_width, self.dst_height)
        
        # Calcola angolo di sterzo
        steering_angle, status_message = self.calculate_steering_angle(left_line, right_line, 
                                                                      self.dst_width)
        
        # Crea visualizzazioni
        result_frame = bird_eye.copy()
        
        # Disegna linee rilevate
        if left_line is not None:
            cv2.line(result_frame, (left_line[0], left_line[1]), 
                    (left_line[2], left_line[3]), (255, 0, 0), 3)
        if right_line is not None:
            cv2.line(result_frame, (right_line[0], right_line[1]), 
                    (right_line[2], right_line[3]), (0, 0, 255), 3)
        
        # Disegna overlay della carreggiata
        result_frame = self.draw_lane_overlay(result_frame, lane_points, 0.5)
        
        # Crea immagini per la visualizzazione
        hough_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        filter_vis = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
        
        return result_frame, hough_vis, filter_vis, steering_angle, status_message
    
    def run(self):
        """Loop principale del sistema"""
        print("Avvio sistema di rilevamento corsie...")
        
        try:
            while True:
                # Cattura frame dalla camera
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Processa il frame
                result, hough_vis, filter_vis, steering_angle, status_message = self.process_frame(frame)
                
                # Controlla il veicolo
                self.px.set_dir_servo_angle(int(steering_angle))
                self.px.forward(self.speed)
                
                # Crea visualizzazione combinata
                top_row = np.hstack([filter_vis, hough_vis])
                bottom_row = np.hstack([result, cv2.resize(frame, (300, 200))])
                combined = np.vstack([top_row, bottom_row])
                
                # Aggiungi testo informativo
                cv2.putText(combined, f"Steering: {steering_angle:.1f}°", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, f"Speed: {self.speed}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, status_message, 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Mostra il risultato
                cv2.imshow('Lane Detection System', combined)
                
                # Controlli da tastiera
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Stop
                    self.px.stop()
                elif key == ord('r'):
                    # Resume
                    pass
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Interruzione da tastiera...")
        
        finally:
            # Cleanup
            self.px.stop()
            self.picam2.stop()
            cv2.destroyAllWindows()
            print("Sistema arrestato.")

def main():
    detector = LaneDetector()
    detector.run()

if __name__ == "__main__":
    main()