import cv2
import numpy as np
import math
import time
from picarx import Picarx
from picamera2 import Picamera2

# Inizializza PiCar-X e camera
px = Picarx()
picam2 = Picamera2()

# Configura la camera
camera_config = picam2.create_preview_configuration(
    main={"format": 'RGB888', "size": (640, 480)}
)
picam2.configure(camera_config)
picam2.start()

# Parametri di controllo
BASE_SPEED = 1
MAX_STEERING_ANGLE = 45
MIN_STEERING_ANGLE = -45
MIN_CARRIAGEWAY_WIDTH = 200  # Soglia minima per la larghezza carreggiata

# Variabili globali
previous_steering_angle = 0
center_x_bird_eye = 150  # Centro dell'immagine bird-eye (300px / 2)

def get_average_line_x(lanes_segments):
    """
    Calcola la coordinata x media dei segmenti di linea forniti.
    """
    if not lanes_segments:
        return None

    all_x_coords = []
    for x1, y1, x2, y2, *_ in lanes_segments:
        all_x_coords.append(x1)
        all_x_coords.append(x2)
    
    if not all_x_coords:
        return None
        
    avg_x = np.mean(all_x_coords)
    return int(avg_x)

def calculate_carriageway_width_pixels(left_segments, right_segments):
    """
    Calcola la larghezza della carreggiata in pixel nella vista bird-eye.
    """
    avg_left_x = get_average_line_x(left_segments)
    avg_right_x = get_average_line_x(right_segments)
    
    if avg_left_x is not None and avg_right_x is not None:
        width_pixels = abs(avg_right_x - avg_left_x)
        return width_pixels, avg_left_x, avg_right_x
    
    return None, avg_left_x, avg_right_x

def get_average_slope(lane_segments):
    """
    Calcola la pendenza media dei segmenti di corsia forniti.
    """
    if not lane_segments:
        return None
    
    slopes = []
    for x1, y1, x2, y2, *_ in lane_segments:
        if x1 != x2:  # Evita divisione per zero
            slope = (float(y2) - float(y1)) / (float(x2) - float(x1))
            slopes.append(slope)
        else:
            # Linea verticale, pendenza infinita
            slopes.append(float('inf') if y2 > y1 else float('-inf'))
    
    if not slopes:
        return None
    
    # Filtra i valori infiniti per il calcolo della media
    finite_slopes = [s for s in slopes if math.isfinite(s)]
    if not finite_slopes:
        return None
    
    return np.mean(finite_slopes)

def calculate_steering_angle_from_slopes(left_segments, right_segments, carriageway_width):
    """
    Calcola l'angolo di sterzata basato sulla pendenza delle linee delle corsie.
    """
    global previous_steering_angle
    
    # Se la carreggiata è troppo piccola, mantieni l'angolo precedente
    if carriageway_width is not None and carriageway_width < MIN_CARRIAGEWAY_WIDTH:
        print(f"Carreggiata troppo piccola ({carriageway_width}px), mantengo angolo precedente: {previous_steering_angle}")
        return previous_steering_angle
    
    # Calcola le pendenze medie delle corsie
    left_slope = get_average_slope(left_segments)
    right_slope = get_average_slope(right_segments)
    
    # Calcola l'angolo di sterzata basato sulle pendenze
    steering_angle = 0
    
    if left_slope is not None and right_slope is not None:
        # Entrambe le corsie disponibili - usa la media degli angoli
        left_angle = math.degrees(math.atan(left_slope)) if math.isfinite(left_slope) else 0
        right_angle = math.degrees(math.atan(right_slope)) if math.isfinite(right_slope) else 0
        
        # Media degli angoli delle corsie
        avg_lane_angle = (left_angle + right_angle) / 2
        
        # Converti l'angolo della corsia in angolo di sterzata
        # Se le corsie puntano a sinistra (angolo negativo), sterza a destra e viceversa
        steering_angle = -avg_lane_angle * 0.8  # Fattore di guadagno
        
    elif left_slope is not None:
        # Solo corsia sinistra disponibile
        left_angle = math.degrees(math.atan(left_slope)) if math.isfinite(left_slope) else 0
        steering_angle = -left_angle * 1.2  # Guadagno maggiore per una sola corsia
        
    elif right_slope is not None:
        # Solo corsia destra disponibile
        right_angle = math.degrees(math.atan(right_slope)) if math.isfinite(right_slope) else 0
        steering_angle = -right_angle * 1.2  # Guadagno maggiore per una sola corsia
        
    else:
        # Nessuna corsia rilevata, mantieni l'angolo precedente
        return previous_steering_angle
    
    # Limita l'angolo nei range consentiti
    steering_angle = max(MIN_STEERING_ANGLE, min(MAX_STEERING_ANGLE, steering_angle))
    
    return steering_angle

def process_frame(frame):
    """
    Processa un singolo frame per rilevare le corsie.
    """
    if frame is None:
        return None, None, None, None
    
    h, w = frame.shape[:2]
    
    # ROI
    roi_top_ratio = 0.6
    roi_bottom_ratio = 1.0
    roi_top = int(h * roi_top_ratio)
    roi_bottom = int(h * roi_bottom_ratio)
    
    # Trasformazione prospettica
    src = np.float32([[0, roi_top], [w, roi_top], [w, roi_bottom], [0, roi_bottom]])
    dst_w, dst_h = 300, 200
    dst = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    bird_eye = cv2.warpPerspective(frame, matrix, (dst_w, dst_h))
    
    # Pre-elaborazione
    gaussian = cv2.GaussianBlur(bird_eye, (5, 5), 0)
    hls = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HLS)
    
    # Maschere per giallo e bianco
    yellow_lower = np.array([15, 30, 115])
    yellow_upper = np.array([35, 200, 255])
    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    
    white_lower = np.array([0, 180, 0])
    white_upper = np.array([180, 255, 255])
    white_mask = cv2.inRange(hls, white_lower, white_upper)
    
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    kernel_close = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Rilevamento bordi e linee
    edges = cv2.Canny(mask, 50, 150)
    detected_lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=30,
        minLineLength=15, maxLineGap=20
    )
    
    bird_eye_with_lines = bird_eye.copy()
    left_lane_segments = []
    right_lane_segments = []
    
    min_abs_slope_threshold = 0.3
    horizontal_slope_threshold = 0.1
    
    if detected_lines is not None:
        for line_data in detected_lines:
            x1, y1, x2, y2 = line_data[0]
            line_color = (0,0,0)
            added_to_lane = False
            
            if x1 == x2:
                if x1 < dst_w / 2:
                    line_color = (0, 0, 255)
                    left_lane_segments.append((x1, y1, x2, y2))
                    added_to_lane = True
                else:
                    line_color = (0, 255, 0)
                    right_lane_segments.append((x1, y1, x2, y2))
                    added_to_lane = True
            else:
                slope = (float(y2) - float(y1)) / (float(x2) - float(x1))
                center_x_segment = (x1 + x2) / 2
                
                if abs(slope) < horizontal_slope_threshold:
                    if center_x_segment < dst_w / 2:
                        line_color = (0, 0, 255)
                        left_lane_segments.append((x1, y1, x2, y2))
                        added_to_lane = True
                    else:
                        line_color = (0, 255, 0)
                        right_lane_segments.append((x1, y1, x2, y2))
                        added_to_lane = True
                elif slope < -min_abs_slope_threshold:
                    if center_x_segment < dst_w * 0.55:
                        line_color = (0, 0, 255)
                        left_lane_segments.append((x1, y1, x2, y2))
                        added_to_lane = True
                elif slope > min_abs_slope_threshold:
                    if center_x_segment > dst_w * 0.45:
                        line_color = (0, 255, 0)
                        right_lane_segments.append((x1, y1, x2, y2))
                        added_to_lane = True
            
            if added_to_lane:
                cv2.line(bird_eye_with_lines, (x1, y1), (x2, y2), line_color, 2)
    
    # Calcola larghezza carreggiata
    carriageway_width_pixels, avg_left_x, avg_right_x = calculate_carriageway_width_pixels(left_lane_segments, right_lane_segments)
    
    return bird_eye_with_lines, carriageway_width_pixels, avg_left_x, avg_right_x, left_lane_segments, right_lane_segments

def main():
    global previous_steering_angle
    
    print("Avvio sistema di controllo PiCar-X...")
    print("Premi Ctrl+C per fermare")
    
    try:
        # Avvia il movimento
        px.forward(BASE_SPEED)
        
        while True:
            # Cattura frame dalla camera
            frame = picam2.capture_array()
            
            if frame is None:
                continue
            
            # Processa il frame
            processed_result = process_frame(frame)
            if processed_result is None:
                continue
                
            bird_eye_with_lines, carriageway_width, avg_left_x, avg_right_x, left_segments, right_segments = processed_result
            
            # Calcola l'angolo di sterzata basato sulle pendenze delle linee
            steering_angle = calculate_steering_angle_from_slopes(left_segments, right_segments, carriageway_width)
            
            # Applica l'angolo di sterzata alla PiCar-X
            px.set_dir_servo_angle(steering_angle)
            
            # Aggiorna l'angolo precedente
            previous_steering_angle = steering_angle
            
            # Debug info con informazioni sulle pendenze
            left_slope = get_average_slope(left_segments) if 'left_segments' in locals() else None
            right_slope = get_average_slope(right_segments) if 'right_segments' in locals() else None
            
            if carriageway_width is not None:
                status = f"Larghezza: {carriageway_width}px, Angolo: {steering_angle:.1f}°"
                if left_slope is not None:
                    status += f", Sx: {left_slope:.2f}"
                if right_slope is not None:
                    status += f", Dx: {right_slope:.2f}"
                if carriageway_width < MIN_CARRIAGEWAY_WIDTH:
                    status += " (MANTENUTO)"
            else:
                status = f"Larghezza: N/A, Angolo: {steering_angle:.1f}° (MANTENUTO)"
            
            print(f"\r{status}", end="", flush=True)
            
            # Visualizza l'immagine processata (opzionale)
            if bird_eye_with_lines is not None:
                # Aggiungi info sull'immagine
                cv2.putText(bird_eye_with_lines, f"Steering: {steering_angle:.1f}°",
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if carriageway_width is not None:
                    cv2.putText(bird_eye_with_lines, f"Width: {carriageway_width}px",
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Aggiungi info sulle pendenze all'immagine
                if left_slope is not None:
                    cv2.putText(bird_eye_with_lines, f"Left slope: {left_slope:.2f}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                if right_slope is not None:
                    cv2.putText(bird_eye_with_lines, f"Right slope: {right_slope:.2f}",
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                cv2.imshow('Lane Detection', bird_eye_with_lines)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            time.sleep(0.1)  # Breve pausa per stabilità
            
    except KeyboardInterrupt:
        print("\nInterruzione rilevata, fermata della PiCar-X...")
    
    finally:
        # Ferma la macchina e resetta lo sterzo
        px.stop()
        px.set_dir_servo_angle(0)
        picam2.stop()
        cv2.destroyAllWindows()
        print("Sistema fermato correttamente.")

if __name__ == '__main__':
    main()