import cv2
import numpy as np
import math
import time
from threading import Thread, Lock
import signal
import sys

# Import per PiCamera2 e PicarX
try:
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA_AVAILABLE = True
    print("PiCamera2 importata con successo")
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Attenzione: PiCamera2 non disponibile, usando video file")

try:
    from picarx import Picarx
    PICARX_AVAILABLE = True
    print("PicarX importata con successo")
except ImportError:
    PICARX_AVAILABLE = False
    print("Attenzione: PicarX non disponibile, solo simulazione")

# Variabili globali
lane_width = 300  # Larghezza carreggiata in pixel
previous_angle = 0  # Angolo di sterzata precedente
frame_center = 150  # Centro fisso dell'immagine (metà di 300px)
dst_w, dst_h = 300, 200  # Dimensioni immagine trasformata

# Parametri per il controllo dell'angolo
MIN_ANGLE = -45.0  # Angolo minimo in gradi
MAX_ANGLE = 45.0   # Angolo massimo in gradi
ANGLE_SMOOTHING = 0.7  # Fattore di smoothing per l'angolo (0-1)

# Parametro per la posizione del centro dinamico (più piccolo = più in alto)
CENTER_Y_RATIO = 0.3  # 0.3 significa al 30% dell'altezza (più in alto rispetto a 0.5)

# Parametri per il controllo del robot PicarX
SPEED_BASE = 30  # Velocità base del robot (0-100)
SPEED_TURN = 20  # Velocità ridotta durante le curve
STEERING_GAIN = 2.0  # Moltiplicatore per l'angolo di sterzo
AUTONOMOUS_MODE = False  # Modalità autonoma on/off

# Thread control
running = True
frame_lock = Lock()
current_frame = None
steering_command = 0
speed_command = 0

class PiCarXController:
    """Classe per controllare il robot PicarX"""
    def __init__(self):
        self.px = None
        self.enabled = False
        self.last_steering = 0
        self.last_speed = 0
        
        if PICARX_AVAILABLE:
            try:
                self.px = Picarx()
                self.enabled = True
                print("PicarX inizializzato con successo")
                # Reset iniziale
                self.px.set_dir_servo_angle(0)
                self.px.stop()
            except Exception as e:
                print(f"Errore nell'inizializzazione di PicarX: {e}")
                self.enabled = False
        
    def update_control(self, steering_angle, speed):
        """Aggiorna i comandi di controllo del robot"""
        if not self.enabled or not AUTONOMOUS_MODE:
            return
            
        try:
            # Converti l'angolo di sterzo per PicarX
            # PicarX usa range tipicamente -30 a +30 gradi
            picarx_steering = np.clip(steering_angle * STEERING_GAIN, -30, 30)
            
            # Riduci velocità durante le curve
            abs_steering = abs(steering_angle)
            if abs_steering > 20:
                current_speed = SPEED_TURN
            else:
                current_speed = SPEED_BASE
            
            # Applica i comandi solo se sono cambiati significativamente
            if abs(picarx_steering - self.last_steering) > 2:
                self.px.set_dir_servo_angle(int(picarx_steering))
                self.last_steering = picarx_steering
                
            if abs(current_speed - self.last_speed) > 5:
                if current_speed > 0:
                    self.px.forward(current_speed)
                else:
                    self.px.stop()
                self.last_speed = current_speed
                
        except Exception as e:
            print(f"Errore nel controllo PicarX: {e}")
    
    def emergency_stop(self):
        """Ferma immediatamente il robot"""
        if self.enabled:
            try:
                self.px.stop()
                self.px.set_dir_servo_angle(0)
                print("STOP di emergenza eseguito")
            except Exception as e:
                print(f"Errore nello stop di emergenza: {e}")
    
    def cleanup(self):
        """Pulizia finale"""
        if self.enabled:
            try:
                self.px.stop()
                self.px.set_dir_servo_angle(0)
                print("PicarX fermato e resettato")
            except Exception as e:
                print(f"Errore nel cleanup PicarX: {e}")

class CameraHandler:
    """Classe per gestire la camera PiCamera2"""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.picam2 = None
        self.enabled = False
        self.resolution = resolution
        self.framerate = framerate
        
        if PICAMERA_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                
                # Configurazione camera
                config = self.picam2.create_preview_configuration(
                    main={"size": resolution, "format": "RGB888"}
                )
                self.picam2.configure(config)
                
                # Impostazioni ottimali per lane detection
                self.picam2.set_controls({
                    "AeEnable": True,
                    "AwbEnable": True,
                    "Brightness": 0.0,
                    "Contrast": 1.0,
                    "Saturation": 1.0,
                    "Sharpness": 1.0,
                    "ExposureTime": 10000,  # Esposizione fissa per ridurre flickering
                })
                
                self.picam2.start()
                time.sleep(2)  # Tempo per stabilizzare
                self.enabled = True
                print(f"PiCamera2 inizializzata: {resolution} @ {framerate}fps")
                
            except Exception as e:
                print(f"Errore nell'inizializzazione PiCamera2: {e}")
                self.enabled = False
    
    def capture_frame(self):
        """Cattura un frame dalla camera"""
        if not self.enabled:
            return None
            
        try:
            # Cattura frame come array numpy
            frame = self.picam2.capture_array()
            # Converti da RGB a BGR per OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame_bgr
        except Exception as e:
            print(f"Errore nella cattura frame: {e}")
            return None
    
    def cleanup(self):
        """Pulizia camera"""
        if self.enabled and self.picam2:
            try:
                self.picam2.stop()
                print("PiCamera2 fermata")
            except Exception as e:
                print(f"Errore nel cleanup camera: {e}")

def signal_handler(sig, frame):
    """Gestisce l'interruzione del programma"""
    global running
    print("\nInterruzione ricevuta, fermando il sistema...")
    running = False

def bird_eye_transform(frame):
    """Applica la trasformazione a occhio d'uccello"""
    h, w = frame.shape[:2]
    
    # Definisce la regione di interesse (ROI) nella parte inferiore del frame
    roi_top, roi_bottom = int(h * 0.6), h
    
    # Punti sorgente per la trasformazione prospettica
    src = np.float32([
        [0, roi_top], 
        [w, roi_top], 
        [w, roi_bottom], 
        [0, roi_bottom]
    ])
    
    # Punti di destinazione
    dst = np.float32([
        [0, 0], 
        [dst_w, 0], 
        [dst_w, dst_h], 
        [0, dst_h]
    ])
    
    # Calcola e applica la trasformazione
    matrix = cv2.getPerspectiveTransform(src, dst)
    bird_eye = cv2.warpPerspective(frame, matrix, (dst_w, dst_h))
    
    return bird_eye

def detect_lane_lines(bird_eye):
    """Rileva le linee delle corsie"""
    # Conversione in HSV
    hsv = cv2.cvtColor(bird_eye, cv2.COLOR_BGR2HSV)
    
    # Maschere per colori bianco e giallo
    mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
    
    # Combina le maschere
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    # Applicazione filtri
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Rilevamento linee con HoughLinesP
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 
        threshold=50, 
        minLineLength=20, 
        maxLineGap=30
    )
    
    return lines, edges, mask

def separate_lines(lines):
    """Separa le linee in sinistra e destra"""
    if lines is None:
        return None, None
    
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calcola la pendenza
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            
            # Classifica la linea in base alla pendenza e posizione
            if slope < 0 and x1 < frame_center and x2 < frame_center:
                left_lines.append(line[0])
            elif slope > 0 and x1 > frame_center and x2 > frame_center:
                right_lines.append(line[0])
    
    return left_lines, right_lines

def average_lines(lines):
    """Calcola la linea media da un gruppo di linee"""
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
    
    # Regressione lineare
    poly = np.polyfit(y_coords, x_coords, 1)
    
    # Calcola i punti della linea
    y1 = dst_h
    y2 = int(dst_h * 0.6)
    x1 = int(poly[0] * y1 + poly[1])
    x2 = int(poly[0] * y2 + poly[1])
    
    return [x1, y1, x2, y2]

def get_line_x_at_y(line, target_y):
    """Calcola la coordinata x di una linea ad una specifica coordinata y"""
    if line is None:
        return None
    
    x1, y1, x2, y2 = line
    
    # Evita divisione per zero
    if y2 - y1 == 0:
        return x1
    
    # Interpolazione lineare per trovare x alla coordinata y target
    t = (target_y - y1) / (y2 - y1)
    x = x1 + t * (x2 - x1)
    
    return x

def validate_lines(left_line, right_line):
    """Valida le linee rilevate"""
    valid_left = False
    valid_right = False
    
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        # Controlla se la linea sinistra è effettivamente a sinistra del centro
        if max(x1, x2) < frame_center:
            line_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            if line_length >= 20:  # Lunghezza minima
                valid_left = True
    
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        # Controlla se la linea destra è effettivamente a destra del centro
        if min(x1, x2) > frame_center:
            line_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            if line_length >= 20:  # Lunghezza minima
                valid_right = True
    
    return valid_left, valid_right

def calculate_lane_center_and_angle(left_line, right_line, valid_left, valid_right):
    """Calcola il centro della carreggiata e l'angolo di sterzata"""
    global previous_angle
    
    lane_center = None
    steering_angle = previous_angle
    
    # Calcola la coordinata y per il centro dinamico (più in alto)
    center_y = int(dst_h * CENTER_Y_RATIO)
    
    # Se abbiamo entrambe le linee valide
    if valid_left and valid_right:
        left_x = get_line_x_at_y(left_line, center_y)
        right_x = get_line_x_at_y(right_line, center_y)
        
        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2
    
    # Se abbiamo solo la linea sinistra
    elif valid_left and not valid_right:
        left_x = get_line_x_at_y(left_line, center_y)
        if left_x is not None:
            lane_center = left_x + lane_width / 2
    
    # Se abbiamo solo la linea destra
    elif valid_right and not valid_left:
        right_x = get_line_x_at_y(right_line, center_y)
        if right_x is not None:
            lane_center = right_x - lane_width / 2
    
    # Calcola l'angolo di sterzata se abbiamo un centro valido
    if lane_center is not None:
        # Distanza dal centro dell'immagine
        center_offset = lane_center - frame_center
        
        # Calcola l'angolo basato sull'offset
        # Usa una funzione più aggressiva per raggiungere i limiti ±45°
        max_offset = dst_w / 2  # Massimo offset possibile
        
        # Normalizza l'offset tra -1 e 1
        normalized_offset = center_offset / max_offset
        
        # Applica una funzione per aumentare la sensibilità
        # Usa una funzione sigmoidea modificata per ottenere la gamma completa
        if abs(normalized_offset) > 0.1:  # Soglia per evitare jitter al centro
            # Funzione esponenziale per aumentare la sensibilità
            sign = 1 if normalized_offset > 0 else -1
            abs_offset = abs(normalized_offset)
            
            # Mappa l'offset con una curva più aggressiva
            if abs_offset < 0.5:
                mapped_offset = abs_offset * 1.5  # Incremento lineare per piccoli offset
            else:
                mapped_offset = 0.75 + (abs_offset - 0.5) * 1.5  # Incremento accelerato
            
            mapped_offset = min(mapped_offset, 1.0)  # Limita a 1
            steering_angle = sign * mapped_offset * MAX_ANGLE
        else:
            steering_angle = 0  # Centro morto per stabilità
        
        # Applica limiti rigidi
        steering_angle = np.clip(steering_angle, MIN_ANGLE, MAX_ANGLE)
        
        # Applica smoothing per ridurre le oscillazioni
        steering_angle = (ANGLE_SMOOTHING * previous_angle + 
                         (1 - ANGLE_SMOOTHING) * steering_angle)
        
        # Aggiorna l'angolo precedente
        previous_angle = steering_angle
    
    return lane_center, steering_angle, center_y

def draw_lanes(img, left_line, right_line, valid_left, valid_right, lane_center, center_y):
    """Disegna le linee delle corsie sull'immagine"""
    line_img = img.copy()
    
    # Disegna la linea sinistra
    if valid_left and left_line is not None:
        cv2.line(line_img, (left_line[0], left_line[1]), 
                (left_line[2], left_line[3]), (0, 255, 0), 3)
        cv2.putText(line_img, "L", (left_line[0]-20, left_line[1]+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Disegna la linea destra
    if valid_right and right_line is not None:
        cv2.line(line_img, (right_line[0], right_line[1]), 
                (right_line[2], right_line[3]), (0, 255, 0), 3)
        cv2.putText(line_img, "R", (right_line[0]+10, right_line[1]+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Disegna il centro della carreggiata (ora più in alto)
    if lane_center is not None:
        cv2.circle(line_img, (int(lane_center), center_y), 8, (255, 0, 255), -1)
        cv2.putText(line_img, "CENTER", (int(lane_center)-30, center_y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Disegna una linea orizzontale per mostrare l'altezza del centro dinamico
    cv2.line(line_img, (0, center_y), (dst_w, center_y), (255, 0, 255), 1)
    
    # Disegna il centro fisso dell'immagine
    cv2.line(line_img, (frame_center, 0), (frame_center, dst_h), 
            (0, 0, 255), 2)
    cv2.putText(line_img, "IMG CENTER", (frame_center-45, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return line_img

def draw_steering_indicator(img, steering_angle):
    """Disegna un indicatore visuale dell'angolo di sterzata"""
    # Posizione dell'indicatore
    center_x, center_y = dst_w - 50, 50
    radius = 30
    
    # Disegna il cerchio di base
    cv2.circle(img, (center_x, center_y), radius, (100, 100, 100), 2)
    
    # Calcola la posizione della freccia basata sull'angolo
    angle_rad = math.radians(steering_angle)
    end_x = int(center_x + radius * 0.8 * math.sin(angle_rad))
    end_y = int(center_y - radius * 0.8 * math.cos(angle_rad))
    
    # Colore basato sull'intensità dell'angolo
    intensity = abs(steering_angle) / MAX_ANGLE
    if intensity < 0.3:
        color = (0, 255, 0)  # Verde per angoli piccoli
    elif intensity < 0.7:
        color = (0, 255, 255)  # Giallo per angoli medi
    else:
        color = (0, 0, 255)  # Rosso per angoli grandi
    
    # Disegna la freccia
    cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), color, 3)
    
    # Aggiungi testo con l'angolo
    cv2.putText(img, f"{steering_angle:.1f}°", 
               (center_x - 25, center_y + radius + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def add_info_overlay(img, steering_angle, lane_center, valid_left, valid_right, center_y):
    """Aggiunge informazioni di testo sull'immagine"""
    global AUTONOMOUS_MODE, SPEED_BASE
    
    info_img = img.copy()
    
    # Informazioni di stato
    y_offset = 30
    # Colore per l'angolo basato sui limiti
    angle_color = (0, 255, 0) if -15 <= steering_angle <= 15 else (0, 255, 255) if -30 <= steering_angle <= 30 else (0, 0, 255)
    cv2.putText(info_img, f"Steering: {steering_angle:.1f}°", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, angle_color, 2)
    
    y_offset += 25
    # Stato modalità autonoma
    mode_color = (0, 255, 0) if AUTONOMOUS_MODE else (0, 0, 255)
    mode_text = "AUTO ON" if AUTONOMOUS_MODE else "AUTO OFF"
    cv2.putText(info_img, f"Mode: {mode_text}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    
    y_offset += 25
    left_status = "OK" if valid_left else "NO"
    cv2.putText(info_img, f"Left Line: {left_status}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
               (0, 255, 0) if valid_left else (0, 0, 255), 2)
    
    y_offset += 25
    right_status = "OK" if valid_right else "NO"
    cv2.putText(info_img, f"Right Line: {right_status}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
               (0, 255, 0) if valid_right else (0, 0, 255), 2)
    
    y_offset += 25
    if lane_center is not None:
        offset = lane_center - frame_center
        cv2.putText(info_img, f"Offset: {offset:.1f}px", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset += 25
    # Velocità corrente
    cv2.putText(info_img, f"Speed: {SPEED_BASE}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Aggiungi indicatore di sterzata visuale
    info_img = draw_steering_indicator(info_img, steering_angle)
    
    return info_img

def process_frame(frame):
    """Processa un singolo frame"""
    # Trasformazione a occhio d'uccello
    bird_eye = bird_eye_transform(frame)
    
    # Rilevamento linee
    lines, edges, mask = detect_lane_lines(bird_eye)
    
    # Separazione linee sinistra/destra
    left_lines, right_lines = separate_lines(lines)
    
    # Calcolo linee medie
    left_line = average_lines(left_lines)
    right_line = average_lines(right_lines)
    
    # Validazione linee
    valid_left, valid_right = validate_lines(left_line, right_line)
    
    # Calcolo centro carreggiata e angolo sterzata
    lane_center, steering_angle, center_y = calculate_lane_center_and_angle(
        left_line, right_line, valid_left, valid_right)
    
    # Creazione immagini di output
    lane_img = draw_lanes(bird_eye, left_line, right_line, 
                         valid_left, valid_right, lane_center, center_y)
    
    info_img = add_info_overlay(lane_img, steering_angle, lane_center, 
                               valid_left, valid_right, center_y)
    
    return bird_eye, info_img, edges, steering_angle

def camera_thread(camera_handler):
    """Thread per cattura continua della camera"""
    global current_frame, running
    
    while running:
        frame = camera_handler.capture_frame()
        if frame is not None:
            with frame_lock:
                current_frame = frame.copy()
        time.sleep(0.03)  # ~30 FPS

def main():
    """Funzione principale"""
    global lane_width, ANGLE_SMOOTHING, CENTER_Y_RATIO, AUTONOMOUS_MODE, SPEED_BASE
    global running, current_frame
    
    # Gestione segnali
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=== Sistema di Rilevamento Corsie con PicarX ===")
    print("Controlli:")
    print("- Spazio: Play/Pause video (solo modalità file)")
    print("- 'q': Esci")
    print("- 'r': Reset video (solo modalità file)")
    print("- '+/-': Aumenta/Diminuisci larghezza carreggiata")
    print("- 's': Aumenta smoothing angolo")
    print("- 'a': Diminuisci smoothing angolo")
    print("- 'u': Sposta centro più in alto")
    print("- 'd': Sposta centro più in basso")
    print("- 'o': Toggle modalità autonoma ON/OFF")
    print("- 'w': Aumenta velocità")
    print("- 'x': Diminuisci velocità")
    print("- 'e': STOP di emergenza")
    print(f"- Angolo limitato tra {MIN_ANGLE}° e {MAX_ANGLE}°")
    
    # Inizializza componenti
    camera_handler = None
    car_controller = None
    cap = None
    paused = False
    
    try:
        # Inizializza PicarX
        car_controller = PiCarXController()
        
        # Inizializza camera o video
        if PICAMERA_AVAILABLE:
            camera_handler = CameraHandler()
            if camera_handler.enabled:
                print("Usando PiCamera2 in tempo reale")
                # Avvia thread camera
                camera_thread_obj = Thread(target=camera_thread, args=(camera_handler,))
                camera_thread_obj.daemon = True
                camera_thread_obj.start()
            else:
                print("Fallback a video file")
                cap = cv2.VideoCapture("video.mp4")
        else:
            print("Usando video file")
            cap = cv2.VideoCapture("video.mp4")
            if not cap.isOpened():
                print("Errore: Impossibile aprire il video")
                return
        
        print(f"Centro dinamico al {CENTER_Y_RATIO:.1f} dell'altezza")
        print(f"Modalità autonoma: {'ON' if AUTONOMOUS_MODE else 'OFF'}")
        
        last_time = time.time()
        fps_counter = 0
        fps_display = 0
        
        while running:
            current_time = time.time()
            fps_counter += 1
            
            # Calcola FPS ogni secondo
            if current_time - last_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                last_time = current_time
            
            # Ottieni frame
            frame = None
            if camera_handler and camera_handler.enabled:
                with frame_lock:
                    if current_frame is not None:
                        frame = current_frame.copy()
            elif cap is not None and not paused:
                ret, frame = cap.read()
                if not ret:
                    if cap is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                        continue
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Processa il frame
            bird_eye, lane_result, edges, steering_angle = process_frame(frame)
            
            # Invia comandi al robot se in modalità autonoma
            if car_controller:
                car_controller.update_control(steering_angle, SPEED_BASE)
            
            # Prepara visualizzazione
            display_width, display_height = 350, 250
            
            frame_resized = cv2.resize(frame, (display_width, display_height))
            bird_eye_resized = cv2.resize(bird_eye, (display_width, display_height))
            lane_result_resized = cv2.resize(lane_result, (display_width, display_height))
            
            # Crea immagine degli edges a colori per la visualizzazione
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_resized = cv2.resize(edges_colored, (display_width, display_height))
            
            # Aggiungi informazioni FPS e stato camera
            cv2.putText(frame_resized, f"FPS: {fps_display}", 
                       (10, display_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cam_status = "PiCam" if (camera_handler and camera_handler.enabled) else "Video"
            cv2.putText(frame_resized, f"Source: {cam_status}", 
                       (10, display_height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Combina tutte le visualizzazioni
            top_row = np.hstack([frame_resized, bird_eye_resized])
            bottom_row = np.hstack([lane_result_resized, edges_resized])
            
            # Crea visualizzazione finale
            combined = np.vstack([top_row, bottom_row])
            
            # Mostra l'immagine
            cv2.imshow('Lane Detection System - PicarX', combined)
            
            # Gestione input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') and cap is not None:  # Spazio per pausa/play (solo video)
                paused = not paused
                print("Video", "in pausa" if paused else "in riproduzione")
            elif key == ord('r') and cap is not None:  # Reset video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False
                print("Video resettato")
            elif key == ord('+') or key == ord('='):  # Aumenta larghezza carreggiata
                lane_width = min(250, lane_width + 10)
                print(f"Larghezza carreggiata: {lane_width}px")
            elif key == ord('-'):  # Diminuisci larghezza carreggiata
                lane_width = max(100, lane_width - 10)
                print(f"Larghezza carreggiata: {lane_width}px")
            elif key == ord('s'):  # Aumenta smoothing
                ANGLE_SMOOTHING = min(0.9, ANGLE_SMOOTHING + 0.1)
                print(f"Smoothing angolo: {ANGLE_SMOOTHING:.1f}")
            elif key == ord('a'):  # Diminuisci smoothing
                ANGLE_SMOOTHING = max(0.1, ANGLE_SMOOTHING - 0.1)
                print(f"Smoothing angolo: {ANGLE_SMOOTHING:.1f}")
            elif key == ord('u'):  # Sposta centro più in alto
                CENTER_Y_RATIO = max(0.1, CENTER_Y_RATIO - 0.1)
                print(f"Centro dinamico spostato più in alto: {CENTER_Y_RATIO:.1f}")
            elif key == ord('d'):  # Sposta centro più in basso
                CENTER_Y_RATIO = min(0.9, CENTER_Y_RATIO + 0.1)
                print(f"Centro dinamico spostato più in basso: {CENTER_Y_RATIO:.1f}")
            elif key == ord('o'):  # Toggle modalità autonoma
                AUTONOMOUS_MODE = not AUTONOMOUS_MODE
                print(f"Modalità autonoma: {'ON' if AUTONOMOUS_MODE else 'OFF'}")
                if not AUTONOMOUS_MODE and car_controller:
                    car_controller.emergency_stop()
            elif key == ord('w'):  # Aumenta velocità
                SPEED_BASE = min(80, SPEED_BASE + 5)
                print(f"Velocità base: {SPEED_BASE}")
            elif key == ord('x'):  # Diminuisci velocità
                SPEED_BASE = max(10, SPEED_BASE - 5)
                print(f"Velocità base: {SPEED_BASE}")
            elif key == ord('e'):  # STOP di emergenza
                print("STOP DI EMERGENZA!")
                if car_controller:
                    car_controller.emergency_stop()
                AUTONOMOUS_MODE = False
            elif key == ord('h'):  # Help
                print("\n=== COMANDI DISPONIBILI ===")
                print("Controllo Robot:")
                print("  'o' - Toggle modalità autonoma ON/OFF")
                print("  'w' - Aumenta velocità (+5)")
                print("  'x' - Diminuisci velocità (-5)")
                print("  'e' - STOP di emergenza")
                print("\nParametri Lane Detection:")
                print("  '+/-' - Aumenta/Diminuisci larghezza carreggiata")
                print("  's/a' - Aumenta/Diminuisci smoothing angolo")
                print("  'u/d' - Sposta centro rilevamento su/giù")
                print("\nControllo Video (solo modalità file):")
                print("  'spazio' - Play/Pausa")
                print("  'r' - Reset video")
                print("\nAltro:")
                print("  'q' - Esci")
                print("  'h' - Mostra questo help")
                print("===============================\n")
    
    except KeyboardInterrupt:
        print("\nInterruzione da tastiera")
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
    finally:
        # Cleanup
        running = False
        print("Cleanup in corso...")
        
        if car_controller:
            car_controller.cleanup()
        
        if camera_handler:
            camera_handler.cleanup()
        
        if cap:
            cap.release()
        
        cv2.destroyAllWindows()
        print("Sistema terminato")

def test_picarx():
    """Funzione di test per verificare il funzionamento di PicarX"""
    print("=== Test PicarX ===")
    
    if not PICARX_AVAILABLE:
        print("PicarX non disponibile")
        return
    
    try:
        px = Picarx()
        print("PicarX inizializzato con successo")
        
        # Test movimenti base
        print("Test sterzo sinistro...")
        px.set_dir_servo_angle(-20)
        time.sleep(1)
        
        print("Test sterzo diritto...")
        px.set_dir_servo_angle(0)
        time.sleep(1)
        
        print("Test sterzo destro...")
        px.set_dir_servo_angle(20)
        time.sleep(1)
        
        print("Test sterzo al centro...")
        px.set_dir_servo_angle(0)
        time.sleep(1)
        
        # Test movimento (commentato per sicurezza)
        # print("Test movimento avanti...")
        # px.forward(30)
        # time.sleep(2)
        # px.stop()
        
        print("Test completato con successo!")
        
    except Exception as e:
        print(f"Errore nel test PicarX: {e}")

def test_camera():
    """Funzione di test per verificare il funzionamento della camera"""
    print("=== Test PiCamera2 ===")
    
    if not PICAMERA_AVAILABLE:
        print("PiCamera2 non disponibile")
        return
    
    try:
        camera = CameraHandler()
        if not camera.enabled:
            print("Impossibile inizializzare la camera")
            return
        
        print("Camera inizializzata, cattura 10 frame di test...")
        
        for i in range(10):
            frame = camera.capture_frame()
            if frame is not None:
                print(f"Frame {i+1}: {frame.shape}")
                # Mostra il primo frame
                if i == 0:
                    cv2.imshow('Test Camera', cv2.resize(frame, (640, 480)))
                    cv2.waitKey(1000)  # Mostra per 1 secondo
            else:
                print(f"Errore cattura frame {i+1}")
            time.sleep(0.1)
        
        camera.cleanup()
        cv2.destroyAllWindows()
        print("Test camera completato!")
        
    except Exception as e:
        print(f"Errore nel test camera: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema Lane Detection per PicarX')
    parser.add_argument('--test-picarx', action='store_true', 
                       help='Esegui test PicarX')
    parser.add_argument('--test-camera', action='store_true', 
                       help='Esegui test PiCamera2')
    parser.add_argument('--video', type=str, default='video.mp4',
                       help='File video da usare se PiCamera2 non disponibile')
    
    args = parser.parse_args()
    
    if args.test_picarx:
        test_picarx()
    elif args.test_camera:
        test_camera()
    else:
        main()