import cv2
import numpy as np
import math
from picamera2 import Picamera2
from time import sleep
from picarx import Picarx

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
sleep(2)  # Aspetta l'inizializzazione

px = Picarx()
px.forward(0)  # assicura che sia fermo

# Variabili globali
lane_width = 300  # Larghezza carreggiata in pixel
previous_angle = 0  # Angolo di sterzata precedente
frame_center = 150  # Centro fisso dell'immagine (metà di 300px)
dst_w, dst_h = 300, 200  # Dimensioni immagine trasformata

# Parametri per il controllo dell'angolo
SPEED = 1 # Velocità di movimento (1-100)
MIN_ANGLE = -45.0  # Angolo minimo in gradi
MAX_ANGLE = 45.0   # Angolo massimo in gradi
ANGLE_SMOOTHING = 0.7  # Fattore di smoothing per l'angolo (0-1)
SINGLE_LINE_OFFSET = 10  # pixel di offset verso il centro della strada

# Parametro per la posizione del centro dinamico (più piccolo = più in alto)
CENTER_Y_RATIO = 0.3  # 0.3 significa al 30% dell'altezza (più in alto rispetto a 0.5)

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
    """Valida le linee rilevate con controllo posizione rispetto al centro"""
    valid_left = False
    valid_right = False
    
    # Calcola la coordinata y per il controllo della posizione (stesso del centro dinamico)
    center_y = int(dst_h * CENTER_Y_RATIO)
    
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        line_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Controlla la lunghezza minima
        if line_length >= 20:
            # Calcola la posizione x della linea al centro dinamico
            line_x_at_center = get_line_x_at_y(left_line, center_y)
            
            # La linea sinistra deve essere effettivamente a sinistra del centro
            if line_x_at_center is not None and line_x_at_center < frame_center:
                valid_left = True
    
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        line_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Controlla la lunghezza minima
        if line_length >= 20:
            # Calcola la posizione x della linea al centro dinamico
            line_x_at_center = get_line_x_at_y(right_line, center_y)
            
            # La linea destra deve essere effettivamente a destra del centro
            if line_x_at_center is not None and line_x_at_center > frame_center:
                valid_right = True
    
    return valid_left, valid_right

def calculate_lane_center_and_angle(left_line, right_line, valid_left, valid_right):
    """Calcola il centro della carreggiata e l'angolo di sterzata"""
    global previous_angle
    
    lane_center = None
    steering_angle = previous_angle  # Mantieni l'angolo precedente come default
    
    # Calcola la coordinata y per il centro dinamico (più in alto)
    center_y = int(dst_h * CENTER_Y_RATIO)
    
    # CONTROLLO AGGIUNTIVO: Verifica che le linee siano nella posizione corretta
    # Se una linea è nella posizione sbagliata, la consideriamo non valida
    position_check_passed = True
    
    if valid_left:
        left_x = get_line_x_at_y(left_line, center_y)
        if left_x is not None and left_x >= frame_center:
            # Linea sinistra è a destra del centro - errore di rilevamento
            valid_left = False
            position_check_passed = False
            print("AVVISO: Linea sinistra rilevata a destra del centro - ignorata")
    
    if valid_right:
        right_x = get_line_x_at_y(right_line, center_y)
        if right_x is not None and right_x <= frame_center:
            # Linea destra è a sinistra del centro - errore di rilevamento
            valid_right = False
            position_check_passed = False
            print("AVVISO: Linea destra rilevata a sinistra del centro - ignorata")
    
    # Se il controllo di posizione non è passato, mantieni l'angolo precedente
    if not position_check_passed:
        print(f"Mantengo angolo precedente: {previous_angle:.1f}°")
        return lane_center, previous_angle, center_y
    
    # Se abbiamo entrambe le linee valide
    if valid_left and valid_right:
        left_x = get_line_x_at_y(left_line, center_y)
        right_x = get_line_x_at_y(right_line, center_y)
        
        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2
    
    # Se abbiamo solo la linea sinistra - aggiungi offset verso destra
    elif valid_left and not valid_right:
        left_x = get_line_x_at_y(left_line, center_y)
        if left_x is not None:
            # Centro carreggiata stimato + offset verso destra per sicurezza
            lane_center = left_x + lane_width / 2 + SINGLE_LINE_OFFSET
    
    # Se abbiamo solo la linea destra - aggiungi offset verso sinistra
    elif valid_right and not valid_left:
        right_x = get_line_x_at_y(right_line, center_y)
        if right_x is not None:
            # Centro carreggiata stimato + offset verso sinistra per sicurezza
            lane_center = right_x - lane_width / 2 - SINGLE_LINE_OFFSET
    
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
    else:
        # Se non abbiamo un centro valido, mantieni l'angolo precedente
        print(f"Nessun centro valido - mantengo angolo precedente: {previous_angle:.1f}°")
    
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
    info_img = img.copy()
    
    # Informazioni di stato
    y_offset = 30
    # Colore per l'angolo basato sui limiti
    angle_color = (0, 255, 0) if -15 <= steering_angle <= 15 else (0, 255, 255) if -30 <= steering_angle <= 30 else (0, 0, 255)
    cv2.putText(info_img, f"Steering: {steering_angle:.1f}°", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, angle_color, 2)
    
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
    # Mostra l'altezza del centro dinamico
    cv2.putText(info_img, f"Center Y: {center_y}px ({CENTER_Y_RATIO:.1f})", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
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

def main():
    """Funzione principale"""
    global lane_width, ANGLE_SMOOTHING, CENTER_Y_RATIO
    
    print("=== Sistema di Rilevamento Corsie ===")
    print("Controlli:")
    print("- Spazio: Play/Pause")
    print("- 'q': Esci")
    print("- 'r': Reset video")
    print("- '+/-': Aumenta/Diminuisci larghezza carreggiata")
    print("- 's': Aumenta smoothing angolo")
    print("- 'a': Diminuisci smoothing angolo")
    print("- 'u': Sposta centro più in alto")
    print("- 'd': Sposta centro più in basso")
    print(f"- Angolo limitato tra {MIN_ANGLE}° e {MAX_ANGLE}°")
    print(f"- Centro dinamico attualmente al {CENTER_Y_RATIO:.1f} dell'altezza")
    
    # Inizializza video
    video_path = "video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il video {video_path}")
        return
    
    print(f"Video caricato: {video_path}")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Fine del video o errore nella lettura")
                break
            
            # Processa il frame
            bird_eye, lane_result, edges, steering_angle = process_frame(frame)
            
            # Ridimensiona tutte le immagini alle stesse dimensioni
            display_width, display_height = 350, 250
            
            frame_resized = cv2.resize(frame, (display_width, display_height))
            bird_eye_resized = cv2.resize(bird_eye, (display_width, display_height))
            lane_result_resized = cv2.resize(lane_result, (display_width, display_height))
            
            # Crea immagine degli edges a colori per la visualizzazione
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_resized = cv2.resize(edges_colored, (display_width, display_height))
            
            # Combina tutte le visualizzazioni con dimensioni uniformi
            top_row = np.hstack([frame_resized, bird_eye_resized])
            bottom_row = np.hstack([lane_result_resized, edges_resized])
            
            # Crea visualizzazione finale
            combined = np.vstack([top_row, bottom_row])
        
        # Mostra l'immagine
        cv2.imshow('Lane Detection System', combined)
        
        # Gestione input
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spazio per pausa/play
            paused = not paused
            print("Video", "in pausa" if paused else "in riproduzione")
        elif key == ord('r'):  # Reset video
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
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

try:
    while True:
        # Acquisisci frame dalla fotocamera
        frame = picam2.capture_array()
        
        # Usa la funzione già definita nel tuo script
        bird_eye, info_img, edges, steering_angle = process_frame(frame)
        
        # Mostra l'immagine elaborata
        cv2.imshow("Lane Detection", info_img)
        
        # Usa l'angolo per controllare il servomotore della direzione
        px.set_dir_servo_angle(steering_angle)
        
        # Facoltativamente: muovi in avanti a bassa velocità
        px.forward(SPEED)
        
        # Interrompi con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('p'):
            SPEED = 0
        
        if cv2.waitKey(1) & 0xFF == ord('l'):
            SPEED = 1
            
except KeyboardInterrupt:
    pass

finally:
    px.stop()
    picam2.stop()
    cv2.destroyAllWindows()