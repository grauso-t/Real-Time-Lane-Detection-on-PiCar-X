import cv2
import time
import numpy as np
from picamera2 import Picamera2
from picarx import Picarx

px = Picarx()  # Inizializza il controllo del veicolo
picam2 = Picamera2()  # Inizializza l'oggetto Picamera2 per acquisire immagini.
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)

# === VARIABILI DI STATO ===
# Memorizzano gli ultimi valori validi per garantire continuità nel controllo
last_valid_steering_angle = 0.0      # Ultimo angolo di sterzata valido complessivo
last_valid_left_angle = None         # Ultimo angolo della linea sinistra valido
last_valid_right_angle = None        # Ultimo angolo della linea destra valido
last_lane_center = None              # Ultimo centro corsia valido
last_detected_side = None            # Ultima configurazione rilevata: 'both', 'left', 'right'

# === PARAMETRI DI CONFIGURAZIONE ===
STEERING_METHOD = 'hybrid_deSantis'  # Metodo di calcolo angolo
SMOOTHING_FACTOR = 0.3               # Fattore di smoothing (0-1): più alto = transizioni più morbide
MIN_LANE_WIDTH = 150                 # Larghezza minima carreggiata in pixel
MAX_LANE_WIDTH = 400                 # Larghezza massima carreggiata in pixel
SPEED = 1                           # Velocità del veicolo

# === PARAMETRI OFFSET LINEA ===
DESIRED_OFFSET_FROM_LINE = 60        # Offset desiderato dalla linea singola (pixel)
LANE_CENTER_PREFERENCE = 0.0         # Bias verso centro corsia (-1.0 a 1.0): -1=sinistra, 0=centro, 1=destra

# === PARAMETRI CONTROLLO STERZATA ===
MAX_STEERING_ANGLE = 35.0           # Angolo massimo di sterzata
STEERING_GAIN = 0.8                 # Guadagno per la risposta di sterzata
ANGULAR_SMOOTHING = 0.7             # Smoothing specifico per l'angolo finale

px.forward(SPEED)  # Avvia il veicolo in avanti

def create_info_panel(width=400, height=700):
    """Crea un pannello informativo con sfondo nero"""
    return np.zeros((height, width, 3), dtype=np.uint8)

def add_text_to_panel(panel, text, position, font_scale=0.6, color=(255, 255, 255), thickness=1):
    """Aggiunge testo al pannello informativo"""
    cv2.putText(panel, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def add_section_title(panel, title, y_pos, color=(0, 255, 255)):
    """Aggiunge un titolo di sezione al pannello"""
    cv2.line(panel, (10, y_pos - 5), (390, y_pos - 5), color, 1)
    add_text_to_panel(panel, title, (10, y_pos + 15), font_scale=0.7, color=color, thickness=2)
    return y_pos + 35

def calculate_steering_angle(slope, method='exponential'):
    """Calcola l'angolo di sterzata basato sulla pendenza con diverse modalità di risposta"""
    if slope is None:
        return 0.0
        
    # Limita la pendenza per evitare valori estremi
    slope = np.clip(slope, -2.0, 2.0)
    
    if method == 'hybrid_deSantis':
        # Ritorno angolo di sterzata precedente se la pendenza è zero
        if slope == 0:
            return last_valid_steering_angle

        if abs(slope) >= 0.6:
            # Applica direttamente l'angolo massimo
            return np.sign(slope) * MAX_STEERING_ANGLE
        else:
            # Risposta esponenziale per pendenze più contenute
            normalized_slope = slope / 0.5
            sign = np.sign(normalized_slope)
            abs_normalized = abs(normalized_slope)
            
            if abs_normalized <= 1.0:
                exponential_response = (np.exp(abs_normalized * 1.5) - 1) / (np.exp(1.5) - 1)
            else:
                exponential_response = 1.0
            
            return sign * exponential_response * MAX_STEERING_ANGLE
    
    # Fallback lineare
    normalized_slope = np.clip(slope / 0.5, -1.0, 1.0)
    return normalized_slope * MAX_STEERING_ANGLE

def validate_lane_geometry(left_poly, right_poly, dst_w, dst_h, min_lane_width=150, max_lane_width=400):
    """Valida la geometria delle linee rilevate"""
    center_x = dst_w // 2
    y_test = dst_h // 2
    
    if left_poly is not None and right_poly is not None:
        # Entrambe le linee rilevate
        x_left = np.polyval(left_poly, y_test)
        x_right = np.polyval(right_poly, y_test)
        
        # Controlli di validità
        if x_left >= center_x:
            return False, 0, "Linea sinistra a destra del centro"
        if x_right <= center_x:
            return False, 0, "Linea destra a sinistra del centro"
        
        lane_width = x_right - x_left
        if lane_width < min_lane_width or lane_width > max_lane_width:
            return False, lane_width, f"Larghezza anomala: {lane_width:.1f}px"
        
        return True, lane_width, "Geometria valida - entrambe linee"
    
    elif left_poly is not None:
        # Solo linea sinistra
        x_left = np.polyval(left_poly, y_test)
        if x_left >= center_x - 20:  # Margine di tolleranza
            return False, 0, "Linea sinistra troppo centrale"
        return True, 0, "Solo linea sinistra valida"
    
    elif right_poly is not None:
        # Solo linea destra
        x_right = np.polyval(right_poly, y_test)
        if x_right <= center_x + 20:  # Margine di tolleranza
            return False, 0, "Linea destra troppo centrale"
        return True, 0, "Solo linea destra valida"
    
    return False, 0, "Nessuna linea rilevata"

def calculate_target_position(left_poly, right_poly, dst_w, dst_h, detected_side):
    """
    Calcola la posizione target basata sulla strategia di offset dinamico
    
    Args:
        left_poly, right_poly: Polinomi delle linee
        dst_w, dst_h: Dimensioni immagine
        detected_side: Configurazione rilevata ('both', 'left', 'right')
    
    Returns:
        tuple: (target_x, strategy_used)
    """
    y_eval = dst_h * 2 // 3  # Punto di valutazione più vicino al veicolo
    image_center = dst_w // 2
    
    if detected_side == 'both' and left_poly is not None and right_poly is not None:
        # STRATEGIA 1: Entrambe le linee - segui il centro corsia con bias
        x_left = np.polyval(left_poly, y_eval)
        x_right = np.polyval(right_poly, y_eval)
        lane_center = (x_left + x_right) / 2
        
        # Applica il bias di preferenza
        lane_width = x_right - x_left
        bias_offset = LANE_CENTER_PREFERENCE * (lane_width * 0.25)  # Max 25% della larghezza corsia
        target_x = lane_center + bias_offset
        
        return target_x, "Centro corsia con bias"
    
    elif detected_side == 'left' and left_poly is not None:
        # STRATEGIA 2: Solo linea sinistra - mantieni offset costante
        x_left = np.polyval(left_poly, y_eval)
        target_x = x_left + DESIRED_OFFSET_FROM_LINE
        
        # Assicurati che il target non sia troppo a destra
        target_x = min(target_x, dst_w - 30)
        
        return target_x, f"Offset da linea SX ({DESIRED_OFFSET_FROM_LINE}px)"
    
    elif detected_side == 'right' and right_poly is not None:
        # STRATEGIA 3: Solo linea destra - mantieni offset costante
        x_right = np.polyval(right_poly, y_eval)
        target_x = x_right - DESIRED_OFFSET_FROM_LINE
        
        # Assicurati che il target non sia troppo a sinistra
        target_x = max(target_x, 30)
        
        return target_x, f"Offset da linea DX ({DESIRED_OFFSET_FROM_LINE}px)"
    
    else:
        # STRATEGIA 4: Fallback - usa ultimo centro valido o centro immagine
        if last_lane_center is not None:
            return last_lane_center, "Ultimo centro valido"
        else:
            return image_center, "Centro immagine (fallback)"

def calculate_steering_from_position_error(target_x, current_x, dst_w):
    """
    Calcola l'angolo di sterzata basato sull'errore di posizione
    
    Args:
        target_x: Posizione target in pixel
        current_x: Posizione corrente (centro immagine)
        dst_w: Larghezza immagine
        
    Returns:
        float: Angolo di sterzata in gradi
    """
    # Errore di posizione in pixel
    position_error = target_x - current_x
    
    # Normalizza l'errore rispetto alla larghezza dell'immagine
    normalized_error = position_error / (dst_w / 2)
    
    # Applica il guadagno e limita l'angolo
    steering_angle = normalized_error * MAX_STEERING_ANGLE * STEERING_GAIN
    steering_angle = np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
    
    return steering_angle

# === CICLO PRINCIPALE DI ELABORAZIONE ===
while True:
    # Leggi il prossimo frame dal video
    frame = picam2.capture_array()
    
    # === PREPARAZIONE DELL'IMMAGINE ===
    h, w = frame.shape[:2]
    roi_top, roi_bottom = int(h * 0.6), h
    
    # === TRASFORMAZIONE PROSPETTICA (BIRD-EYE VIEW) ===
    src = np.float32([[0, roi_top], [w, roi_top], [w, roi_bottom], [0, roi_bottom]])
    dst_w, dst_h = 300, 200
    dst = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]])
    
    matrix = cv2.getPerspectiveTransform(src, dst)
    bird_eye = cv2.warpPerspective(frame, matrix, (dst_w, dst_h))
    
    # === CONVERSIONE SPAZIO COLORE E CREAZIONE MASCHERE ===
    hsv = cv2.cvtColor(bird_eye, cv2.COLOR_BGR2HSV)
    
    # Maschere per linee bianche e gialle
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    # === PREPROCESSING PER RILEVAMENTO LINEE ===
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # === RILEVAMENTO LINEE CON HOUGH TRANSFORM ===
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=30)
    
    # Inizializza immagine per il disegno e liste per le linee
    line_img = bird_eye.copy()
    left_lines = []
    right_lines = []
    
    # === CLASSIFICAZIONE DELLE LINEE ===
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            
            if abs(slope) < 0.5:  # Filtra linee troppo orizzontali
                continue
            
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    def draw_average_line(lines, color, label):
        """Calcola e restituisce il polinomio medio di un insieme di linee"""
        if lines:
            x_coords = []
            y_coords = []
            for x1, y1, x2, y2 in lines:
                x_coords += [x1, x2]
                y_coords += [y1, y2]
            
            if len(x_coords) > 0:
                poly = np.polyfit(y_coords, x_coords, deg=1)
                slope = poly[0]
                
                raw_steering_angle = calculate_steering_angle(slope, STEERING_METHOD)
                
                # Smoothing temporale per linea specifica
                if label == 'Left' and last_valid_left_angle is not None:
                    steering_angle = (SMOOTHING_FACTOR * last_valid_left_angle + 
                                    (1 - SMOOTHING_FACTOR) * raw_steering_angle)
                elif label == 'Right' and last_valid_right_angle is not None:
                    steering_angle = (SMOOTHING_FACTOR * last_valid_right_angle + 
                                    (1 - SMOOTHING_FACTOR) * raw_steering_angle)
                else:
                    steering_angle = raw_steering_angle
                
                return poly, slope, steering_angle
        
        return None, None, None

    # === CALCOLO DEI POLINOMI DELLE LINEE ===
    left_poly, left_slope, left_angle = draw_average_line(left_lines, (255, 0, 0), "Left")
    right_poly, right_slope, right_angle = draw_average_line(right_lines, (0, 0, 255), "Right")

    # === DETERMINAZIONE CONFIGURAZIONE RILEVATA ===
    if left_poly is not None and right_poly is not None:
        current_detected_side = 'both'
    elif left_poly is not None:
        current_detected_side = 'left'
    elif right_poly is not None:
        current_detected_side = 'right'
    else:
        current_detected_side = 'none'

    # === VALIDAZIONE GEOMETRICA ===
    geometry_valid, lane_width, validation_message = validate_lane_geometry(
        left_poly, right_poly, dst_w, dst_h, MIN_LANE_WIDTH, MAX_LANE_WIDTH
    )
    
    # === CALCOLO POSIZIONE TARGET E CONTROLLO ===
    image_center = dst_w // 2
    
    if geometry_valid and current_detected_side != 'none':
        # *** GEOMETRIA VALIDA - CALCOLA NUOVO TARGET ***
        
        target_x, strategy_used = calculate_target_position(
            left_poly, right_poly, dst_w, dst_h, current_detected_side
        )
        
        # Calcola angolo di sterzata basato sull'errore di posizione
        position_based_steering = calculate_steering_from_position_error(
            target_x, image_center, dst_w
        )
        
        # Aggiorna valori memorizzati
        last_lane_center = target_x
        last_detected_side = current_detected_side
        
        if left_angle is not None:
            last_valid_left_angle = left_angle
        if right_angle is not None:
            last_valid_right_angle = right_angle
        
        # Usa l'angolo basato sulla posizione come angolo finale
        current_steering_angle = position_based_steering
        
        # Disegna le linee valide
        line_color_left = (255, 0, 0) if left_poly is not None else None
        line_color_right = (0, 0, 255) if right_poly is not None else None
        
    else:
        # *** GEOMETRIA NON VALIDA - USA STRATEGIA DI FALLBACK ***
        
        if last_lane_center is not None:
            target_x = last_lane_center
            strategy_used = "Ultimo target valido"
        else:
            target_x = image_center
            strategy_used = "Centro immagine (emergency)"
        
        # Calcola angolo di sterzata basato sull'ultimo target valido
        current_steering_angle = calculate_steering_from_position_error(
            target_x, image_center, dst_w
        )
        
        # Disegna le linee rilevate in grigio (invalide)
        line_color_left = (100, 100, 100) if left_poly is not None else None
        line_color_right = (100, 100, 100) if right_poly is not None else None

    # === DISEGNO DELLE LINEE ===
    if left_poly is not None and line_color_left:
        y_start, y_end = 0, dst_h
        x_start = int(np.polyval(left_poly, y_start))
        x_end = int(np.polyval(left_poly, y_end))
        cv2.line(line_img, (x_start, y_start), (x_end, y_end), line_color_left, 3)
    
    if right_poly is not None and line_color_right:
        y_start, y_end = 0, dst_h
        x_start = int(np.polyval(right_poly, y_start))
        x_end = int(np.polyval(right_poly, y_end))
        cv2.line(line_img, (x_start, y_start), (x_end, y_end), line_color_right, 3)

    # === SMOOTHING FINALE DELL'ANGOLO ===
    if current_steering_angle is not None:
        if last_valid_steering_angle != 0.0:
            final_steering_angle = (ANGULAR_SMOOTHING * last_valid_steering_angle + 
                                  (1 - ANGULAR_SMOOTHING) * current_steering_angle)
        else:
            final_steering_angle = current_steering_angle
        
        last_valid_steering_angle = final_steering_angle
    else:
        final_steering_angle = last_valid_steering_angle

    # === COMANDO VEICOLO ===
    px.set_dir_servo_angle(final_steering_angle)

    # === VISUALIZZAZIONE TARGET E TRAIETTORIA ===
    # Disegna il target
    target_color = (0, 255, 0) if geometry_valid else (255, 165, 0)
    target_y = dst_h * 2 // 3
    cv2.circle(line_img, (int(target_x), int(target_y)), 8, target_color, -1)
    cv2.circle(line_img, (int(target_x), int(target_y)), 12, target_color, 2)
    
    # Disegna la linea di traiettoria
    cv2.line(line_img, (image_center, dst_h), (int(target_x), int(target_y)), target_color, 2)
    
    # Disegna il centro immagine
    cv2.circle(line_img, (image_center, dst_h - 10), 5, (255, 255, 255), -1)

    # === CREAZIONE PANNELLO INFORMATIVO MIGLIORATO ===
    info_panel = create_info_panel(450, 800)
    current_y = 30
    
    # === SEZIONE STATO SISTEMA ===
    current_y = add_section_title(info_panel, "STATO SISTEMA", current_y, (255, 255, 100))
    system_status = "OPERATIVO" if geometry_valid else "FALLBACK"
    status_color = (0, 255, 0) if geometry_valid else (255, 165, 0)
    add_text_to_panel(info_panel, f"Stato: {system_status}", (20, current_y), color=status_color, thickness=2)
    current_y += 25
    add_text_to_panel(info_panel, f"Configurazione: {current_detected_side.upper()}", (20, current_y))
    current_y += 25
    add_text_to_panel(info_panel, f"Strategia: {strategy_used}", (20, current_y), font_scale=0.5)
    current_y += 40
    
    # === SEZIONE CONTROLLO POSIZIONE ===
    current_y = add_section_title(info_panel, "CONTROLLO POSIZIONE", current_y, (100, 255, 255))
    add_text_to_panel(info_panel, f"Target X: {target_x:.1f}px", (20, current_y), thickness=2)
    current_y += 25
    add_text_to_panel(info_panel, f"Centro Img: {image_center}px", (20, current_y))
    current_y += 25
    position_error = target_x - image_center
    error_direction = "DESTRA" if position_error > 0 else "SINISTRA"
    add_text_to_panel(info_panel, f"Errore: {abs(position_error):.1f}px {error_direction}", (20, current_y))
    current_y += 40
    
    # === SEZIONE STERZATA ===
    current_y = add_section_title(info_panel, "CONTROLLO STERZATA", current_y, (255, 100, 255))
    add_text_to_panel(info_panel, f"Angolo Finale: {final_steering_angle:.1f}°", (20, current_y), 
                     font_scale=0.8, thickness=2, color=(255, 255, 255))
    current_y += 30
    add_text_to_panel(info_panel, f"Angolo Corrente: {current_steering_angle:.1f}°" if current_steering_angle else "Angolo Corrente: N/A", (20, current_y))
    current_y += 25
    steering_direction = "DESTRA" if final_steering_angle > 0 else "SINISTRA" if final_steering_angle < 0 else "DRITTO"
    add_text_to_panel(info_panel, f"Direzione: {steering_direction}", (20, current_y))
    current_y += 40
    
    # === SEZIONE RILEVAMENTO LINEE ===
    current_y = add_section_title(info_panel, "RILEVAMENTO LINEE", current_y, (255, 100, 100))
    
    # Linea sinistra
    if left_poly is not None:
        add_text_to_panel(info_panel, f"Sinistra: SI (slope: {left_slope:.3f})", (20, current_y), color=(0, 255, 0))
    else:
        add_text_to_panel(info_panel, f"Sinistra: NO", (20, current_y), color=(0, 0, 255))
    current_y += 25
    
    # Linea destra
    if right_poly is not None:
        add_text_to_panel(info_panel, f"Destra: SI (slope: {right_slope:.3f})", (20, current_y), color=(0, 255, 0))
    else:
        add_text_to_panel(info_panel, f"Destra: NO", (20, current_y), color=(0, 0, 255))
    current_y += 25
    
    # Larghezza corsia
    if lane_width > 0:
        add_text_to_panel(info_panel, f"Larghezza Corsia: {lane_width:.1f}px", (20, current_y))
    else:
        add_text_to_panel(info_panel, f"Larghezza Corsia: N/A", (20, current_y))
    current_y += 40
    
    # === SEZIONE PARAMETRI ===
    current_y = add_section_title(info_panel, "PARAMETRI", current_y, (200, 200, 200))
    add_text_to_panel(info_panel, f"Offset Linea: {DESIRED_OFFSET_FROM_LINE}px", (20, current_y), font_scale=0.5)
    current_y += 20
    add_text_to_panel(info_panel, f"Max Sterzata: {MAX_STEERING_ANGLE}°", (20, current_y), font_scale=0.5)
    current_y += 20
    add_text_to_panel(info_panel, f"Guadagno: {STEERING_GAIN}", (20, current_y), font_scale=0.5)
    current_y += 20
    add_text_to_panel(info_panel, f"Smoothing: {ANGULAR_SMOOTHING}", (20, current_y), font_scale=0.5)

    # === VISUALIZZAZIONE FINALE ===
    cv2.imshow("Original", frame)
    cv2.imshow("Bird Eye with Target", line_img)
    cv2.imshow("Info Panel", info_panel)
    
    # === CONTROLLO USCITA ===
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    time.sleep(0.03)  # ~33 FPS

# === CLEANUP ===
picam2.stop()
px.set_dir_servo_angle(0)
px.stop()
cv2.destroyAllWindows()