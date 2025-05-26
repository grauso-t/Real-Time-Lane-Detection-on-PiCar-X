import cv2
import numpy as np
import os
import time
from picamera2 import Picamera2
from picarx import Picarx

# Costanti per simulazione
MAX_STEERING_ANGLE    = 45    # gradi
STEERING_AGGRESSION   = 2
MIN_LANE_WIDTH        = 120   # px, soglia per carreggiata piccola
MIN_CENTER_DISTANCE   = 60    # px, soglia distanza minima linea-centro
OUTPUT_DIR            = "./ignored_frames"

# Nuove costanti per centraggio
DEFAULT_LANE_WIDTH    = 200   # px, larghezza stimata della carreggiata
LANE_WIDTH_MEMORY     = []    # memoria delle larghezze recenti
MEMORY_SIZE           = 10    # numero di frame da ricordare
CENTER_OFFSET_FACTOR  = 0.3   # fattore per correzione graduale del centraggio

# Costanti per PicarX  
BASE_SPEED           = 0     # velocità base del veicolo
STEERING_MULTIPLIER  = 1      # moltiplicatore per conversione angolo->servo

# Setup directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inizializzazione PicarX
print("Inizializzazione PicarX...")
px = Picarx()
px.forward(0)  # ferma il veicolo all'avvio

# Inizializzazione Picamera2
print("Inizializzazione Picamera2...")
picam2 = Picamera2()

# Configurazione camera per ottimizzare prestazioni
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 30}
)
picam2.configure(config)
picam2.start()

# Variabili globali
previous_steering_angle = 0.0
ignored_count = 0
running = True
estimated_lane_center = None  # Centro stimato della carreggiata

def cleanup():
    """Pulizia risorse al termine"""
    global running
    running = False
    px.forward(0)  # ferma il veicolo
    px.set_dir_servo_angle(0)  # centra il servo
    picam2.stop()
    cv2.destroyAllWindows()
    print("Cleanup completato")

def apply_steering_to_picarx(steering_angle_deg):
    """Applica l'angolo di sterzo al PicarX"""
    # Converte l'angolo di sterzo in angolo servo
    servo_angle = steering_angle_deg * STEERING_MULTIPLIER
    servo_angle = np.clip(servo_angle, -45, 45)
    
    # Applica al servo di direzione
    px.set_dir_servo_angle(servo_angle)
    
    # Controlla velocità in base all'angolo di sterzo
    if abs(steering_angle_deg) < 10:
        speed = BASE_SPEED
    elif abs(steering_angle_deg) < 25:
        speed = BASE_SPEED
    else:
        speed = BASE_SPEED
    
    px.forward(int(speed))

def update_lane_width_memory(width):
    """Aggiorna la memoria delle larghezze della carreggiata"""
    global LANE_WIDTH_MEMORY
    LANE_WIDTH_MEMORY.append(width)
    if len(LANE_WIDTH_MEMORY) > MEMORY_SIZE:
        LANE_WIDTH_MEMORY.pop(0)

def get_average_lane_width():
    """Calcola la larghezza media della carreggiata dalla memoria"""
    if len(LANE_WIDTH_MEMORY) == 0:
        return DEFAULT_LANE_WIDTH
    return sum(LANE_WIDTH_MEMORY) / len(LANE_WIDTH_MEMORY)

def estimate_lane_center(left_line, right_line, left_valid, right_valid, image_width):
    """Stima il centro della carreggiata basandosi sulle linee disponibili"""
    global estimated_lane_center
    
    # Calcola posizioni base delle linee
    base_y = 199  # bottom della BEV
    
    if left_valid:
        lx1, ly1, lx2, ly2 = left_line
        left_base = int(np.interp(base_y, [ly2, ly1], [lx2, lx1]))
    
    if right_valid:
        rx1, ry1, rx2, ry2 = right_line
        right_base = int(np.interp(base_y, [ry2, ry1], [rx2, rx1]))
    
    if left_valid and right_valid:
        # Entrambe le linee visibili: centro reale
        lane_center = (left_base + right_base) / 2
        lane_width = abs(right_base - left_base)
        update_lane_width_memory(lane_width)
        estimated_lane_center = lane_center
        return lane_center, "both_lines"
        
    elif left_valid and not right_valid:
        # Solo linea sinistra: stima la destra
        avg_width = get_average_lane_width()
        estimated_right = left_base + avg_width
        lane_center = (left_base + estimated_right) / 2
        
        # Aggiustamento graduale se abbiamo una stima precedente
        if estimated_lane_center is not None:
            lane_center = estimated_lane_center + (lane_center - estimated_lane_center) * CENTER_OFFSET_FACTOR
        
        estimated_lane_center = lane_center
        return lane_center, "left_only"
        
    elif right_valid and not left_valid:
        # Solo linea destra: stima la sinistra
        avg_width = get_average_lane_width()
        estimated_left = right_base - avg_width
        lane_center = (estimated_left + right_base) / 2
        
        # Aggiustamento graduale se abbiamo una stima precedente
        if estimated_lane_center is not None:
            lane_center = estimated_lane_center + (lane_center - estimated_lane_center) * CENTER_OFFSET_FACTOR
        
        estimated_lane_center = lane_center
        return lane_center, "right_only"
        
    else:
        # Nessuna linea: mantieni stima precedente o usa centro immagine
        if estimated_lane_center is not None:
            return estimated_lane_center, "memory"
        else:
            return image_width / 2, "default"

def calculate_steering_from_center(lane_center, image_center, image_width):
    """Calcola l'angolo di sterzo per mantenere il centro della carreggiata"""
    # Differenza tra centro carreggiata e centro immagine
    center_offset = lane_center - image_center
    
    # Normalizza l'offset rispetto alla larghezza dell'immagine
    normalized_offset = center_offset / (image_width / 2)
    
    # Converti in angolo di sterzo (invertito perché sinistra = negativo)
    steering_angle = -normalized_offset * MAX_STEERING_ANGLE * STEERING_AGGRESSION
    
    # Limita l'angolo
    steering_angle = np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
    
    return steering_angle

def average_line(lines, side, width, height):
    """Calcola la linea media data una lista di segmenti"""
    if len(lines) == 0:
        # default line se non ne trova
        if side == 'left':
            return (0, height, width // 3, int(height * 0.2))
        else:
            return (width, height, (2 * width) // 3, int(height * 0.2))

    xs, ys = [], []
    for x1, y1, x2, y2 in lines:
        xs += [x1, x2]
        ys += [y1, y2]
    
    # Verifica che abbiamo abbastanza punti
    if len(xs) < 2:
        if side == 'left':
            return (0, height, width // 3, int(height * 0.2))
        else:
            return (width, height, (2 * width) // 3, int(height * 0.2))
    
    m, b = np.polyfit(ys, xs, 1)
    y1, y2 = height, int(height * 0.2)
    x1 = int(m * y1 + b)
    x2 = int(m * y2 + b)
    return (x1, y1, x2, y2)

def draw_steering_trajectory(img, steering_angle_deg):
    """Disegna la traiettoria prevista sul frame"""
    h, w = img.shape[:2]
    cx, cy = w // 2, h - 30
    L = 100
    rad = np.radians(-steering_angle_deg)
    ex = int(cx + L * np.sin(rad))
    ey = int(cy - L * np.cos(rad))
    
    # Colore in base all'angolo
    if abs(steering_angle_deg) < 15:
        color = (0, 255, 0)  # Verde: tutto ok
    elif abs(steering_angle_deg) < 30:
        color = (0, 165, 255)  # Arancione: attenzione
    else:
        color = (0, 0, 255)  # Rosso: curva stretta
    
    cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 4)
    cv2.putText(img, f"Steering: {steering_angle_deg:.1f}°", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def process_frame():
    
    global BASE_SPEED
    
    """Processa un singolo frame dalla camera"""
    global previous_steering_angle, ignored_count
    
    # Cattura frame dalla Picamera2
    frame = picam2.capture_array()
    
    if frame is None:
        return False
    
    h, w = frame.shape[:2]
    roi_top, roi_bot = int(h * 0.6), h 

    # Prepara overlay semitrasparente per info
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, roi_top), (w, roi_bot), (0, 0, 0), -1)
    info_overlay = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # Bird's Eye View per calcolo carreggiata
    src = np.float32([[0, roi_top], [w, roi_top], [w, roi_bot], [0, roi_bot]])
    dst = np.float32([[0, 0], [300, 0], [300, 200], [0, 200]])
    bev = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (300, 200))

    # Segmentazione colori e rilevamento bordi
    hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
    # Maschera per linee gialle
    ym = cv2.inRange(hls, np.array([15, 40, 100]), np.array([35, 255, 255]))
    # Maschera per linee bianche
    wm = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 255]))
    
    mask = cv2.morphologyEx(cv2.bitwise_or(ym, wm), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    edges = cv2.Canny(mask, 50, 150)
    
    # Rilevamento linee con Hough Transform
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=15, maxLineGap=20)

    # Separa segmenti sinistri e destri
    left_ls, right_ls = [], []
    if raw_lines is not None:
        for l in raw_lines:
            x1, y1, x2, y2 = l[0]
            if x2 - x1 == 0:  # Evita divisione per zero
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.6:  # Ignora linee troppo orizzontali
                continue
            (left_ls if slope < 0 else right_ls).append((x1, y1, x2, y2))

    # Calcola linee medie
    left_line  = average_line(left_ls,  'left',  bev.shape[1], bev.shape[0])
    right_line = average_line(right_ls, 'right', bev.shape[1], bev.shape[0])
    lx1, ly1, lx2, ly2 = left_line
    rx1, ry1, rx2, ry2 = right_line
    lv, rv = bool(left_ls), bool(right_ls)  # linee valide

    # **NUOVA LOGICA DI CENTRAGGIO**
    # Stima il centro della carreggiata
    bev_center = bev.shape[1] / 2
    lane_center, center_method = estimate_lane_center(left_line, right_line, lv, rv, bev.shape[1])
    
    # Calcola angolo di sterzo basato sul centraggio
    steering_angle = calculate_steering_from_center(lane_center, bev_center, bev.shape[1])
    
    # Smoothing per evitare movimenti bruschi
    if abs(steering_angle - previous_steering_angle) > 15:
        steering_angle = previous_steering_angle + np.sign(steering_angle - previous_steering_angle) * 15
    
    previous_steering_angle = steering_angle

    # Applica sterzo al PicarX
    apply_steering_to_picarx(steering_angle)

    # Visualizzazione risultati
    # Disegna linee e area carreggiata sul frame originale
    invM = cv2.getPerspectiveTransform(dst, src)
    pts = np.array([[lx1, ly1], [lx2, ly2], [rx2, ry2], [rx1, ry1]], dtype=np.float32).reshape(-1,1,2)
    pts_orig = cv2.perspectiveTransform(pts, invM).reshape(-1,2)
    l1, l2, r2, r1 = pts_orig
    
    # Disegna linee
    line_color_left = (255, 0, 0) if lv else (128, 128, 128)  # Blu se valida, grigio se stimata
    line_color_right = (0, 0, 255) if rv else (128, 128, 128)  # Rosso se valida, grigio se stimata
    
    cv2.line(frame, tuple(l1.astype(int)), tuple(l2.astype(int)), line_color_left, 4)
    cv2.line(frame, tuple(r1.astype(int)), tuple(r2.astype(int)), line_color_right, 4)
    
    # Disegna area carreggiata (overlay verde semitrasparente)
    overlay_lane = frame.copy()
    cv2.fillPoly(overlay_lane, [np.array([l1, l2, r2, r1], dtype=np.int32)], (0,255,0))
    frame = cv2.addWeighted(overlay_lane, 0.3, frame, 0.7, 0)

    # Disegna centro stimato della carreggiata
    center_bev_point = np.array([[[lane_center, bev.shape[0]-1]]], dtype=np.float32)
    center_orig_point = cv2.perspectiveTransform(center_bev_point, invM)[0][0]
    cv2.circle(frame, tuple(center_orig_point.astype(int)), 10, (255, 255, 0), -1)  # Cerchio ciano per centro

    # Aggiunge informazioni testuali
    avg_width = get_average_lane_width()
    cv2.putText(frame, f"Larghezza media: {avg_width:.0f}px", (10, h-130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Metodo centro: {center_method}", (10, h-100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Offset centro: {lane_center - bev_center:.1f}px", (10, h-70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    
    # Disegna traiettoria di sterzo
    draw_steering_trajectory(frame, steering_angle)

    # Mostra video (opzionale, commenta se non hai display)
    try:
        cv2.imshow("Camera View", frame)
        cv2.imshow("BEV View", bev)
        cv2.imshow("Edges", edges)
        cv2.imshow("Mask", mask)
        
        # Controlla input tastiera per uscita
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord(' '):  # Spazio per pausa
            BASE_SPEED = 0
            px.forward(BASE_SPEED)
        elif key == ord('.'):
            BASE_SPEED = 1
            px.forward(BASE_SPEED)
    except:
        # Se non c'è display, ignora gli errori di visualizzazione
        pass
    
    return True

def main():
    """Funzione principale"""
    print("Avvio sistema di lane detection con centraggio migliorato...")
    print("Premi 'q' per uscire, 'spazio' per pausa")
    
    try:
        # Loop principale
        while running:
            if not process_frame():
                break
            
            # Piccola pausa per non sovraccaricare il sistema
            time.sleep(0.05)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\nInterruzione da tastiera")
    except Exception as e:
        print(f"Errore: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()