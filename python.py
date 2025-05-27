import cv2
import numpy as np
import os
import time
from picamera2 import Picamera2
from picarx import Picarx

# Costanti per controllo sterzo
MAX_STEERING_ANGLE    = 30    # Ridotto da 45 per evitare sterzate eccessive
STEERING_AGGRESSION   = 0.7   # Fattore di aggressività (0.5-1.0)
MIN_LANE_WIDTH        = 120   # px, soglia per carreggiata piccola
MIN_CENTER_DISTANCE   = 30    # px, soglia distanza minima linea-centro
OUTPUT_DIR            = "./ignored_frames"

# Costanti per PicarX
BASE_SPEED           = 0      # velocità base del veicolo
STEERING_MULTIPLIER  = 1.0    # Ridotto per conversione più delicata

# Costanti per controllo posizione
DESIRED_OFFSET       = 20     # px, offset desiderato dalle linee
LANE_CENTER_TOLERANCE = 15    # px, tolleranza per considerare "centrato"

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
    """Applica l'angolo di sterzo al PicarX con mapping migliorato"""
    # Limita l'angolo prima della conversione
    steering_angle_deg = np.clip(steering_angle_deg, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
    
    # Converte l'angolo di sterzo in angolo servo con mapping più delicato
    servo_angle = steering_angle_deg * STEERING_MULTIPLIER
    servo_angle = np.clip(servo_angle, -30, 30)  # Limita ulteriormente per sicurezza
    
    # Applica al servo di direzione
    px.set_dir_servo_angle(servo_angle)

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

def calculate_steering_angle(left_line, right_line, left_valid, right_valid, bev_width, bev_height):
    """Calcola l'angolo di sterzo basato sulle linee rilevate"""
    center_x = bev_width // 2
    base_y = bev_height - 1
    
    # Calcola le posizioni base delle linee
    if left_valid:
        lx1, ly1, lx2, ly2 = left_line
        left_base_x = int(np.interp(base_y, [ly2, ly1], [lx2, lx1]))
    
    if right_valid:
        rx1, ry1, rx2, ry2 = right_line
        right_base_x = int(np.interp(base_y, [ry2, ry1], [rx2, ry1]))
    
    target_x = center_x  # Posizione target di default (centro)
    
    if left_valid and right_valid:
        # Entrambe le linee visibili: calcola il centro della carreggiata
        lane_center = (left_base_x + right_base_x) // 2
        
        # Verifica se siamo già abbastanza centrati
        if abs(lane_center - center_x) <= LANE_CENTER_TOLERANCE:
            return 0.0  # Mantieni direzione dritta
        
        # Altrimenti, punta verso il centro della carreggiata
        target_x = lane_center
        
    elif left_valid:
        # Solo linea sinistra: mantieni offset a destra
        target_x = left_base_x + DESIRED_OFFSET
        
    elif right_valid:
        # Solo linea destra: mantieni offset a sinistra
        target_x = right_base_x - DESIRED_OFFSET
    
    else:
        # Nessuna linea: mantieni direzione precedente
        return previous_steering_angle
    
    # Calcola deviazione dal centro del BEV
    deviation = target_x - center_x
    
    # Normalizza la deviazione (-1 a 1) e converti in angolo
    max_deviation = bev_width // 2
    normalized_deviation = np.clip(deviation / max_deviation, -1.0, 1.0)
    
    # Applica aggressività e calcola angolo finale
    steering_angle = -normalized_deviation * MAX_STEERING_ANGLE * STEERING_AGGRESSION
    
    return steering_angle

def draw_steering_trajectory(img, steering_angle_deg):
    """Disegna la traiettoria prevista sul frame"""
    h, w = img.shape[:2]
    cx, cy = w // 2, h - 30
    L = 100
    rad = np.radians(-steering_angle_deg)
    ex = int(cx + L * np.sin(rad))
    ey = int(cy - L * np.cos(rad))
    
    # Colore in base all'angolo
    if abs(steering_angle_deg) < 10:
        color = (0, 255, 0)  # Verde: tutto ok
    elif abs(steering_angle_deg) < 20:
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

    # Calcola base X di ciascuna linea
    base_y = bev.shape[0] - 1
    lx_base = int(np.interp(base_y, [ly2, ly1], [lx2, lx1])) if lv else 0
    rx_base = int(np.interp(base_y, [ry2, ry1], [rx2, rx1])) if rv else bev.shape[1] - 1
    lane_width = abs(rx_base - lx_base)

    # Verifiche per ignorare il frame
    ignored_reasons = []
    if lv and rv and lane_width < MIN_LANE_WIDTH:
        ignored_reasons.append("Carreggiata troppo piccola")
    
    center_x = bev.shape[1] // 2
    if (lv and abs(lx_base - center_x) < MIN_CENTER_DISTANCE) or \
       (rv and abs(rx_base - center_x) < MIN_CENTER_DISTANCE):
        ignored_reasons.append("Linea vicina al centro")

    # Calcola angolo di sterzo
    if ignored_reasons:
        steering_angle = previous_steering_angle
        print(f"Frame ignorato: {', '.join(ignored_reasons)}")
    else:
        steering_angle = calculate_steering_angle(
            left_line, right_line, lv, rv, 
            bev.shape[1], bev.shape[0]
        )

    # Smoothing migliorato per evitare movimenti bruschi
    max_change = 15  # Massimo cambiamento per frame
    if abs(steering_angle - previous_steering_angle) > max_change:
        change_direction = np.sign(steering_angle - previous_steering_angle)
        steering_angle = previous_steering_angle + change_direction * max_change
    
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
    cv2.line(frame, tuple(l1.astype(int)), tuple(l2.astype(int)), (255,0,0), 4)  # Sinistra: blu
    cv2.line(frame, tuple(r1.astype(int)), tuple(r2.astype(int)), (0,0,255), 4)  # Destra: rosso
    
    # Disegna area carreggiata (overlay verde semitrasparente)
    overlay_lane = frame.copy()
    cv2.fillPoly(overlay_lane, [np.array([l1, l2, r2, r1], dtype=np.int32)], (0,255,0))
    frame = cv2.addWeighted(overlay_lane, 0.3, frame, 0.7, 0)

    # Aggiunge informazioni testuali con debug aggiuntivo
    cv2.putText(frame, f"Larghezza: {lane_width}px", (10, h-100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Velocita: {BASE_SPEED}px/s", (10, h-130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    
    # Mostra stato linee
    status = f"L:{lv} R:{rv}"
    cv2.putText(frame, status, (10, h-160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    
    if ignored_reasons:
        text = ", ".join(ignored_reasons)
        cv2.putText(frame, f"Ignorato: {text}", (10, h-70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    # Disegna traiettoria di sterzo
    draw_steering_trajectory(frame, -steering_angle)

    # Salva frame ignorati per debug
    if ignored_reasons:
        ignored_count += 1
        filename = f"frame_ignored_{ignored_count}_{int(time.time()*1000)}.png"
        #cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)

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
    print("Avvio sistema di lane detection...")
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