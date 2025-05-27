import cv2
import numpy as np
import os
import time
from picamera2 import Picamera2
from picarx import Picarx

# Costanti per simulazione
MAX_STEERING_ANGLE    = 45    # gradi
STEERING_AGGRESSION   = 1
MIN_LANE_WIDTH        = 120   # px, soglia per carreggiata piccola
MIN_CENTER_DISTANCE   = 30    # px, soglia distanza minima linea-centro
OUTPUT_DIR            = "./ignored_frames"

# Costanti per smoothing dello sterzo
STEERING_SMOOTHING_FACTOR = 0.3   # Fattore di smoothing (0.1 = molto lento, 1.0 = immediato)
MAX_STEERING_CHANGE = 5           # Massimo cambiamento di sterzo per frame (gradi)
STEERING_DEADZONE = 2             # Zona morta per piccole correzioni (gradi)

# Costanti per PicarX
BASE_SPEED           = 0     # velocità base del veicolo
STEERING_MULTIPLIER  = 10      # moltiplicatore per conversione angolo->servo

# Costanti per offset e visualizzazione
LANE_OFFSET_RATIO    = 0.15   # ratio dell'offset rispetto alla larghezza carreggiata (15%)
MIN_OFFSET_PX        = 20     # offset minimo in pixel
MAX_OFFSET_PX        = 60     # offset massimo in pixel
ESTIMATED_LANE_WIDTH = 200    # larghezza stimata della carreggiata in pixel (BEV)

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
last_valid_lane = None  # Memorizza l'ultima carreggiata valida

def cleanup():
    """Pulizia risorse al termine"""
    global running
    running = False
    px.forward(0)  # ferma il veicolo
    px.set_dir_servo_angle(0)  # centra il servo
    picam2.stop()
    cv2.destroyAllWindows()
    print("Cleanup completato")

def smooth_steering_angle(target_angle, previous_angle):
    """Applica smoothing graduale all'angolo di sterzo"""
    
    # Calcola la differenza tra target e angolo precedente
    angle_diff = target_angle - previous_angle
    
    # Applica zona morta per evitare micro-correzioni
    if abs(angle_diff) < STEERING_DEADZONE:
        return previous_angle
    
    # Limita il cambiamento massimo per frame
    if abs(angle_diff) > MAX_STEERING_CHANGE:
        angle_diff = np.sign(angle_diff) * MAX_STEERING_CHANGE
    
    # Applica smoothing esponenziale
    smoothed_change = angle_diff * STEERING_SMOOTHING_FACTOR
    new_angle = previous_angle + smoothed_change
    
    # Clamp finale per sicurezza
    new_angle = np.clip(new_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
    
    return new_angle

def set_steering_angle(steering_angle_deg):
    """Applica l'angolo di sterzo al PicarX"""
    # Converte l'angolo di sterzo in angolo servo
    servo_angle = steering_angle_deg * STEERING_MULTIPLIER
    servo_angle = np.clip(servo_angle, -45, 45)
    
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

def calculate_lane_center_and_offset(left_line, right_line, left_valid, right_valid, bev_shape):
    """Calcola il centro della carreggiata e l'offset target"""
    h, w = bev_shape[:2]
    base_y = h - 1
    
    # Calcola posizioni base delle linee
    if left_valid:
        lx1, ly1, lx2, ly2 = left_line
        left_base = int(np.interp(base_y, [ly2, ly1], [lx2, lx1]))
    else:
        left_base = None
        
    if right_valid:
        rx1, ry1, rx2, ry2 = right_line
        right_base = int(np.interp(base_y, [ry2, ry1], [rx2, rx1]))
    else:
        right_base = None
    
    # Calcola centro e larghezza carreggiata
    if left_valid and right_valid:
        # Entrambe le linee visibili
        lane_center = (left_base + right_base) // 2
        lane_width = abs(right_base - left_base)
        
        # Calcola offset dinamico basato sulla larghezza
        dynamic_offset = int(lane_width * LANE_OFFSET_RATIO)
        offset = np.clip(dynamic_offset, MIN_OFFSET_PX, MAX_OFFSET_PX)
        
        # Target al centro della carreggiata
        target_x = lane_center
        
    elif left_valid:
        # Solo linea sinistra visibile - stima la linea destra
        estimated_right_base = left_base + ESTIMATED_LANE_WIDTH
        lane_center = (left_base + estimated_right_base) // 2
        lane_width = ESTIMATED_LANE_WIDTH
        
        # Calcola offset dinamico
        dynamic_offset = int(lane_width * LANE_OFFSET_RATIO)
        offset = np.clip(dynamic_offset, MIN_OFFSET_PX, MAX_OFFSET_PX)
        
        # Target al centro della carreggiata stimata
        target_x = lane_center
        
    elif right_valid:
        # Solo linea destra visibile - stima la linea sinistra
        estimated_left_base = right_base - ESTIMATED_LANE_WIDTH
        lane_center = (estimated_left_base + right_base) // 2
        lane_width = ESTIMATED_LANE_WIDTH
        
        # Calcola offset dinamico
        dynamic_offset = int(lane_width * LANE_OFFSET_RATIO)
        offset = np.clip(dynamic_offset, MIN_OFFSET_PX, MAX_OFFSET_PX)
        
        # Target al centro della carreggiata stimata
        target_x = lane_center
        
    else:
        # Nessuna linea visibile - usa centro immagine
        lane_center = w // 2
        lane_width = ESTIMATED_LANE_WIDTH
        offset = 0
        target_x = lane_center
    
    return lane_center, target_x, offset

def draw_lane_triangle(img, left_line, right_line, left_valid, right_valid, invM, color=(0, 255, 0), alpha=0.3, estimated=False):
    """Disegna il triangolo della carreggiata sul frame originale"""
    h_bev = 200  # Altezza del BEV
    
    # Se dobbiamo stimare le linee mancanti
    if estimated:
        if left_valid and not right_valid:
            # Stima linea destra basandosi sulla sinistra
            lx1, ly1, lx2, ly2 = left_line
            rx1 = lx1 + ESTIMATED_LANE_WIDTH
            ry1 = ly1
            rx2 = lx2 + ESTIMATED_LANE_WIDTH
            ry2 = ly2
            right_line = (rx1, ry1, rx2, ry2)
            right_valid = True
            
        elif right_valid and not left_valid:
            # Stima linea sinistra basandosi sulla destra
            rx1, ry1, rx2, ry2 = right_line
            lx1 = rx1 - ESTIMATED_LANE_WIDTH
            ly1 = ry1
            lx2 = rx2 - ESTIMATED_LANE_WIDTH
            ly2 = ry2
            left_line = (lx1, ly1, lx2, ly2)
            left_valid = True
    
    if not (left_valid or right_valid):
        return img
    
    # Coordinate nel BEV
    lx1, ly1, lx2, ly2 = left_line
    rx1, ry1, rx2, ry2 = right_line
    
    # Crea punti del triangolo nel BEV
    pts_bev = np.array([[lx1, ly1], [lx2, ly2], [rx2, ry2], [rx1, ry1]], dtype=np.float32).reshape(-1, 1, 2)
    
    # Trasforma nel frame originale
    pts_orig = cv2.perspectiveTransform(pts_bev, invM).reshape(-1, 2)
    
    # Disegna il triangolo riempito
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(pts_orig, dtype=np.int32)], color)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    return img

def draw_center_and_target(img, lane_center, target_x, invM, bev_shape):
    """Disegna il centro della carreggiata e il target sul frame originale"""
    h_bev = bev_shape[0]
    base_y = h_bev - 1
    
    # Punti nel BEV
    center_bev = np.array([[[lane_center, base_y]]], dtype=np.float32)
    target_bev = np.array([[[target_x, base_y]]], dtype=np.float32)
    
    # Trasforma nel frame originale
    center_orig = cv2.perspectiveTransform(center_bev, invM).reshape(-1, 2)[0]
    target_orig = cv2.perspectiveTransform(target_bev, invM).reshape(-1, 2)[0]
    
    # Disegna centro carreggiata (cerchio blu)
    cv2.circle(img, tuple(center_orig.astype(int)), 8, (255, 0, 0), -1)
    cv2.putText(img, "Centro", tuple((center_orig + [0, -20]).astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Disegna target (cerchio verde)
    cv2.circle(img, tuple(target_orig.astype(int)), 8, (0, 255, 0), -1)
    cv2.putText(img, "Target", tuple((target_orig + [0, -20]).astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Linea che collega centro e target
    cv2.line(img, tuple(center_orig.astype(int)), tuple(target_orig.astype(int)), (0, 255, 255), 2)
    
    return img

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
    global BASE_SPEED, last_valid_lane
    
    """Processa un singolo frame dalla camera"""
    global previous_steering_angle, ignored_count
    
    # Cattura frame dalla Picamera2
    frame = picam2.capture_array()
    
    if frame is None:
        return False
    
    h, w = frame.shape[:2]
    roi_top, roi_bot = int(h * 0.6), h 

    # Bird's Eye View per calcolo carreggiata
    src = np.float32([[0, roi_top], [w, roi_top], [w, roi_bot], [0, roi_bot]])
    dst = np.float32([[0, 0], [300, 0], [300, 200], [0, 200]])
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    bev = cv2.warpPerspective(frame, M, (300, 200))

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

    # Calcola larghezza carreggiata per verifiche
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

    # Determina se usare la carreggiata corrente o l'ultima valida
    use_current_lane = not ignored_reasons and (lv or rv)
    
    if use_current_lane:
        # Salva la carreggiata corrente come ultima valida
        last_valid_lane = {
            'left_line': left_line,
            'right_line': right_line,
            'left_valid': lv,
            'right_valid': rv
        }
        display_left_line = left_line
        display_right_line = right_line
        display_lv, display_rv = lv, rv
    else:
        # Usa l'ultima carreggiata valida se disponibile
        if last_valid_lane:
            display_left_line = last_valid_lane['left_line']
            display_right_line = last_valid_lane['right_line']
            display_lv = last_valid_lane['left_valid']
            display_rv = last_valid_lane['right_valid']
        else:
            display_left_line = left_line
            display_right_line = right_line
            display_lv, display_rv = lv, rv

    # Calcola centro e target
    lane_center, target_x, offset = calculate_lane_center_and_offset(
        display_left_line, display_right_line, display_lv, display_rv, bev.shape
    )

    # Calcola angolo di sterzo basato sulla deviazione dal target
    bev_center = bev.shape[1] // 2
    deviation = target_x - bev_center
    
    if ignored_reasons and not last_valid_lane:
        target_steering_angle = previous_steering_angle
    else:
        # Normalizza e mappa in angolo di sterzo
        target_steering_angle = -np.clip(deviation / bev_center, -1.0, 1.0) * MAX_STEERING_ANGLE

    # Applica smoothing graduale per sterzate più dolci
    steering_angle = smooth_steering_angle(target_steering_angle, previous_steering_angle)
    
    previous_steering_angle = steering_angle

    set_steering_angle(steering_angle)

    # Visualizzazione risultati
    # Determina se stiamo stimando una linea
    needs_estimation = (display_lv and not display_rv) or (display_rv and not display_lv)
    
    # Disegna triangolo della carreggiata
    if display_lv or display_rv:
        triangle_color = (0, 255, 0) if use_current_lane else (0, 255, 255)  # Verde se corrente, giallo se storica
        if needs_estimation:
            triangle_color = (255, 0, 255)  # Magenta per carreggiata stimata
        
        frame = draw_lane_triangle(frame, display_left_line, display_right_line, 
                                 display_lv, display_rv, invM, triangle_color, estimated=needs_estimation)
    
    # Disegna linee della carreggiata (incluse quelle stimate)
    if needs_estimation:
        # Genera le linee stimate per la visualizzazione
        if display_lv and not display_rv:
            # Stima linea destra
            lx1, ly1, lx2, ly2 = display_left_line
            estimated_right_line = (lx1 + ESTIMATED_LANE_WIDTH, ly1, lx2 + ESTIMATED_LANE_WIDTH, ly2)
            display_right_line = estimated_right_line
            display_rv = True
        elif display_rv and not display_lv:
            # Stima linea sinistra
            rx1, ry1, rx2, ry2 = display_right_line
            estimated_left_line = (rx1 - ESTIMATED_LANE_WIDTH, ry1, rx2 - ESTIMATED_LANE_WIDTH, ry2)
            display_left_line = estimated_left_line
            display_lv = True
    
    pts = np.array([[display_left_line[0], display_left_line[1]], 
                   [display_left_line[2], display_left_line[3]], 
                   [display_right_line[2], display_right_line[3]], 
                   [display_right_line[0], display_right_line[1]]], dtype=np.float32).reshape(-1,1,2)
    pts_orig = cv2.perspectiveTransform(pts, invM).reshape(-1,2)
    l1, l2, r2, r1 = pts_orig
    
    # Disegna linee con colori diversi
    if needs_estimation:
        # Colori per linee stimate
        if bool(left_ls):  # Se la linea sinistra è reale
            left_color = (255, 0, 0)      # Blu per linea reale
            right_color = (255, 0, 255)   # Magenta per linea stimata
        else:  # Se la linea destra è reale
            left_color = (255, 0, 255)    # Magenta per linea stimata
            right_color = (0, 0, 255)     # Rosso per linea reale
    else:
        # Colori normali
        left_color = (255, 0, 0) if display_lv else (128, 128, 128)  # Blu se valida, grigio se no
        right_color = (0, 0, 255) if display_rv else (128, 128, 128)  # Rosso se valida, grigio se no
        
        if not use_current_lane:
            left_color = (0, 255, 255)  # Giallo per linee storiche
            right_color = (0, 255, 255)
    
    cv2.line(frame, tuple(l1.astype(int)), tuple(l2.astype(int)), left_color, 4)
    cv2.line(frame, tuple(r1.astype(int)), tuple(r2.astype(int)), right_color, 4)
    
    # Disegna centro e target
    frame = draw_center_and_target(frame, lane_center, target_x, invM, bev.shape)

    # Aggiunge informazioni testuali
    cv2.putText(frame, f"Larghezza: {lane_width}px", (10, h-130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Velocita: {BASE_SPEED}px/s", (10, h-160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Offset: {offset}px", (10, h-190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Target: {target_steering_angle:.1f}° -> Smooth: {steering_angle:.1f}°", (10, h-100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    
    if ignored_reasons:
        text = ", ".join(ignored_reasons)
        cv2.putText(frame, f"Ignorato: {text}", (10, h-70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    if not use_current_lane and last_valid_lane:
        cv2.putText(frame, "Usando carreggiata storica", (10, h-40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    if needs_estimation:
        estimation_text = "Stimando linea " + ("destra" if bool(left_ls) else "sinistra")
        cv2.putText(frame, estimation_text, (10, h-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
    
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