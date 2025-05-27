import cv2
import numpy as np
import os
import time
from picamera2 import Picamera2
from picarx import Picarx

from time import sleep
from robot_hat import Music,TTS

# Costanti per simulazione
MAX_STEERING_ANGLE    = 45    # gradi
STEERING_AGGRESSION   = 1
MIN_LANE_WIDTH        = 120   # px, soglia per carreggiata piccola
MIN_CENTER_DISTANCE   = 30    # px, soglia distanza minima linea-centro
OUTPUT_DIR            = "./ignored_frames"

# Costanti per PicarX
BASE_SPEED           = 0     # velocità base del veicolo
STEERING_MULTIPLIER  = 10      # moltiplicatore per conversione angolo->servo

# NUOVE COSTANTI PER SMOOTH STEERING
SMOOTHING_FACTOR     = 0.3    # Range consigliato: 0.1-0.5 (più basso = più smooth, più alto = più reattivo)
MAX_STEERING_CHANGE  = 8.0    # gradi, massimo cambio di sterzo per frame (range: 5-15°)

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

music = Music()

flag_bgm = False
music.music_set_volume(20)

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

def apply_smooth_steering(target_angle):
    """Applica uno smoothing progressivo all'angolo di sterzo"""
    global previous_steering_angle
    
    # Calcola la differenza tra angolo target e precedente
    angle_diff = target_angle - previous_steering_angle
    
    # Limita il cambio massimo per frame
    if abs(angle_diff) > MAX_STEERING_CHANGE:
        angle_diff = np.sign(angle_diff) * MAX_STEERING_CHANGE
    
    # Applica smoothing usando media pesata
    smooth_angle = previous_steering_angle + (angle_diff * SMOOTHING_FACTOR)
    
    # Aggiorna l'angolo precedente
    previous_steering_angle = smooth_angle
    
    return smooth_angle

def apply_steering_to_picarx(steering_angle_deg):
    """Applica l'angolo di sterzo al PicarX"""
    # Converte l'angolo di sterzo in angolo servo
    servo_angle = steering_angle_deg * STEERING_MULTIPLIER
    servo_angle = np.clip(servo_angle, -45, 45)
    
    # Applica al servo di direzione
    px.set_dir_servo_angle(servo_angle)

def calculate_lane_center(left_line, right_line, bev_shape):
    """Calcola il centro della carreggiata dalla base delle linee"""
    h, w = bev_shape[:2]
    base_y = h - 1  # Linea di base (bottom del BEV)
    
    # Estrae coordinate delle linee
    lx1, ly1, lx2, ly2 = left_line
    rx1, ry1, rx2, ry2 = right_line
    
    # Calcola X alla base per entrambe le linee
    left_x_base = int(np.interp(base_y, [ly2, ly1], [lx2, lx1]))
    right_x_base = int(np.interp(base_y, [ry2, ry1], [rx2, rx1]))
    
    # Centro della carreggiata
    lane_center_x = (left_x_base + right_x_base) // 2
    
    return lane_center_x, left_x_base, right_x_base

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

    # Calcola angolo di sterzo TARGET (prima dello smoothing)
    target_steering_angle = 0.0
    
    if ignored_reasons:
        target_steering_angle = previous_steering_angle
        print(f"Frame ignorato: {', '.join(ignored_reasons)}")
    else:
        if lv and rv:
            # ENTRAMBE LE LINEE VISIBILI: CENTRA IL VEICOLO
            lane_center_x, left_x_base, right_x_base = calculate_lane_center(left_line, right_line, bev.shape)
            
            # Calcola deviazione dal centro del BEV
            bev_center_x = bev.shape[1] // 2
            deviation = lane_center_x - bev_center_x
            
            # Normalizza e converte in angolo di sterzo
            max_deviation = bev.shape[1] // 4  # massima deviazione accettabile
            normalized_deviation = np.clip(deviation / max_deviation, -1.0, 1.0)
            target_steering_angle = -normalized_deviation * MAX_STEERING_ANGLE * 0.5  # fattore ridotto per rettilinei
            
        elif lv or rv:
            # Solo una linea visibile: segui quella linea con offset
            OFFSET_PX = 20 # offset laterale desiderato

            if lv:
                # Calcolo offset rispetto alla linea sinistra
                base_x = int(np.interp(bev.shape[0] - 1, [ly2, ly1], [lx2, lx1]))
                offset_target = base_x + OFFSET_PX
            else:
                # Calcolo offset rispetto alla linea destra
                base_x = int(np.interp(bev.shape[0] - 1, [ry2, ry1], [rx2, rx1]))
                offset_target = base_x - OFFSET_PX

            # Calcolo deviazione da centro del BEV
            center_x = bev.shape[1] // 2
            dx = offset_target - center_x

            # Normalizza e mappa in angolo di sterzo
            target_steering_angle = -np.clip(dx / center_x, -1.0, 1.0) * MAX_STEERING_ANGLE

        else:
            # Nessuna linea: mantieni direzione precedente
            target_steering_angle = previous_steering_angle

    # APPLICA SMOOTHING ALLA STERZATA
    smooth_steering_angle = apply_smooth_steering(target_steering_angle)

    # Applica sterzo al PicarX
    apply_steering_to_picarx(smooth_steering_angle)

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

    # Disegna il centro della carreggiata se entrambe le linee sono visibili
    if lv and rv and not ignored_reasons:
        lane_center_x, _, _ = calculate_lane_center(left_line, right_line, bev.shape)
        # Trasforma il punto del centro dal BEV al frame originale
        center_bev = np.array([[lane_center_x, bev.shape[0]-1]], dtype=np.float32).reshape(-1,1,2)
        center_orig = cv2.perspectiveTransform(center_bev, invM).reshape(-1,2)[0]
        cv2.circle(frame, tuple(center_orig.astype(int)), 8, (255,255,0), -1)  # Cerchio giallo per il centro

    # Aggiunge informazioni testuali
    cv2.putText(frame, f"Larghezza: {lane_width}px", (10, h-130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Velocita: {BASE_SPEED}px/s", (10, h-160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Smooth Factor: {SMOOTHING_FACTOR:.1f}", (10, h-100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"Target: {target_steering_angle:.1f}° | Smooth: {smooth_steering_angle:.1f}°", (10, h-70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    
    if ignored_reasons:
        text = ", ".join(ignored_reasons)
        cv2.putText(frame, f"Ignorato: {text}", (10, h-40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    # Disegna traiettoria di sterzo (usa l'angolo smooth)
    draw_steering_trajectory(frame, -smooth_steering_angle)

    # Salva frame ignorati per debug
    if ignored_reasons:
        ignored_count += 1
        filename = f"frame_ignored_{ignored_count}_{int(time.time()*1000)}.png"
        #cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)

    global flag_bgm

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
        elif key == ord('m'):
            flag_bgm = not flag_bgm
            if flag_bgm:
                print("Musica di sottofondo attivata")
                music.sound_play_threading("sound.wav")
            else:
                print("Musica di sottofondo disattivata")
                music.music_stop()
    except:
        # Se non c'è display, ignora gli errori di visualizzazione
        pass
    
    return True

def main():
    """Funzione principale"""
    print("Avvio sistema di lane detection...")
    print("Premi 'q' per uscire, 'spazio' per pausa")
    print(f"Smoothing Factor: {SMOOTHING_FACTOR} (range consigliato: 0.1-0.5)")
    print(f"Max Steering Change: {MAX_STEERING_CHANGE}° (range consigliato: 5-15°)")
    
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