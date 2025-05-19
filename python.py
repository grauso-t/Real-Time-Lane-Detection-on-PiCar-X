import cv2
import numpy as np
from picamera2 import Picamera2
from picarx import Picarx
import time

# Inizializza Picarx
px = Picarx()
px.set_dir_servo_angle(0)
px.forward(10)

# Configura e avvia Picamera2
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    lores={"size": (320, 240)},
    display="lores"
)
picam2.configure(preview_config)
picam2.start()
print("Picamera2 avviata, inizio acquisizione...")

# Parametri
fps = 24
frame_time = int(1000 / fps)
width = 640
height = 480
roi_w = width
roi_h = height // 4
x0, y0 = 0, height - roi_h

# Parametri PID per il controllo dello sterzo
kp = 0.4  # Proporzionale
ki = 0.02  # Integrativo
kd = 0.2   # Derivativo

# Variabili per PID
integral = 0
prev_error = 0
max_integral = 100  # Anti-windup

# Storia degli angoli per smoothing e fallback
angle_history = [0] * 5  # Mantiene gli ultimi 5 angoli validi

def avg_line(segs):
    if not segs: return None
    xs, ys = [], []
    for x1,y1,x2,y2 in segs:
        xs += [x1,x2]
        ys += [y1,y2]
    
    # Se ci sono troppo pochi punti, non è affidabile
    if len(xs) < 3:
        return None
        
    try:
        m, q = np.polyfit(ys, xs, 1)
        return (int(m*roi_h + q), roi_h, int(q), 0)
    except:
        return None  # In caso di errore di fitting

def calculate_steering_angle(deviation, current_speed):
    """Calcola l'angolo di sterzo usando PID e adattandolo alla velocità"""
    global integral, prev_error
    
    # PID control
    integral = np.clip(integral + deviation, -max_integral, max_integral)
    derivative = deviation - prev_error
    
    # Velocità-dipendente: più veloce = sterzata più aggressiva in curva
    speed_factor = max(1.0, current_speed / 10.0)
    
    angle = kp * deviation + ki * integral + kd * derivative
    angle = int(np.clip(angle * speed_factor, -40, 40))
    
    prev_error = deviation
    return angle

def smooth_angle(new_angle):
    """Applica smoothing all'angolo per evitare sterzate brusche"""
    global angle_history
    angle_history.pop(0)
    angle_history.append(new_angle)
    return int(sum(angle_history) / len(angle_history))

def get_dominant_side_in_curve(left_segs, right_segs):
    """Determina quale lato ha linee più affidabili in una curva"""
    if not left_segs and not right_segs:
        return None
    
    left_quality = sum(np.sqrt((x2-x1)**2 + (y2-y1)**2) for x1,y1,x2,y2 in left_segs)
    right_quality = sum(np.sqrt((x2-x1)**2 + (y2-y1)**2) for x1,y1,x2,y2 in right_segs)
    
    # Pesi: lunghezza totale delle linee su ciascun lato
    if left_quality > right_quality * 1.5:
        return "left"
    elif right_quality > left_quality * 1.5:
        return "right"
    else:
        return "both"

try:
    # Velocità iniziale
    current_speed = 10
    
    while True:
        start_time = time.time()
        frame = picam2.capture_array()
        
        # Verifica dimensioni del frame
        actual_height, actual_width = frame.shape[:2]
        if actual_height != height or actual_width != width:
            print(f"ATTENZIONE: Dimensioni frame ({actual_width}x{actual_height}) diverse da quelle attese ({width}x{height})")
            height, width = actual_height, actual_width
        
        roi_h = height // 4
        roi_w = width
        x0, y0 = 0, height - roi_h
        
        # Controllo dimensioni ROI
        if y0 < 0 or y0 + roi_h > height or x0 < 0 or x0 + roi_w > width:
            print("ERRORE: ROI fuori dai limiti del frame!")
            y0 = max(0, min(height - roi_h, y0))
            x0 = max(0, min(width - roi_w, x0))
            
        roi = frame[y0:y0 + roi_h, x0:x0 + roi_w]

        # Bird's eye view transformation
        src_pts = np.float32([
            [0, roi_h],
            [roi_w, roi_h],
            [roi_w * 0.65, roi_h * 0.35],
            [roi_w * 0.35, roi_h * 0.35]
        ])
        
        dst_pts = np.float32([
            [0, roi_h],
            [roi_w, roi_h],
            [roi_w, 0],
            [0, 0]
        ])
        
        try:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
            bird = cv2.warpPerspective(roi, M, (roi_w, roi_h))
        except Exception as e:
            print(f"Errore nella trasformazione prospettica: {e}")
            continue

        # Maschere colore - Aumentata la sensibilità per rilevare meglio le linee
        hls = cv2.cvtColor(bird, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(bird, cv2.COLOR_BGR2HSV)
        
        # Maschera per linee bianche - resa più sensibile
        mask_w = cv2.inRange(hls, np.array([0,180,0],np.uint8), np.array([180,255,90],np.uint8))
        
        # Maschera per linee gialle - resa più sensibile
        mask_y = cv2.inRange(hsv, np.array([15,70,70],np.uint8), np.array([40,255,255],np.uint8))
        
        mask = cv2.bitwise_or(mask_w, mask_y)
        filtered = cv2.bitwise_and(bird, bird, mask=mask)

        # Miglioramento del rilevamento bordi
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(gray)
        blur = cv2.GaussianBlur(cl1, (5,5), 0)
        edges = cv2.Canny(blur, 40, 150)  # Parametri più sensibili

        # Dilatazione per connettere bordi vicini
        kernel = np.ones((3,3),np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Rilevamento linee con parametri migliorati
        lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=25, minLineLength=15, maxLineGap=100)
        left_segs, right_segs = [], []

        if lines is not None:
            for l in lines:
                x1,y1,x2,y2 = l[0]
                if x2==x1: continue
                slope = (y2-y1)/(x2-x1)
                
                # Filtraggio migliorato per le linee
                if abs(slope) < 0.2: continue  # Ignora linee quasi orizzontali
                
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length < 15: continue
                
                # Classiifica le linee
                if slope < 0:
                    left_segs.append((x1,y1,x2,y2))
                else:
                    right_segs.append((x1,y1,x2,y2))

        # Determina quale lato è dominante in una curva
        dominant_side = get_dominant_side_in_curve(left_segs, right_segs)

        # Ottieni linee medie per ciascun lato
        left_avg = avg_line(left_segs)
        right_avg = avg_line(right_segs)

        # Logica per gestire le curve strette
        is_curve = False
        if dominant_side == "left" and not right_avg:
            # Curva a sinistra stretta, solo la linea sinistra è visibile
            is_curve = True
            # Stima la posizione della linea destra basandosi sulla larghezza tipica della carreggiata
            if left_avg:
                estimated_lane_width = roi_w * 0.6  # Larghezza stimata della carreggiata
                right_x_bottom = min(roi_w-1, left_avg[0] + estimated_lane_width)
                right_x_top = min(roi_w-1, left_avg[2] + estimated_lane_width * 0.8)  # Più stretta in alto
                right_avg = (right_x_bottom, roi_h, right_x_top, 0)
                
        elif dominant_side == "right" and not left_avg:
            # Curva a destra stretta, solo la linea destra è visibile
            is_curve = True
            # Stima la posizione della linea sinistra
            if right_avg:
                estimated_lane_width = roi_w * 0.6
                left_x_bottom = max(0, right_avg[0] - estimated_lane_width)
                left_x_top = max(0, right_avg[2] - estimated_lane_width * 0.8)
                left_avg = (left_x_bottom, roi_h, left_x_top, 0)
                
        # Se ancora non abbiamo linee valide, usa valori di fallback
        if not left_avg:
            left_avg = (0, roi_h, 0, 0)
        if not right_avg:
            right_avg = (roi_w-1, roi_h, roi_w-1, 0)

        # Calcola la larghezza della carreggiata
        lane_width = right_avg[0] - left_avg[0]
        lane_width_top = right_avg[2] - left_avg[2]
        
        # Validazione della carreggiata con parametri più flessibili
        min_lane_width = roi_w * 0.2  # Più permissivo (era 0.3)
        max_lane_width = roi_w * 0.9  # Massima larghezza ragionevole
        
        # Una carreggiata è valida se è all'interno di limiti ragionevoli
        valid_lane = (min_lane_width < lane_width < max_lane_width and 
                     min_lane_width * 0.3 < lane_width_top < max_lane_width)
        
        # Calcola il centro della carreggiata e la deviazione
        center_line = (left_avg[0] + right_avg[0]) // 2
        deviation = center_line - (roi_w // 2)
        
        # Calcola l'angolo di sterzo
        if valid_lane or is_curve:
            # Se siamo in una curva, aumenta l'aggressività dello sterzo
            curve_factor = 1.5 if is_curve else 1.0
            angle = calculate_steering_angle(deviation * curve_factor, current_speed)
            
            # Applica smoothing per evitare sterzate brusche
            angle = smooth_angle(angle)
            
            # In curva stretta, riduci leggermente la velocità
            if is_curve:
                target_speed = max(5, current_speed - 2)
                current_speed = current_speed * 0.8 + target_speed * 0.2  # Smooth transition
                px.forward(current_speed)
            else:
                # Se non in curva, aumenta gradualmente fino alla velocità normale
                current_speed = min(10, current_speed + 0.5)
                px.forward(current_speed)
        else:
            # Se la carreggiata non è valida, usa l'angolo medio recente
            angle = sum(angle_history) // len(angle_history)
            
            # Riduci la velocità se c'è incertezza
            current_speed = max(5, current_speed * 0.9)
            px.forward(current_speed)
        
        # Applica l'angolo di sterzo
        px.set_dir_servo_angle(angle)

        # Visualizzazione
        overlay = np.zeros_like(bird)
        for x1,y1,x2,y2 in left_segs:
            cv2.line(overlay, (x1,y1),(x2,y2),(255,100,100),2)
        for x1,y1,x2,y2 in right_segs:
            cv2.line(overlay, (x1,y1),(x2,y2),(100,255,100),2)
            
        # Visualizza le linee medie
        cv2.line(overlay, (left_avg[0],left_avg[1]),(left_avg[2],left_avg[3]),(255,0,0),4)
        cv2.line(overlay, (right_avg[0],right_avg[1]),(right_avg[2],right_avg[3]),(0,255,0),4)

        # Area della carreggiata
        pts = np.array([[left_avg[0],left_avg[1]],
                        [left_avg[2],left_avg[3]],
                        [right_avg[2],right_avg[3]],
                        [right_avg[0],right_avg[1]]], np.int32)
        fill = np.zeros_like(bird)
        
        # Colora la carreggiata in base alla validità
        if is_curve:
            fill_color = (255,165,0)  # Arancione per curve
        elif valid_lane:
            fill_color = (0,255,255)  # Giallo se valida
        else:
            fill_color = (0,0,255)    # Rosso se non valida
            
        cv2.fillPoly(fill, [pts], fill_color)
        bird_vis = cv2.addWeighted(overlay, 1, fill, 0.3, 0)

        # Aggiungi indicatori di stato
        if is_curve:
            status_text = f"Curva: {dominant_side.upper()}"
        elif valid_lane:
            status_text = "Carreggiata: Valida"
        else:
            status_text = "Carreggiata: Invalida"
            
        cv2.putText(bird_vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(bird_vis, f"Larghezza: {lane_width:.1f}px", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(bird_vis, f"Angolo: {angle}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(bird_vis, f"Velocità: {current_speed:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Visualizza centro della carreggiata e punto centrale ideale
        cv2.circle(bird_vis, (center_line, roi_h//2), 5, (255,0,255), -1)  # Centro carreggiata
        cv2.circle(bird_vis, (roi_w//2, roi_h//2), 5, (0,255,255), -1)     # Centro ideale
        cv2.line(bird_vis, (roi_w//2, roi_h//2), (center_line, roi_h//2), (255,255,0), 2)  # Deviazione

        try:
            # Visualizzazione su frame originale
            back = cv2.warpPerspective(bird_vis, Minv, (roi_w, roi_h))
            out = frame.copy()
            roi_area = out[y0:y0+roi_h, x0:x0+roi_w]
            out[y0:y0+roi_h, x0:x0+roi_w] = cv2.addWeighted(roi_area, 0.8, back, 1, 0)

            cv2.imshow('Frame con Lanes', out)
            cv2.imshow('Bird\'s-eye', bird_vis)
            cv2.imshow('Edges', dilated_edges)  # Visualizza bordi dilatati
            
            # Calcola FPS
            elapsed = time.time() - start_time
            fps_actual = 1 / elapsed if elapsed > 0 else 0
            cv2.putText(out, f"FPS: {fps_actual:.1f}", (width-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
        except Exception as e:
            print(f"Errore nella visualizzazione: {e}")
            continue

        if cv2.waitKey(frame_time) & 0xFF == ord('q'):
            break

finally:
    print("Pulizia in corso...")
    picam2.stop()
    px.stop()
    px.set_dir_servo_angle(0)
    cv2.destroyAllWindows()