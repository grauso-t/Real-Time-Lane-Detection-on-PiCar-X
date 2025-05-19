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
width = 480
height = 640
roi_w = width
roi_h = height // 4
x0, y0 = 0, height - roi_h
dst_pts = np.float32([
    [0, roi_h],
    [roi_w, roi_h],
    [roi_w, 0],
    [0, 0]
])

def avg_line(segs):
    if not segs: return None
    xs, ys = [], []
    for x1,y1,x2,y2 in segs:
        xs += [x1,x2]
        ys += [y1,y2]
    m, q = np.polyfit(ys, xs, 1)
    return (int(m*roi_h + q), roi_h, int(q), 0)

try:
    while True:
        frame = picam2.capture_array()
        roi = frame[y0:y0 + roi_h, x0:x0 + roi_w]

        # Adatta in base alla tua camera
        src_pts = np.float32([
            [0, roi_h],
            [roi_w, roi_h],
            [roi_w, 20],
            [20, 0]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        bird = cv2.warpPerspective(roi, M, (roi_w, roi_h))

        # Maschere colore
        hls = cv2.cvtColor(bird, cv2.COLOR_BGR2HLS)
        mask_w = cv2.inRange(hls, np.array([0,200,0],np.uint8), np.array([180,255,80],np.uint8))
        hsv = cv2.cvtColor(bird, cv2.COLOR_BGR2HSV)
        mask_y = cv2.inRange(hsv, np.array([18,80,80],np.uint8), np.array([35,255,255],np.uint8))
        mask = cv2.bitwise_or(mask_w, mask_y)
        filtered = cv2.bitwise_and(bird, bird, mask=mask)

        # Edges
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Linee
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=50)
        left_segs, right_segs = [], []

        if lines is not None:
            for l in lines:
                x1,y1,x2,y2 = l[0]
                if x2==x1: continue
                slope = (y2-y1)/(x2-x1)
                
                # Ignora linee con pendenza troppo piccola (orizzontali)
                if abs(slope)<0.3: continue
                
                # Ignora segmenti troppo corti
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length < 25: continue  # Aumentato soglia minima lunghezza
                
                if slope < 0:
                    left_segs.append((x1,y1,x2,y2))
                else:
                    right_segs.append((x1,y1,x2,y2))

        left_avg  = avg_line(left_segs)  or (0,roi_h,0,0)
        right_avg = avg_line(right_segs) or (roi_w-1,roi_h,roi_w-1,0)

        # Calcola la larghezza della carreggiata
        lane_width = right_avg[0] - left_avg[0]  # Larghezza in basso
        lane_width_top = right_avg[2] - left_avg[2]  # Larghezza in alto
        
        # Controlla che la carreggiata non sia troppo stretta
        min_lane_width = roi_w * 0.3  # La carreggiata deve essere almeno il 30% della larghezza totale
        
        valid_lane = lane_width > min_lane_width and lane_width_top > min_lane_width * 0.5
        
        # Variabile per mantenere l'angolo precedente (definita fuori dal loop)
        try:
            previous_angle
        except NameError:
            previous_angle = 0
        
        # Deviazione dal centro solo se la carreggiata è valida
        if valid_lane:
            center_line = (left_avg[0] + right_avg[0]) // 2
            deviation = center_line - (roi_w // 2)
            
            # PID semplice: proporzionale
            k = 0.4  # coefficiente di guadagno (aumenta per sterzate più aggressive)
            angle = int(np.clip(k * deviation, -40, 40))
            previous_angle = angle  # Memorizza l'angolo valido per il prossimo frame
        else:
            # Se la carreggiata non è valida, mantieni la direzione precedente
            angle = previous_angle
            
        px.set_dir_servo_angle(angle)

        # Visualizzazione
        overlay = np.zeros_like(bird)
        for x1,y1,x2,y2 in left_segs:
            cv2.line(overlay, (x1,y1),(x2,y2),(255,100,100),2)
        for x1,y1,x2,y2 in right_segs:
            cv2.line(overlay, (x1,y1),(x2,y2),(100,255,100),2)
        cv2.line(overlay, (left_avg[0],left_avg[1]),(left_avg[2],left_avg[3]),(255,0,0),4)
        cv2.line(overlay, (right_avg[0],right_avg[1]),(right_avg[2],right_avg[3]),(0,255,0),4)

        pts = np.array([[left_avg[0],left_avg[1]],
                        [left_avg[2],left_avg[3]],
                        [right_avg[2],right_avg[3]],
                        [right_avg[0],right_avg[1]]], np.int32)
        fill = np.zeros_like(bird)
        
        # Colora la carreggiata in base alla validità
        fill_color = (0,255,255) if valid_lane else (0,0,255)  # Giallo se valida, rosso se troppo stretta
        cv2.fillPoly(fill, [pts], fill_color)
        
        bird_vis = cv2.addWeighted(overlay, 1, fill, 0.3, 0)

        # Aggiungi indicatore di stato
        status_text = "Carreggiata: Valida" if valid_lane else "Carreggiata: Invalida (uso angolo precedente)"
        cv2.putText(bird_vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(bird_vis, f"Larghezza: {lane_width:.1f}px", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(bird_vis, f"Angolo: {angle}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        back = cv2.warpPerspective(bird_vis, Minv, (roi_w, roi_h))
        out = frame.copy()
        roi_area = out[y0:y0+roi_h, x0:x0+roi_w]
        out[y0:y0+roi_h, x0:x0+roi_w] = cv2.addWeighted(roi_area, 0.8, back, 1, 0)

        cv2.imshow('Frame con Lanes', out)
        cv2.imshow('Bird\'s-eye', bird_vis)
        cv2.imshow('Edges', edges)

        if cv2.waitKey(frame_time) & 0xFF == ord('q'):
            break

finally:
    print("Pulizia in corso...")
    picam2.stop()
    px.stop()
    px.set_dir_servo_angle(0)
    cv2.destroyAllWindows()