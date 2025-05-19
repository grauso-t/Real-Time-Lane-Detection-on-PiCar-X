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
width, height = picam2.stream_configuration["main"]["size"]
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
                if abs(slope)<0.3: continue
                if slope < 0:
                    left_segs.append((x1,y1,x2,y2))
                else:
                    right_segs.append((x1,y1,x2,y2))

        left_avg  = avg_line(left_segs)  or (0,roi_h,0,0)
        right_avg = avg_line(right_segs) or (roi_w-1,roi_h,roi_w-1,0)

        # Deviazione dal centro
        center_line = (left_avg[0] + right_avg[0]) // 2
        deviation   = center_line - (roi_w // 2)

        # PID semplice: proporzionale
        k = 0.1  # coefficiente di guadagno (aumenta per sterzate piÃ¹ aggressive)
        angle = int(np.clip(k * deviation, -40, 40))
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
        cv2.fillPoly(fill, [pts], (0,255,255))
        bird_vis = cv2.addWeighted(overlay, 1, fill, 0.3, 0)

        back = cv2.warpPerspective(bird_vis, Minv, (roi_w, roi_h))
        out = frame.copy()
        roi_area = out[y0:y0+roi_h, x0:x0+roi_w]
        out[y0:y0+roi_h, x0:x0+roi_w] = cv2.addWeighted(roi_area, 0.8, back, 1, 0)

        cv2.imshow('Frame con Lanes', out)
        cv2.imshow('Birdâ€™s-eye', bird)
        cv2.imshow('Edges', edges)

        if cv2.waitKey(frame_time) & 0xFF == ord('q'):
            break

finally:
    print("Pulizia in corso...")
    picam2.stop()
    px.stop()
    px.set_dir_servo_angle(0)
    cv2.destroyAllWindows()