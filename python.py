import cv2
import numpy as np
from picarx import Picarx
from time import sleep
from picamera2 import Picamera2

# Inizializza la camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
sleep(2)

# Inizializza Picarx
px = Picarx()
px.set_dir_servo_angle(0)

# Ottieni le dimensioni del frame
frame = picam2.capture_array()
h, w = frame.shape[:2]

# Definisci i punti per la ROI (Region of Interest)
roi_points = np.array([
    [0, h],
    [w//3, h//2],
    [2*w//3, h//2],
    [w, h]
], dtype=np.int32)

# Definisci i punti per la trasformazione Bird's Eye
src_points = np.float32([
    [w//3, h//2],    # Top-left della ROI
    [2*w//3, h//2],  # Top-right della ROI
    [w, h],          # Bottom-right della ROI
    [0, h]           # Bottom-left della ROI
])

# Punti di destinazione per la trasformazione bird's eye
dst_points = np.float32([[0, 0], [400, 0], [400, 600], [0, 600]])
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
output_size = (400, 600)

prev_left = None
prev_right = None

def separa_linee(lines, width):
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.5:
            left_lines.append(line[0])
        elif slope > 0.5:
            right_lines.append(line[0])
    return left_lines, right_lines

def media_linea(linee):
    if len(linee) == 0:
        return None
    x = []
    y = []
    for x1, y1, x2, y2 in linee:
        x += [x1, x2]
        y += [y1, y2]
    poly = np.polyfit(y, x, 1)
    y1, y2 = 600, 300
    x1, x2 = int(poly[0]*y1 + poly[1]), int(poly[0]*y2 + poly[1])
    return (x1, y1, x2, y2)

def accorcia_linea(linea, percentuale=0.3):
    if linea is None:
        return None
    x1, y1, x2, y2 = linea
    x_mid = int(x1 + (x2 - x1) * percentuale)
    y_mid = int(y1 + (y2 - y1) * percentuale)
    return (x1, y1, x_mid, y_mid)

def rileva_corsie(frame, prev_left, prev_right):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white_hls = np.array([0, 200, 0])
    upper_white_hls = np.array([180, 255, 255])
    mask_white_hls = cv2.inRange(hls, lower_white_hls, upper_white_hls)

    lower_yellow_hsv = np.array([15, 80, 80])
    upper_yellow_hsv = np.array([40, 255, 255])
    mask_yellow_hsv = cv2.inRange(hsv, lower_yellow_hsv, upper_yellow_hsv)

    combined_mask = cv2.bitwise_or(mask_white_hls, mask_yellow_hsv)

    blur = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Non applichiamo la ROI qui, poiché è già stata applicata prima
    masked_edges = edges

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=50)

    line_image = np.zeros_like(frame)
    left_line, right_line = None, None

    if lines is not None:
        left_lines, right_lines = separa_linee(lines, output_size[0])
        left_line = media_linea(left_lines)
        right_line = media_linea(right_lines)

    if left_line is None and prev_left is not None:
        left_line = accorcia_linea(prev_left, 0.3)
    if right_line is None and prev_right is not None:
        right_line = accorcia_linea(prev_right, 0.3)

    if left_line:
        cv2.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 5)
    if right_line:
        cv2.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 5)

    if left_line and right_line:
        punti = np.array([[ (left_line[0], left_line[1]), (left_line[2], left_line[3]),
                            (right_line[2], right_line[3]), (right_line[0], right_line[1]) ]], dtype=np.int32)
        cv2.fillPoly(line_image, punti, (0, 255, 255))

    return line_image, left_line, right_line, mask_white_hls, mask_yellow_hsv, combined_mask, edges

# Loop principale
while True:
    # Cattura il frame
    frame = picam2.capture_array()
    frame_with_roi = frame.copy()
    
    # Disegna la ROI sul frame originale per visualizzazione
    cv2.polylines(frame_with_roi, [roi_points], True, (0, 0, 255), 2)
    
    # Crea una maschera per la ROI
    roi_mask = np.zeros_like(frame[:,:,0])
    cv2.fillPoly(roi_mask, [roi_points], 255)
    
    # Applica la maschera ROI all'immagine originale
    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
    
    # Applica la trasformazione bird's eye alla ROI
    bird_eye = cv2.warpPerspective(masked_frame, matrix, output_size)
    
    # Applica il rilevamento delle corsie sulla vista bird's eye
    overlay, prev_left, prev_right, white_mask, yellow_mask, combined_mask, edges = rileva_corsie(
        bird_eye, prev_left, prev_right)
    
    # Combina bird's eye con overlay
    combined = cv2.addWeighted(bird_eye, 1.0, overlay, 0.7, 0)

    # Logica di steering
    max_angle = 30
    max_offset = output_size[0] // 2

    if prev_left is not None and prev_right is not None:
        centro_strada = (prev_left[2] + prev_right[2]) // 2
        centro_frame = output_size[0] // 2
        offset = centro_strada - centro_frame

        angle = int((offset / max_offset) * max_angle)
        angle = np.clip(angle, -max_angle, max_angle)

        print(f"Offset: {offset} px -> Angolo sterzo: {angle}°")
        px.set_dir_servo_angle(angle)
        px.forward(20)

    elif prev_left is not None or prev_right is not None:
        print("Una linea visibile. Riduzione sterzata.")
        centro_frame = output_size[0] // 2

        if prev_left is not None:
            offset = (prev_left[2] + 50) - centro_frame
        else:
            offset = (prev_right[2] - 50) - centro_frame

        angle = int((offset / max_offset) * max_angle)
        angle = np.clip(angle, -15, 15)

        print(f"Offset stimato: {offset} -> Angolo limitato: {angle}")
        px.set_dir_servo_angle(angle)
        px.forward(10)

    else:
        print("Linee non rilevate. Stop.")
        px.stop()

    # Visualizzazione
    cv2.imshow("Vista originale", frame)
    cv2.imshow("Regione di interesse", frame_with_roi)
    cv2.imshow("ROI applicata", masked_frame)
    cv2.imshow("Bird Eye View", bird_eye)
    cv2.imshow("Edge Detection", edges)
    cv2.imshow("Maschera Bianca (HLS)", white_mask)
    cv2.imshow("Maschera Gialla (HSV)", yellow_mask)
    cv2.imshow("Maschera Combinata", combined_mask)
    cv2.imshow("Corsie rilevate", overlay)
    cv2.imshow("Vista con sovrapposizione", combined)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Arresto finale
px.stop()
picam2.close()
cv2.destroyAllWindows()