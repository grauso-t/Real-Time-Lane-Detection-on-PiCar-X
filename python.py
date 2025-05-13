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

frame = picam2.capture_array()
h, w = frame.shape[:2]
src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst_points = np.float32([[0, 0], [400, 0], [400, 600], [0, 600]])
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
output_size = (400, 600)

prev_left = None
prev_right = None

px = Picarx()
px.set_dir_servo_angle(0)

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
    h, w = frame.shape[:2]  # Prendi altezza e larghezza del frame
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

    mask = np.zeros_like(edges)
    
    # Definizione della ROI (25% inferiore dell'immagine)
    roi_corners = np.array([[
        (0, h),  # parte inferiore sinistra
        (0, int(h * 0.80)),  # 20% dal basso
        (w, int(h * 0.80)),  # 20% dal basso
        (w, h)  # parte inferiore destra
    ]], dtype=np.int32)

    # Disegna la ROI sul frame per visualizzarla
    cv2.polylines(frame, roi_corners, isClosed=True, color=(0, 255, 0), thickness=3)  # Linea verde per la ROI
    cv2.fillPoly(frame, roi_corners, (0, 255, 0))  # Colora la ROI in verde trasparente

    cv2.fillPoly(mask, roi_corners, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Esegui il rilevamento delle linee sulla Bird-eye View
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=150)

    line_image = np.zeros_like(frame)
    left_line, right_line = None, None

    if lines is not None:
        left_lines, right_lines = separa_linee(lines, w)
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

    return line_image, left_line, right_line, mask_white_hls, mask_yellow_hsv, combined_mask

# Loop principale
while True:
    frame = picam2.capture_array()
    bird_eye = cv2.warpPerspective(frame, matrix, output_size)

    # Applica rilevamento delle corsie
    overlay, prev_left, prev_right, white_mask, yellow_mask, combined_mask = rileva_corsie(
        bird_eye, prev_left, prev_right)
    
    combined = cv2.addWeighted(bird_eye, 1.0, overlay, 0.7, 0)

    max_angle = 30  # Angolo massimo consentito
    max_offset = output_size[0] // 2  # Offset massimo dal centro del frame

    if prev_left is not None and prev_right is not None:
        centro_strada = (prev_left[2] + prev_right[2]) // 2
        centro_frame = output_size[0] // 2
        offset = centro_strada - centro_frame

        # Calcola l'angolo in base all'offset
        angle = int((offset / max_offset) * max_angle)
        angle = np.clip(angle, -max_angle, max_angle)

        print(f"Offset: {offset} px -> Angolo sterzo: {angle}Â°")
        px.set_dir_servo_angle(angle)
        px.forward(20)

    elif prev_left is not None or prev_right is not None:
        print("Una linea visibile. Correzioni piÃ¹ aggressive.")
        centro_frame = output_size[0] // 2

        # Se solo una linea Ã¨ visibile, calcolare l'offset
        if prev_left is not None:
            offset = (prev_left[2] + 50) - centro_frame  # Aggiungi un piccolo margine per evitare errori
        else:
            offset = (prev_right[2] - 50) - centro_frame  # Aggiungi un piccolo margine per evitare errori

        # Calcola l'angolo di sterzo in modo dinamico
        angle = int((offset / max_offset) * max_angle)

        # Correzioni dinamiche sull'angolo in base alla distanza
        if abs(offset) > max_offset // 2:  # Se l'offset Ã¨ piÃ¹ grande della metÃ  della larghezza
            angle = np.clip(angle, -max_angle, max_angle)  # Permetti correzioni piÃ¹ ampie
        else:
            angle = np.clip(angle, -20, 20)  # Se l'offset Ã¨ ridotto, limita l'angolo per evitare movimenti troppo piccoli

        print(f"Offset stimato: {offset} -> Angolo limitato: {angle}")
        px.set_dir_servo_angle(angle)
        px.forward(10)  # Riduci la velocitÃ  quando una sola linea Ã¨ visibile

    else:
        print("Linee non rilevate. Stop.")
        px.stop()

    # Visualizzazione
    #cv2.imshow("Vista originale", frame)
    #cv2.imshow("Bird Eye View", bird_eye)
    #cv2.imshow("Maschera Bianca (HLS)", white_mask)
    #cv2.imshow("Maschera Gialla (HSV)", yellow_mask)
    #cv2.imshow("Maschera Combinata", combined_mask)
    cv2.imshow("Corsie rilevate", overlay)
    cv2.imshow("Vista con sovrapposizione", combined)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Arresto finale
px.stop()
picam2.close()
cv2.destroyAllWindows()