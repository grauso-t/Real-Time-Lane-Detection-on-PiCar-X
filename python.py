import cv2
import numpy as np
from picamera2 import Picamera2
from picarx import Picarx # Assicurati che il nome del modulo sia corretto
import time

# --- Parametri Configurabili ---
ROI_TOP_FACTOR = 0.8  # Percentuale dell'altezza per la parte superiore della ROI
DST_HEIGHT = 200      # Altezza dell'immagine bird's-eye di destinazione
DST_WIDTH = 300       # Larghezza dell'immagine bird's-eye di destinazione

# Range HSV per il colore giallo (potrebbe necessitare di aggiustamenti)
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([30, 255, 255])

# Range HSV per il colore bianco (potrebbe necessitare di aggiustamenti)
LOWER_WHITE = np.array([0, 0, 200])  # Aumentato il valore minimo di V per bianco più luminoso
UPPER_WHITE = np.array([180, 50, 255]) # Ridotto S_max per escludere colori troppo saturi

GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 30 # Abbassato per rilevare anche linee più deboli/tratteggiate
HOUGH_MIN_LINE_LENGTH = 20 # Abbassato
HOUGH_MAX_LINE_GAP = 10    # Aumentato per linee tratteggiate

FIXED_SPEED = 1      # Velocità fissa del Picar-X
MAX_STEERING_ANGLE = 45 # Angolo di sterzata massimo

# --- Inizializzazione ---
try:
    px = Picarx()
except Exception as e:
    print(f"Errore inizializzazione Picar-X: {e}")
    print("Il programma continuerà senza controllo del Picar-X.")
    px = None

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2) # Attendi che la camera si stabilizzi
print("Camera inizializzata.")

def get_camera_dimensions():
    """Ottiene le dimensioni del frame dalla configurazione della camera."""
    # Tentativo di ottenere le dimensioni dalla configurazione attiva
    try:
        sensor_modes = picam2.sensor_modes
        active_mode_index = picam2.camera_controls['ScalerCrop'][3] # Indice della modalità attiva
        active_mode = sensor_modes[active_mode_index]
        width = active_mode['size'][0]
        height = active_mode['size'][1]
        # Applica lo scaler crop se presente
        crop = picam2.camera_controls['ScalerCrop'] # (x_offset, y_offset, width, height)
        # Se ScalerCrop è (0,0,0,0) o simile, usa le dimensioni del sensore
        if crop[2] > 0 and crop[3] > 0:
             # Questo crop è sull'output del sensore prima del resize finale.
             # La configurazione main={"size": (W, H)} dovrebbe essere più affidabile
             pass # Lasciamo che le dimensioni del config prevalgano
        stream_config = picam2.camera_configuration()['main']
        width = stream_config['size'][0]
        height = stream_config['size'][1]

    except Exception as e:
        print(f"Errore nel recuperare dimensioni precise dalla camera: {e}. Uso quelle del config.")
        stream_config = picam2.camera_configuration()['main']
        width = stream_config['size'][0]
        height = stream_config['size'][1]

    return int(height), int(width)


def perspective_warp(img, src_pts, dst_pts):
    """Applica la trasformazione prospettica (bird's-eye view)."""
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(img, matrix, (DST_WIDTH, DST_HEIGHT))
    return warped_img, matrix

def color_filter(img):
    """Applica il filtro colore HSV per bianco e giallo."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Maschera gialla
    mask_yellow = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    # Maschera bianca
    mask_white = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

    # Combina le maschere
    mask_combined = cv2.bitwise_or(mask_yellow, mask_white)
    filtered_img = cv2.bitwise_and(img, img, mask=mask_combined)
    return filtered_img, mask_combined

def get_line_points(lines, img_height, img_width, is_left):
    """
    Estrae i punti delle linee da HoughLinesP.
    Se is_left è True, cerca linee con pendenza negativa.
    Se is_left è False, cerca linee con pendenza positiva.
    Restituisce una lista di tuple (x1, y1, x2, y2).
    """
    line_segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # Linea verticale, pendenza infinita
                slope = float('inf')
            else:
                slope = (y2 - y1) / (x2 - x1)

            # Filtra per pendenza approssimativa e posizione
            # Le coordinate y sono invertite (0 in alto)
            # Pendenza negativa per la linea sinistra, positiva per la destra
            # nel bird's eye view, le linee tendono a convergere verso il centro in alto
            if is_left:
                if slope < -0.1 : # Pendenza negativa
                    line_segments.append(line[0])
            else: # is_right
                if slope > 0.1: # Pendenza positiva
                    line_segments.append(line[0])
    return line_segments

def average_slope_intercept(lines, image_height):
    """
    Calcola la media delle linee rilevate per ottenere una singola linea rappresentativa.
    Restituisce (slope, intercept) della linea media.
    Se nessuna linea valida, restituisce None.
    """
    valid_lines = []
    if not lines:
        return None

    for line in lines:
        x1, y1, x2, y2 = line
        if x2 - x1 == 0: # Evita divisione per zero per linee verticali
            continue
        slope = (y2 - y1) / (x2 - x1)
        # Filtra pendenze estreme o piatte che potrebbero non essere linee di corsia
        if abs(slope) < 0.2 or abs(slope) > 2.0: # Ajusta questi valori se necessario
            continue
        intercept = y1 - slope * x1
        valid_lines.append((slope, intercept))

    if not valid_lines:
        return None

    # Calcola la media delle pendenze e degli intercetti
    avg_slope = np.mean([l[0] for l in valid_lines])
    avg_intercept = np.mean([l[1] for l in valid_lines])

    return avg_slope, avg_intercept


def make_coordinates(image_height, slope_intercept):
    """
    Converte (slope, intercept) in coordinate (x1, y1, x2, y2) per disegnare la linea.
    y1 è la parte inferiore dell'immagine, y2 è una frazione dell'altezza.
    """
    if slope_intercept is None:
        return None
    slope, intercept = slope_intercept
    y1 = image_height
    y2 = int(y1 * 0.1) # La linea si estende fino al 10% superiore dell'immagine bird's eye

    # Assicurati che la pendenza non sia zero per evitare divisione per zero
    if slope == 0: # Se la pendenza è zero, la linea è orizzontale
        # Potrebbe non essere utile per il rilevamento di corsia, ma gestiamolo
        x1 = 0
        x2 = DST_WIDTH # Larghezza dell'immagine bird's eye
    else:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def draw_lines(img, lines, color=(0, 0, 255), thickness=5):
    """Disegna le linee sull'immagine."""
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img


def calculate_steering_angle(left_line, right_line, warped_width, warped_height):
    """
    Calcola l'angolo di sterzata.
    Se una linea è persa, la assume nell'angolo inferiore corrispondente.
    """
    center_x_car = warped_width // 2

    if left_line is not None and right_line is not None:
        # Entrambe le linee rilevate
        # Punto medio inferiore delle linee
        _, ly1, _, ly2 = left_line
        _, ry1, _, ry2 = right_line

        # Prendiamo i punti inferiori delle linee (y più grande)
        # Se la linea è definita da (x1,y1,x2,y2), y1 è il bottom della warped
        lx_bottom = left_line[0]
        rx_bottom = right_line[0]

        center_x_lane = (lx_bottom + rx_bottom) // 2
        error = center_x_car - center_x_lane

    elif left_line is not None: # Solo linea sinistra rilevata
        lx_bottom = left_line[0] # x1 della linea sinistra (in basso)
        # Assumiamo che la linea destra sia all'estrema destra in basso
        # Creiamo un "triangolo" virtuale
        # L'errore è la deviazione dal centro della linea sinistra + una frazione della larghezza
        error = center_x_car - (lx_bottom + warped_width // 4) # Tende a sterzare a destra
        print("Linea destra persa, assumendo posizione.")

    elif right_line is not None: # Solo linea destra rilevata
        rx_bottom = right_line[0] # x1 della linea destra (in basso)
        # Assumiamo che la linea sinistra sia all'estrema sinistra in basso
        error = center_x_car - (rx_bottom - warped_width // 4) # Tende a sterzare a sinistra
        print("Linea sinistra persa, assumendo posizione.")

    else: # Nessuna linea rilevata
        print("Nessuna linea rilevata, mantenendo l'ultimo angolo o andando dritto.")
        return 0 # Angolo zero, vai dritto

    # Calcolo dell'angolo di sterzata (semplice proporzionale)
    # Più grande l'errore, più grande l'angolo
    # Il segno dell'errore determina la direzione (positivo -> sterza a sinistra, negativo -> sterza a destra)
    # Bisogna invertire il segno per il PicarX: positivo sterza a destra, negativo a sinistra
    steering_angle = -float(error) * 0.3 # Fattore Kp, da calibrare
    steering_angle = np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

    return steering_angle

def draw_lane_overlay(original_img, warped_img, left_line, right_line, inv_matrix):
    """Disegna la corsia sull'immagine originale."""
    overlay = np.zeros_like(warped_img, dtype=np.uint8)
    lane_color = (0, 255, 0) # Verde

    pts_left = []
    pts_right = []

    # Coordinate y per il poligono (dal basso verso l'alto della bird's eye)
    y_bottom_warped = warped_img.shape[0]
    y_top_warped = int(y_bottom_warped * 0.1) # Stesso y2 usato in make_coordinates

    if left_line is not None:
        x1_left, _, x2_left, _ = left_line # (x_bottom, y_bottom, x_top, y_top)
        pts_left = [[x1_left, y_bottom_warped], [x2_left, y_top_warped]]
    else:
        # Linea sinistra persa, forma un triangolo nell'angolo inferiore sinistro
        pts_left = [[0, y_bottom_warped], [0, y_top_warped]] # Bordo sinistro

    if right_line is not None:
        x1_right, _, x2_right, _ = right_line
        pts_right = [[x2_right, y_top_warped], [x1_right, y_bottom_warped]] # Invertito per il poligono
    else:
        # Linea destra persa, forma un triangolo nell'angolo inferiore destro
        pts_right = [[warped_img.shape[1], y_top_warped], [warped_img.shape[1], y_bottom_warped]] # Bordo destro

    if not pts_left or not pts_right:
        return original_img # Non fare nulla se non ci sono punti

    # Combina i punti per formare un poligono
    # Ordine: basso-sx, alto-sx, alto-dx, basso-dx
    lane_pts_warped = np.array(pts_left + pts_right, dtype=np.int32)

    if len(lane_pts_warped) >= 3 : # Deve avere almeno 3 punti per un poligono
        cv2.fillPoly(overlay, [lane_pts_warped], lane_color)

        # Trasforma l'overlay dalla vista bird's-eye all'immagine originale
        overlay_original_perspective = cv2.warpPerspective(overlay, inv_matrix, (original_img.shape[1], original_img.shape[0]))

        # Combina l'overlay con l'immagine originale
        result_img = cv2.addWeighted(original_img, 1, overlay_original_perspective, 0.5, 0)
        return result_img
    else:
        return original_img


# --- Ciclo Principale ---
try:
    height, width = get_camera_dimensions()
    print(f"Dimensioni frame: {width}x{height}")

    # Definizione ROI e punti di trasformazione
    roi_top = int(height * ROI_TOP_FACTOR)
    roi_bottom = height

    src_pts = np.float32([
        [0, roi_top],
        [width, roi_top],
        [width, roi_bottom],
        [0, roi_bottom]
    ])

    dst_pts = np.float32([
        [0, 0],
        [DST_WIDTH, 0],
        [DST_WIDTH, DST_HEIGHT],
        [0, DST_HEIGHT]
    ])

    # Matrice inversa per proiettare la corsia sull'immagine originale
    _, inv_perspective_matrix = perspective_warp(np.zeros((height, width, 3), dtype=np.uint8), dst_pts, src_pts) # Inverti src e dst per ottenere l'inversa


    while True:
        frame_rgb = picam2.capture_array() # Cattura come RGB

        # 1. Applica Gaussian Blur
        blurred_frame = cv2.GaussianBlur(frame_rgb, GAUSSIAN_BLUR_KERNEL_SIZE, 0)

        # 2. Filtro Colore (Bianco e Giallo)
        color_filtered_frame, combined_mask = color_filter(blurred_frame)

        # 3. Trasformazione Bird's-Eye View (sulla maschera o sull'immagine filtrata per colore)
        # Applichiamola sulla maschera binaria, è più efficiente per Hough
        warped_mask, perspective_matrix = perspective_warp(combined_mask, src_pts, dst_pts)
        # Per visualizzazione, possiamo trasformare anche l'immagine colorata
        warped_color_filtered, _ = perspective_warp(color_filtered_frame, src_pts, dst_pts)


        # 4. Rilevamento Linee con Hough Transform sulla maschera warped
        # L'input per HoughLinesP deve essere un'immagine binaria (bordi)
        # La nostra warped_mask è già binaria dopo il filtro colore.
        # Se non lo fosse (es. Canny), dovremmo applicare Canny qui.
        lines_hough = cv2.HoughLinesP(
            warped_mask,
            HOUGH_RHO,
            HOUGH_THETA,
            HOUGH_THRESHOLD,
            np.array([]),
            minLineLength=HOUGH_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_MAX_LINE_GAP
        )

        # Immagine per disegnare le linee di Hough per visualizzazione
        hough_visualization_img = np.zeros((DST_HEIGHT, DST_WIDTH, 3), dtype=np.uint8)
        if lines_hough is not None:
            for line in lines_hough:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_visualization_img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Linee Hough in blu


        # 5. Filtra e calcola la media delle linee sinistra e destra
        left_line_segments = get_line_points(lines_hough, DST_HEIGHT, DST_WIDTH, is_left=True)
        right_line_segments = get_line_points(lines_hough, DST_HEIGHT, DST_WIDTH, is_left=False)

        left_line_params = average_slope_intercept(left_line_segments, DST_HEIGHT)
        right_line_params = average_slope_intercept(right_line_segments, DST_HEIGHT)

        left_lane_line = make_coordinates(DST_HEIGHT, left_line_params)
        right_lane_line = make_coordinates(DST_HEIGHT, right_line_params)

        # Immagine per disegnare le linee di corsia mediate (sulla warped)
        averaged_lines_img = draw_lines(warped_color_filtered, [left_lane_line, right_lane_line])


        # 6. Calcola Angolo di Sterzata
        steering_angle = calculate_steering_angle(left_lane_line, right_lane_line, DST_WIDTH, DST_HEIGHT)

        # 7. Applica comandi al Picar-X
        if px:
            # La libreria picarx potrebbe avere forward() e set_dir_servo_angle()
            # o una singola funzione per controllare entrambi.
            # Assumiamo set_dir_servo_angle per l'angolo e una velocità fissa.
            # px.forward(FIXED_SPEED) # Questo potrebbe essere necessario se la velocità non è persistente
            try:
                px.set_dir_servo_angle(steering_angle)
                px.set_motor_speed(1, FIXED_SPEED) # Motore 1 (destro)
                px.set_motor_speed(2, FIXED_SPEED) # Motore 2 (sinistro)
                # Oppure se c'è un comando generico di velocità:
                # px.set_speed(FIXED_SPEED)
            except Exception as e:
                print(f"Errore nel controllo Picar-X: {e}")


        # 8. Disegna l'overlay della corsia sull'immagine originale
        # Usa la matrice di trasformazione inversa
        final_frame_with_overlay = draw_lane_overlay(frame_rgb.copy(), warped_mask, left_lane_line, right_lane_line, inv_perspective_matrix)


        # 9. Visualizzazione
        cv2.imshow("Original Frame", frame_rgb)
        cv2.imshow("Hough Lines on Warped Mask", hough_visualization_img)
        cv2.imshow("Color Filtered", color_filtered_frame)
        cv2.imshow("Warped Mask (Bird's Eye)", warped_mask)
        cv2.imshow("Averaged Lane Lines on Warped", cv2.addWeighted(warped_color_filtered, 0.8, averaged_lines_img, 1, 0))
        cv2.imshow("Final Overlay", final_frame_with_overlay)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Stop Picar-X e chiusura.")
    if px:
        px.set_dir_servo_angle(0) # Resetta lo sterzo
        px.stop() # Ferma i motori
    cv2.destroyAllWindows()
    picam2.stop()
    print("Programma terminato.")