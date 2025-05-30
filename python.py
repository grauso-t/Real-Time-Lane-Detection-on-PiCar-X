import cv2
import time
import numpy as np
from picamera2 import Picamera2
from picarx import Picarx

px = Picarx()  # Inizializza il controllo del veicolo
picam2 = Picamera2()  # Inizializza l'oggetto Picamera2 per acquisire immagini.
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)

# === VARIABILI DI STATO ===
# Memorizzano gli ultimi valori validi per garantire continuità nel controllo
last_valid_steering_angle = 0.0      # Ultimo angolo di sterzata valido complessivo
last_valid_left_angle = None         # Ultimo angolo della linea sinistra valido
last_valid_right_angle = None        # Ultimo angolo della linea destra valido

# === PARAMETRI DI CONFIGURAZIONE ===
STEERING_METHOD = 'hybrid_deSantis'      # Metodo di calcolo angolo: 'linear', 'exponential', 'sigmoid', 'quadratic'
SMOOTHING_FACTOR = 0              # Fattore di smoothing (0-1): più alto = transizioni più morbide
MIN_LANE_WIDTH = 180                # Larghezza minima carreggiata in pixel per considerarla valida
SPEED = 1                       # Velocità del veicolo (puoi regolare in base al tuo setup)

px.forward(SPEED)  # Avvia il veicolo in avanti

def create_info_panel(width=400, height=600):
    """
    Crea un pannello informativo con sfondo nero
    
    Args:
        width: Larghezza del pannello
        height: Altezza del pannello
    
    Returns:
        Immagine numpy con sfondo nero
    """
    return np.zeros((height, width, 3), dtype=np.uint8)

def add_text_to_panel(panel, text, position, font_scale=0.6, color=(255, 255, 255), thickness=1):
    """
    Aggiunge testo al pannello informativo
    
    Args:
        panel: Immagine del pannello
        text: Testo da aggiungere
        position: Tupla (x, y) della posizione
        font_scale: Dimensione del font
        color: Colore del testo (B, G, R)
        thickness: Spessore del testo
    """
    cv2.putText(panel, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def add_section_title(panel, title, y_pos, color=(0, 255, 255)):
    """
    Aggiunge un titolo di sezione al pannello
    
    Args:
        panel: Immagine del pannello
        title: Titolo della sezione
        y_pos: Posizione verticale
        color: Colore del titolo
    
    Returns:
        Nuova posizione y per il contenuto della sezione
    """
    # Linea separatrice sopra il titolo
    cv2.line(panel, (10, y_pos - 5), (390, y_pos - 5), color, 1)
    
    # Titolo
    add_text_to_panel(panel, title, (10, y_pos + 15), font_scale=0.7, color=color, thickness=2)
    
    return y_pos + 35

def calculate_steering_angle(slope, method='exponential'):
    """
    Calcola l'angolo di sterzata basato sulla pendenza con diverse modalità di risposta
    
    Args:
        slope: Pendenza della linea calcolata come (y2-y1)/(x2-x1)
        method: Metodo di calcolo - 'linear', 'exponential', 'sigmoid', 'quadratic'
    
    Returns:
        Angolo di sterzata in gradi nel range [-45, 45]
    """
    # Limita la pendenza per evitare valori estremi che potrebbero causare comportamenti erratici
    slope = np.clip(slope, -2.0, 2.0)
    
    if method == 'linear':
        # Mappatura lineare classica: pendenza -> angolo
        # Normalizza la pendenza dividendo per 0.5 (pendenza di riferimento)
        normalized_slope = np.clip(slope / 0.5, -1.0, 1.0)
        return normalized_slope * 45.0  # Scala all'angolo massimo di 45°
        
    elif method == 'hybrid_deSantis':
        # Ritorno angolo di sterzata precedente se la pendenza è zero
        if slope == 0 or slope is None:
            return last_valid_steering_angle

        if abs(slope) >= 0.6:
            # Applica direttamente l'angolo massimo
            return np.sign(slope) * 45.0
        else:
            # Risposta esponenziale per pendenze più contenute
            normalized_slope = slope / 0.5
            sign = np.sign(normalized_slope)
            abs_normalized = abs(normalized_slope)
            
            if abs_normalized <= 1.0:
                exponential_response = (np.exp(abs_normalized * 1.5) - 1) / (np.exp(1.5) - 1)
            else:
                exponential_response = 1.0
            
            return sign * exponential_response * 45.0
        
    elif method == 'sigmoid':
        # Risposta sigmoidale - transizione smooth ma con crescita rapida al centro
        # Buona per avere controllo fine ma reattività quando necessario
        k = 3.0  # Fattore di steepness (ripidità della curva)
        sigmoid_response = 2 / (1 + np.exp(-k * slope)) - 1
        return sigmoid_response * 45.0
        
    elif method == 'quadratic':
        # Risposta quadratica - crescita progressiva, più dolce degli altri metodi
        normalized_slope = np.clip(slope / 0.5, -1.0, 1.0)
        sign = np.sign(normalized_slope)
        quadratic_response = normalized_slope ** 2  # Eleva al quadrato per curva parabolica
        return sign * quadratic_response * 45.0 

    else:
        # Fallback al metodo esponenziale se il metodo specificato non è riconosciuto
        return calculate_steering_angle(slope, 'exponential')

def validate_lane_geometry(left_poly, right_poly, dst_w, dst_h, min_lane_width=180):
    """
    Valida la geometria delle linee rilevate per assicurarsi che abbiano senso
    dal punto di vista della guida reale
    
    Args:
        left_poly: Coefficienti del polinomio della linea sinistra (None se non rilevata)
        right_poly: Coefficienti del polinomio della linea destra (None se non rilevata)
        dst_w: Larghezza dell'immagine bird-eye view
        dst_h: Altezza dell'immagine bird-eye view
        min_lane_width: Larghezza minima accettabile della carreggiata in pixel
        
    Returns:
        tuple: (is_valid, lane_width, error_message)
            - is_valid: True se la geometria è valida
            - lane_width: Larghezza della carreggiata in pixel (0 se non calcolabile)
            - error_message: Descrizione del risultato della validazione
    """
    center_x = dst_w // 2  # Centro orizzontale dell'immagine
    y_test = dst_h // 2    # Punto verticale per testare le posizioni delle linee
    
    if left_poly is not None and right_poly is not None:
        # Caso: entrambe le linee sono state rilevate
        
        # Calcola le posizioni x delle linee al punto di test
        x_left = np.polyval(left_poly, y_test)   # Posizione x della linea sinistra
        x_right = np.polyval(right_poly, y_test) # Posizione x della linea destra
        
        # Controllo 1: La linea sinistra deve essere effettivamente a sinistra del centro
        if x_left >= center_x:
            return False, 0, "Linea sinistra a destra del centro"
        
        # Controllo 2: La linea destra deve essere effettivamente a destra del centro
        if x_right <= center_x:
            return False, 0, "Linea destra a sinistra del centro"
        
        # Controllo 3: Calcola larghezza carreggiata e verifica che sia ragionevole
        lane_width = x_right - x_left
        if lane_width < min_lane_width:
            return False, lane_width, f"Carreggiata stretta: {lane_width:.1f}px"
        
        # Tutti i controlli superati
        return True, lane_width, "Geometria valida"
    
    elif left_poly is not None:
        # Caso: solo linea sinistra rilevata
        x_left = np.polyval(left_poly, y_test)
        if x_left >= center_x:
            return False, 0, "Linea sinistra a destra del centro"
        return True, 0, "Solo linea sinistra valida"
    
    elif right_poly is not None:
        # Caso: solo linea destra rilevata
        x_right = np.polyval(right_poly, y_test)
        if x_right <= center_x:
            return False, 0, "Linea destra a sinistra del centro"
        return True, 0, "Solo linea destra valida"
    
    # Caso: nessuna linea rilevata
    return False, 0, "Nessuna linea rilevata"

# === CICLO PRINCIPALE DI ELABORAZIONE ===
while True:
    # Leggi il prossimo frame dal video
    frame = picam2.capture_array()
    
    # === PREPARAZIONE DELL'IMMAGINE ===
    h, w = frame.shape[:2]  # Dimensioni del frame originale
    
    # Definisce la ROI (Region of Interest) - la parte bassa del frame dove sono le linee
    roi_top, roi_bottom = int(h * 0.6), h  # Dal 60% in giù dell'immagine
    
    # === TRASFORMAZIONE PROSPETTICA (BIRD-EYE VIEW) ===
    # Definisce i punti sorgente (trapezio della strada nel frame originale)
    src = np.float32([[0, roi_top], [w, roi_top], [w, roi_bottom], [0, roi_bottom]])
    
    # Definisce le dimensioni e i punti di destinazione (rettangolo della bird-eye view)
    dst_w, dst_h = 300, 200
    dst = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]])
    
    # Calcola la matrice di trasformazione e applica la trasformazione
    matrix = cv2.getPerspectiveTransform(src, dst)
    bird_eye = cv2.warpPerspective(frame, matrix, (dst_w, dst_h))
    
    # === CONVERSIONE SPAZIO COLORE E CREAZIONE MASCHERE ===
    hsv = cv2.cvtColor(bird_eye, cv2.COLOR_BGR2HSV)
    
    # Maschera per linee bianche (alta luminosità, bassa saturazione)
    lower_white = np.array([0, 0, 200])      # H, S, V minimi
    upper_white = np.array([180, 30, 255])   # H, S, V massimi
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Maschera per linee gialle (tonalità gialla)
    lower_yellow = np.array([20, 100, 100])  # H, S, V minimi
    upper_yellow = np.array([30, 255, 255])  # H, S, V massimi
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combina le due maschere per rilevare sia linee bianche che gialle
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    # === PREPROCESSING PER RILEVAMENTO LINEE ===
    # Applica filtro gaussiano per ridurre il rumore
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Rilevamento bordi con algoritmo Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # === RILEVAMENTO LINEE CON HOUGH TRANSFORM ===
    lines = cv2.HoughLinesP(
        edges,              # Immagine dei bordi
        1,                  # Risoluzione in pixel
        np.pi / 180,        # Risoluzione angolare in radianti
        threshold=50,       # Soglia minima per considerare una linea
        minLineLength=20,   # Lunghezza minima della linea
        maxLineGap=30       # Gap massimo tra segmenti per considerarli una linea continua
    )
    
    # Inizializza immagine per il disegno e liste per le linee
    line_img = bird_eye.copy()
    left_lines = []   # Lista per linee con pendenza negativa (linee sinistre)
    right_lines = []  # Lista per linee con pendenza positiva (linee destre)
    
    # === CLASSIFICAZIONE DELLE LINEE ===
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Coordinate degli estremi della linea
            
            # Calcola la pendenza, evitando divisione per zero
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            
            # Filtra linee troppo orizzontali (pendenza bassa)
            if abs(slope) < 0.5:
                continue
            
            # Classifica in base alla pendenza
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))   # Pendenza negativa = linea sinistra
            else:
                right_lines.append((x1, y1, x2, y2))  # Pendenza positiva = linea destra

    def draw_average_line(lines, color, label):
        """
        Calcola e restituisce il polinomio medio di un insieme di linee
        
        Args:
            lines: Lista di tuple (x1, y1, x2, y2) rappresentanti le linee
            color: Colore per il disegno (non utilizzato in questa versione)
            label: Etichetta per debug ("Left" o "Right")
            
        Returns:
            tuple: (poly, slope, steering_angle)
                - poly: Coefficienti del polinomio della linea media
                - slope: Pendenza della linea media
                - steering_angle: Angolo di sterzata calcolato
        """
        if lines:
            # Raccoglie tutte le coordinate dei punti delle linee
            x_coords = []
            y_coords = []
            for x1, y1, x2, y2 in lines:
                x_coords += [x1, x2]  # Aggiunge entrambi gli estremi
                y_coords += [y1, y2]
            
            if len(x_coords) > 0:
                # Calcola il polinomio di primo grado che meglio approssima i punti
                # Nota: polyfit(y, x) perché vogliamo x in funzione di y (x = m*y + b)
                poly = np.polyfit(y_coords, x_coords, deg=1)
                slope = poly[0]  # Il coefficiente angolare m
                
                # Calcola l'angolo di sterzata basato sulla pendenza
                raw_steering_angle = calculate_steering_angle(slope, STEERING_METHOD)
                
                # === SMOOTHING TEMPORALE ===
                # Applica smoothing se abbiamo un valore precedente valido
                if label == 'Left' and last_valid_left_angle is not None:
                    steering_angle = (SMOOTHING_FACTOR * last_valid_left_angle + 
                                    (1 - SMOOTHING_FACTOR) * raw_steering_angle)
                elif label == 'Right' and last_valid_right_angle is not None:
                    steering_angle = (SMOOTHING_FACTOR * last_valid_right_angle + 
                                    (1 - SMOOTHING_FACTOR) * raw_steering_angle)
                else:
                    steering_angle = raw_steering_angle
                
                return poly, slope, steering_angle
        
        # Nessuna linea trovata
        return None, None, None

    # === CALCOLO DEI POLINOMI DELLE LINEE ===
    left_poly, left_slope, left_angle = draw_average_line(left_lines, (255, 0, 0), "Left")
    right_poly, right_slope, right_angle = draw_average_line(right_lines, (0, 0, 255), "Right")

    # === VALIDAZIONE GEOMETRICA ===
    # Controlla se la geometria delle linee rilevate ha senso
    geometry_valid, lane_width, validation_message = validate_lane_geometry(
        left_poly, right_poly, dst_w, dst_h, MIN_LANE_WIDTH
    )
    
    # === INIZIALIZZAZIONE VARIABILI PER IL DISPLAY ===
    current_steering_angle = None
    
    # === GESTIONE BASATA SULLA VALIDITÀ DELLA GEOMETRIA ===
    if geometry_valid:
        # *** GEOMETRIA VALIDA - PROCEDI CON AGGIORNAMENTO ***
        
        # Disegna la linea sinistra se rilevata
        if left_poly is not None:
            y_start, y_end = 0, dst_h  # Estremi verticali per il disegno
            x_start = int(np.polyval(left_poly, y_start))  # x corrispondente a y_start
            x_end = int(np.polyval(left_poly, y_end))      # x corrispondente a y_end
            cv2.line(line_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)  # Blu
        
        # Disegna la linea destra se rilevata
        if right_poly is not None:
            y_start, y_end = 0, dst_h
            x_start = int(np.polyval(right_poly, y_start))
            x_end = int(np.polyval(right_poly, y_end))
            cv2.line(line_img, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3)  # Rosso
        
        # === CALCOLO NUOVO ANGOLO DI STERZATA ===
        if left_angle is not None and right_angle is not None:
            # Entrambe le linee: media degli angoli
            current_steering_angle = (left_angle + right_angle) / 2
            last_valid_left_angle = left_angle    # Aggiorna ultimo valore valido
            last_valid_right_angle = right_angle  # Aggiorna ultimo valore valido
        elif left_angle is not None:
            # Solo linea sinistra
            current_steering_angle = left_angle
            last_valid_left_angle = left_angle
        elif right_angle is not None:
            # Solo linea destra
            current_steering_angle = right_angle
            last_valid_right_angle = right_angle
    
    else:
        # *** GEOMETRIA NON VALIDA - MANTIENI VALORI PRECEDENTI ***
        
        # Non aggiornare l'angolo, mantieni l'ultimo valore valido
        current_steering_angle = last_valid_steering_angle
        
        # Disegna comunque le linee rilevate ma in grigio per indicare che sono invalide
        if left_poly is not None:
            y_start, y_end = 0, dst_h
            x_start = int(np.polyval(left_poly, y_start))
            x_end = int(np.polyval(left_poly, y_end))
            cv2.line(line_img, (x_start, y_start), (x_end, y_end), (100, 100, 100), 2)  # Grigio
        
        if right_poly is not None:
            y_start, y_end = 0, dst_h
            x_start = int(np.polyval(right_poly, y_start))
            x_end = int(np.polyval(right_poly, y_end))
            cv2.line(line_img, (x_start, y_start), (x_end, y_end), (100, 100, 100), 2)  # Grigio

    # === GESTIONE CASO INIZIALE ===
    # Se non abbiamo un angolo corrente (primo frame o nessuna linea mai rilevata)
    if current_steering_angle is None:
        current_steering_angle = last_valid_steering_angle

    # === SMOOTHING GLOBALE FINALE ===
    # Applica smoothing aggiuntivo all'angolo di sterzata finale per maggiore stabilità
    if current_steering_angle is not None and current_steering_angle != last_valid_steering_angle:
        if last_valid_steering_angle != 0.0:
            # Smoothing più aggressivo per l'angolo finale
            final_steering_angle = (SMOOTHING_FACTOR * 1.2 * last_valid_steering_angle + 
                                  (1 - SMOOTHING_FACTOR * 1.2) * current_steering_angle)
        else:
            final_steering_angle = current_steering_angle
        
        # Aggiorna il valore memorizzato
        last_valid_steering_angle = final_steering_angle
    else:
        final_steering_angle = last_valid_steering_angle

    px.set_dir_servo_angle(final_steering_angle)  # Imposta l'angolo di sterzata al veicolo

    # === STIMA DEL CENTRO CORSIA ===
    # Calcola la posizione del centro della corsia per la visualizzazione
    lane_width_px = 300  # Larghezza standard stimata in pixel
    y_pos = dst_h // 3   # Altezza a cui calcolare il centro
    
    if geometry_valid:
        # Usa la geometria attuale se valida
        if left_poly is not None and right_poly is not None:
            # Entrambe le linee: centro tra le due
            x_left = int(np.polyval(left_poly, y_pos))
            x_right = int(np.polyval(right_poly, y_pos))
            lane_center = (x_left + x_right) // 2
        elif left_poly is not None:
            # Solo linea sinistra: stima il centro aggiungendo metà larghezza standard
            x_left = int(np.polyval(left_poly, y_pos))
            lane_center = x_left + lane_width_px // 2
        elif right_poly is not None:
            # Solo linea destra: stima il centro sottraendo metà larghezza standard
            x_right = int(np.polyval(right_poly, y_pos))
            lane_center = x_right - lane_width_px // 2
        else:
            # Nessuna linea valida: usa il centro dell'immagine
            lane_center = dst_w // 2
    else:
        # Geometria non valida: usa il centro dell'immagine come fallback
        lane_center = dst_w // 2

    # === VISUALIZZAZIONE CENTRO CORSIA ===
    # Colore diverso in base alla validità della geometria
    center_color = (0, 255, 255) if geometry_valid else (100, 100, 100)  # Giallo se valido, grigio se invalido
    
    # Disegna il punto centrale e la linea verticale
    cv2.circle(line_img, (lane_center, y_pos), 5, center_color, -1)            # Cerchio
    cv2.line(line_img, (lane_center, 0), (lane_center, dst_h), center_color, 2) # Linea verticale

    # === CREAZIONE PANNELLO INFORMATIVO ===
    info_panel = create_info_panel(400, 800)
    
    # Posizione verticale corrente per il testo
    current_y = 30
      
    # === SEZIONE LINEA SINISTRA ===
    current_y = add_section_title(info_panel, "LINEA SINISTRA", current_y, (255, 100, 100))
    if left_poly is not None:
        add_text_to_panel(info_panel, f"Rilevata: SI", (20, current_y), color=(0, 255, 0))
        current_y += 25
        add_text_to_panel(info_panel, f"Pendenza: {left_slope:.3f}", (20, current_y))
        current_y += 25
        add_text_to_panel(info_panel, f"Angolo: {left_angle:.1f}", (20, current_y))
        current_y += 25
        add_text_to_panel(info_panel, f"Ultimo Valid: {last_valid_left_angle:.1f}" if last_valid_left_angle else "Ultimo Valid: N/A", (20, current_y))
    else:
        add_text_to_panel(info_panel, f"Rilevata: NO", (20, current_y), color=(0, 0, 255))
        current_y += 25
        add_text_to_panel(info_panel, f"Ultimo Valid: {last_valid_left_angle:.1f}" if last_valid_left_angle else "Ultimo Valid: N/A", (20, current_y))
    current_y += 40
    
    # === SEZIONE LINEA DESTRA ===
    current_y = add_section_title(info_panel, "LINEA DESTRA", current_y, (100, 100, 255))
    if right_poly is not None:
        add_text_to_panel(info_panel, f"Rilevata: SI", (20, current_y), color=(0, 255, 0))
        current_y += 25
        add_text_to_panel(info_panel, f"Pendenza: {right_slope:.3f}", (20, current_y))
        current_y += 25
        add_text_to_panel(info_panel, f"Angolo: {right_angle:.1f}", (20, current_y))
        current_y += 25
        add_text_to_panel(info_panel, f"Ultimo Valid: {last_valid_right_angle:.1f}" if last_valid_right_angle else "Ultimo Valid: N/A", (20, current_y))
    else:
        add_text_to_panel(info_panel, f"Rilevata: NO", (20, current_y), color=(0, 0, 255))
        current_y += 25
        add_text_to_panel(info_panel, f"Ultimo Valid: {last_valid_right_angle:.1f}" if last_valid_right_angle else "Ultimo Valid: N/A", (20, current_y))
    current_y += 40
    
    # === SEZIONE VALIDAZIONE GEOMETRICA ===
    current_y = add_section_title(info_panel, "VALIDAZIONE GEOMETRICA", current_y, (255, 255, 100))
    validation_color = (0, 255, 0) if geometry_valid else (0, 0, 255)
    add_text_to_panel(info_panel, f"Stato: {'VALIDA' if geometry_valid else 'INVALIDA'}", (20, current_y), color=validation_color)
    current_y += 25
    add_text_to_panel(info_panel, f"Messaggio: {validation_message}", (20, current_y), font_scale=0.5)
    current_y += 25
    if lane_width > 0:
        add_text_to_panel(info_panel, f"Larghezza Corsia: {lane_width:.1f}px", (20, current_y))
    else:
        add_text_to_panel(info_panel, f"Larghezza Corsia: N/A", (20, current_y))
    current_y += 40
    
    # === SEZIONE CONTROLLO STERZATA ===
    current_y = add_section_title(info_panel, "CONTROLLO STERZATA", current_y, (100, 255, 255))
    add_text_to_panel(info_panel, f"Angolo Finale: {final_steering_angle:.1f}", (20, current_y), 
                     font_scale=0.8, thickness=2, color=(255, 255, 255))
    current_y += 30
    add_text_to_panel(info_panel, f"Angolo Corrente: {current_steering_angle:.1f}" if current_steering_angle else "Angolo Corrente: N/A", (20, current_y))
    current_y += 25
    add_text_to_panel(info_panel, f"Ultimo Valido: {last_valid_steering_angle:.1f}", (20, current_y))
    current_y += 40
    
    # === SEZIONE CENTRO CORSIA ===
    current_y = add_section_title(info_panel, "CENTRO CORSIA", current_y, (255, 255, 0))
    add_text_to_panel(info_panel, f"Posizione X: {lane_center}px", (20, current_y))
    current_y += 25
    add_text_to_panel(info_panel, f"Centro Immagine: {dst_w // 2}px", (20, current_y))
    current_y += 25
    offset = lane_center - (dst_w // 2)
    offset_direction = "DESTRA" if offset > 0 else "SINISTRA"
    add_text_to_panel(info_panel, f"Offset: {abs(offset)}px {offset_direction}", (20, current_y))
    current_y += 40
    
    # Stato del sistema
    if geometry_valid:
        system_status = "OPERATIVO"
        status_color = (0, 255, 0)
    else:
        system_status = "FALLBACK"
        status_color = (255, 165, 0)
    
    add_text_to_panel(info_panel, f"Stato Sistema: {system_status}", (20, current_y + 10), 
                     font_scale=0.7, color=status_color, thickness=2)

    # === VISUALIZZAZIONE FINALE ===
    cv2.imshow("Original", frame)        # Frame originale
    cv2.imshow("Bird Eye", line_img)     # Vista bird-eye con linee
    cv2.imshow("Info Panel", info_panel) # Pannello informativo
    
    # === CONTROLLO USCITA ===
    # Esci se viene premuto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Pausa per controllare la velocità di riproduzione
    time.sleep(0.03)  # ~33 FPS

# === CLEANUP ===
# Rilascia le risorse
picam2.stop()
px.set_dir_servo_angle(0)
px.stop()  # Ferma il veicolo
cv2.destroyAllWindows()