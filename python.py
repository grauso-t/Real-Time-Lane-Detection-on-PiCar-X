import time
import numpy as np
from picamera2 import Picamera2
from picarx import Picarx
import cv2

# === Control Parameters ===
# Metodo di sterzata da utilizzare. Le opzioni includono 'linear', 'exponential', 'sigmoid', 'quadratic'.
# 'exponential' offre una risposta più aggressiva per piccole deviazioni.
STEERING_METHOD = 'exponential'
# Fattore di smoothing per l'angolo di sterzata. Un valore più alto (es. 0.9) porta a sterzate più lente e fluide,
# ma anche a una risposta più lenta. Un valore più basso (es. 0.5) rende la sterzata più reattiva.
SMOOTHING_FACTOR = 0.7
# Velocità del veicolo. Impostato a 0 inizialmente e successivamente con px.forward(SPEED).
SPEED = 0
# Variabile per memorizzare l'ultimo angolo di sterzata valido, utilizzata per lo smoothing e
# per mantenere una direzione in assenza di linee rilevate.
last_valid_steering_angle = 0.0

# Larghezza minima della carreggiata (in pixel nella vista bird-eye) per considerare le linee valide.
# Se la larghezza calcolata è inferiore a questo valore, l'angolo di sterzata precedente verrà mantenuto.
MIN_LANE_WIDTH_PIXELS = 180

# === Setup PiCar-X ===
# Inizializza l'oggetto Picarx per controllare il veicolo.
px = Picarx()
# Imposta la velocità iniziale del motore a 0 (fermo).
px.forward(0)

# === Setup Camera ===
# Inizializza l'oggetto Picamera2 per acquisire immagini.
picam2 = Picamera2()
# Configura la camera per acquisire immagini con una risoluzione di 640x480 pixel.
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
# Avvia la cattura delle immagini dalla camera.
picam2.start()
# Attende 2 secondi per permettere alla camera di avviarsi e scaldarsi.
time.sleep(2)

# Imposta la velocità di movimento del veicolo dopo l'avvio della camera.
px.forward(SPEED)

# === Steering Function ===
# Funzione per calcolare l'angolo di sterzata basandosi sulla pendenza rilevata della corsia.
# 'slope': La pendenza media calcolata delle linee della corsia (più ripida = più deviazione).
# 'method': Il metodo di mappatura della pendenza all'angolo di sterzata.
def calculate_steering_angle(slope, method='exponential'):
    # Limita la pendenza tra -2.0 e 2.0 per evitare valori estremi che potrebbero causare sterzate brusche.
    slope = np.clip(slope, -2.0, 2.0)
    if method == 'linear':
        # Metodo lineare: l'angolo di sterzata è direttamente proporzionale alla pendenza normalizzata.
        # Normalizza la pendenza dividendo per 0.5 e limitandola tra -1.0 e 1.0.
        normalized_slope = np.clip(slope / 0.5, -1.0, 1.0)
        # L'angolo massimo di sterzata è ±45 gradi.
        return normalized_slope * 45.0
    elif method == 'exponential':
        # Metodo esponenziale: la risposta di sterzata aumenta più rapidamente con l'aumentare della pendenza.
        # Offre una maggiore sensibilità per piccole deviazioni.
        normalized_slope = slope / 0.5
        # Determina il segno della pendenza per mantenere la direzione di sterzata.
        sign = np.sign(normalized_slope)
        abs_normalized = abs(normalized_slope)
        # Calcola la risposta esponenziale. Il denominatore normalizza il risultato tra 0 e 1.
        exponential_response = (np.exp(abs_normalized * 1.5) - 1) / (np.exp(1.5) - 1) if abs_normalized <= 1.0 else 1.0
        return sign * exponential_response * 45.0
    elif method == 'sigmoid':
        # Metodo sigmoide: una curva a "S" che offre una risposta graduale all'inizio,
        # poi più ripida e infine si appiattisce.
        k = 3.0 # Fattore di ripidità della curva sigmoide.
        sigmoid_response = 2 / (1 + np.exp(-k * slope)) - 1
        return sigmoid_response * 45.0
    elif method == 'quadratic':
        # Metodo quadratico: la risposta di sterzata aumenta più rapidamente man mano che la pendenza aumenta.
        normalized_slope = np.clip(slope / 0.5, -1.0, 1.0)
        sign = np.sign(normalized_slope)
        quadratic_response = normalized_slope ** 2 # La risposta è il quadrato della pendenza normalizzata.
        return sign * quadratic_response * 45.0
    else:
        # Se il metodo specificato non è valido, usa 'exponential' come predefinito.
        return calculate_steering_angle(slope, 'exponential')

try:
    # Ciclo principale di elaborazione del frame.
    while True:
        # Acquisisce un frame dall'array della camera.
        frame = picam2.capture_array()
        # Ottiene altezza (h) e larghezza (w) del frame.
        h, w = frame.shape[:2]
        # Definisce la regione di interesse (ROI) nella parte inferiore del frame,
        # dove si presume che le linee della corsia siano più visibili e pertinenti.
        roi_top, roi_bottom = int(h * 0.6), h

        # === Bird-eye transform (Trasformazione a occhio d'uccello) ===
        # Definisce i punti sorgente (ROI nel frame originale) per la trasformazione prospettica.
        # Questi punti rappresentano gli angoli del trapezio nel frame originale.
        src = np.float32([[0, roi_top], [w, roi_top], [w, roi_bottom], [0, roi_bottom]])
        # Definisce le dimensioni desiderate per l'immagine trasformata (larghezza, altezza).
        dst_w, dst_h = 300, 200
        # Definisce i punti di destinazione (un rettangolo nell'immagine trasformata) per la trasformazione.
        dst = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]])
        # Calcola la matrice di trasformazione prospettica dai punti sorgente e destinazione.
        matrix = cv2.getPerspectiveTransform(src, dst)
        # Applica la trasformazione prospettica al frame, ottenendo la vista a occhio d'uccello.
        bird_eye = cv2.warpPerspective(frame, matrix, (dst_w, dst_h))

        # === Elaborazione delle immagini per il rilevamento delle linee ===
        # Converte l'immagine da BGR a HSV (Hue, Saturation, Value) per una migliore segmentazione del colore.
        hsv = cv2.cvtColor(bird_eye, cv2.COLOR_BGR2HSV)
        # Crea una maschera per rilevare i colori bianchi.
        # np.array([0, 0, 200]) è il limite inferiore HSV per il bianco.
        # np.array([180, 30, 255]) è il limite superiore HSV per il bianco.
        mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        # Crea una maschera per rilevare i colori gialli.
        # np.array([20, 100, 100]) è il limite inferiore HSV per il giallo.
        # np.array([30, 255, 255]) è il limite superiore HSV per il giallo.
        mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        # Combina le due maschere (bianco e giallo) usando l'operazione OR bit a bit.
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        # Applica un filtro gaussiano per sfocare l'immagine e ridurre il rumore.
        # (5, 5) è la dimensione del kernel, 0 è la deviazione standard in X (calcolata automaticamente).
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        # Applica l'algoritmo Canny per rilevare i bordi.
        # 50 e 150 sono i valori di soglia bassa e alta per il rilevamento dei bordi.
        edges = cv2.Canny(blurred, 50, 150)

        # === Rilevamento linee con HoughLinesP ===
        # Applica la Trasformata di Hough Probabilistica per rilevare le linee nell'immagine dei bordi.
        # 'edges': Immagine binaria dei bordi.
        # 1: Risoluzione del raggio in pixel.
        # np.pi / 180: Risoluzione dell'angolo in radianti (1 grado).
        # threshold=50: Numero minimo di intersezioni per essere considerato una linea.
        # minLineLength=20: Lunghezza minima della linea per essere rilevata.
        # maxLineGap=30: Distanza massima tra segmenti di linea per considerarli un'unica linea.
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=30)
        # Liste per memorizzare le linee di corsia sinistra e destra.
        left_lines, right_lines = [], []

        # Crea un'immagine nera vuota per disegnare le linee rilevate e visualizzarle.
        line_image = np.zeros_like(bird_eye)

        if lines is not None:
            for line in lines:
                # Estrae le coordinate dei punti di inizio e fine della linea.
                x1, y1, x2, y2 = line[0]
                # Calcola la pendenza della linea. Aggiunge 1e-6 per evitare divisioni per zero.
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                # Ignora le linee con pendenza quasi orizzontale (pendenza assoluta minore di 0.5),
                # poiché non rappresentano le linee di corsia.
                if abs(slope) < 0.5:
                    continue
                # Classifica la linea come "sinistra" se la pendenza è negativa (inclinata verso l'alto a sinistra)
                # o "destra" se la pendenza è positiva (inclinata verso l'alto a destra).
                (left_lines if slope < 0 else right_lines).append((x1, y1, x2, y2))

        # === Funzione per ottenere la pendenza media e i punti della linea ===
        # 'lines': Lista di segmenti di linea (x1, y1, x2, y2).
        def get_avg_slope(lines):
            if not lines:
                # Se non ci sono linee, restituisce None per pendenza e coordinate.
                return None, None
            x_coords, y_coords = [], []
            # Raccoglie tutte le coordinate x e y da tutti i segmenti di linea.
            for x1, y1, x2, y2 in lines:
                x_coords += [x1, x2]
                y_coords += [y1, y2]
            if not x_coords:
                return None, None
            # Esegue una regressione lineare (fit polinomiale di grado 1) per trovare la linea media.
            # `poly[0]` sarà la pendenza e `poly[1]` l'intercetta y.
            poly = np.polyfit(y_coords, x_coords, deg=1)
            # Calcola i punti x per disegnare la linea media, basandosi sui valori min e max di y.
            min_y = min(y_coords)
            max_y = max(y_coords)
            fit_x1 = int(poly[0] * min_y + poly[1])
            fit_x2 = int(poly[0] * max_y + poly[1])
            # Restituisce la pendenza e le coordinate (x1, y1, x2, y2) della linea media.
            return poly[0], (fit_x1, min_y, fit_x2, max_y)

        # Calcola la pendenza media e le coordinate per le linee di corsia sinistra e destra.
        left_slope, left_line_coords = get_avg_slope(left_lines)
        right_slope, right_line_coords = get_avg_slope(right_lines)

        angles = []
        lane_width = 0 # Inizializza la larghezza della carreggiata
        
        # Inizializza le coordinate x medie delle linee per il calcolo della larghezza
        left_line_x_avg = None
        right_line_x_avg = None

        # Se sono state rilevate linee di corsia sinistra:
        if left_line_coords:
            # Disegna la linea di corsia sinistra sull'immagine `line_image` in blu.
            cv2.line(line_image, (left_line_coords[0], left_line_coords[1]), (left_line_coords[2], left_line_coords[3]), (255, 0, 0), 5)
            # Calcola l'angolo di sterzata basandosi sulla pendenza della corsia sinistra.
            angle = calculate_steering_angle(left_slope, STEERING_METHOD)
            angles.append(angle)
            # Visualizza la pendenza della corsia sinistra sull'immagine.
            cv2.putText(line_image, f"L Slope: {left_slope:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Calcola la coordinata x media della linea sinistra a metà altezza dell'immagine bird-eye
            left_line_x_avg = int((left_line_coords[0] + left_line_coords[2]) / 2)


        # Se sono state rilevate linee di corsia destra:
        if right_line_coords:
            # Disegna la linea di corsia destra sull'immagine `line_image` in rosso.
            cv2.line(line_image, (right_line_coords[0], right_line_coords[1]), (right_line_coords[2], right_line_coords[3]), (0, 0, 255), 5)
            # Calcola l'angolo di sterzata basandosi sulla pendenza della corsia destra.
            angle = calculate_steering_angle(right_slope, STEERING_METHOD)
            angles.append(angle)
            # Visualizza la pendenza della corsia destra sull'immagine.
            cv2.putText(line_image, f"R Slope: {right_slope:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Calcola la coordinata x media della linea destra a metà altezza dell'immagine bird-eye
            right_line_x_avg = int((right_line_coords[0] + right_line_coords[2]) / 2)

        # Calcola la larghezza della carreggiata se entrambe le linee sono state rilevate
        if left_line_x_avg is not None and right_line_x_avg is not None:
            lane_width = abs(right_line_x_avg - left_line_x_avg)
            cv2.putText(line_image, f"Width: {lane_width} px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) # Giallo per la larghezza

        # Calcola l'angolo di sterzata corrente.
        if angles and lane_width > MIN_LANE_WIDTH_PIXELS:
            # Se sono stati calcolati angoli e la carreggiata è sufficientemente larga, ne fa la media.
            current_steering_angle = sum(angles) / len(angles)
        else:
            # Se non sono state rilevate linee valide o la carreggiata è troppo stretta,
            # mantiene l'ultimo angolo di sterzata valido.
            current_steering_angle = last_valid_steering_angle
            if lane_width <= MIN_LANE_WIDTH_PIXELS and lane_width != 0:
                cv2.putText(line_image, "Lane too narrow!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) # Rosso per avviso

        # Applica lo smoothing all'angolo di sterzata finale.
        # Combina l'ultimo angolo valido con l'angolo corrente per una transizione più fluida.
        final_angle = (SMOOTHING_FACTOR * 1.2 * last_valid_steering_angle +
                       (1 - SMOOTHING_FACTOR * 1.2) * current_steering_angle)
        # Aggiorna l'ultimo angolo di sterzata valido.
        last_valid_steering_angle = final_angle

        # Invia l'angolo di sterzata calcolato al servo del PicarX.
        # L'angolo deve essere un intero.
        px.set_dir_servo_angle(int(final_angle))

        # === Visualizzazione Debug ===
        # Sovrappone l'immagine delle linee (`line_image`) sulla vista a occhio d'uccello (`bird_eye`).
        # 0.8 e 1 sono i pesi (alfa) per la fusione delle due immagini.
        combined_view = cv2.addWeighted(bird_eye, 0.8, line_image, 1, 0)
        # Visualizza l'angolo di sterzata finale sulla vista combinata.
        cv2.putText(combined_view, f"Steering: {final_angle:.1f} deg", (10, dst_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Mostra la vista a occhio d'uccello con le linee e l'angolo di sterzata.
        cv2.imshow("Bird-Eye View with Lanes", combined_view)
        # Mostra l'immagine dei bordi per ulteriore debugging.
        cv2.imshow("Edges", edges)

        # Controlla se il tasto 'q' è stato premuto per uscire dal ciclo.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    # Gestisce l'interruzione da tastiera (Ctrl+C) per fermare il programma in modo pulito.
    print("Interrupted, stopping...")
finally:
    # === Pulizia finale ===
    # Imposta l'angolo del servo a 0 gradi (posizione centrale).
    px.set_dir_servo_angle(0)
    # Ferma i motori del PicarX.
    px.stop()
    # Ferma la camera.
    picam2.stop()
    # Chiude tutte le finestre OpenCV aperte.
    cv2.destroyAllWindows()