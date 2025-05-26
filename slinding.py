import cv2
import time
from picamera2 import Picamera2
from picarx import Picarx
import readchar # Per leggere l'input da tastiera carattere per carattere

# --- Parametri di Configurazione ---
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 15.0  # Framerate per la registrazione (inizia basso)
OUTPUT_FILENAME_BASE = "picarx_recording" # Il nome del file sarà tipo "picarx_recording_20231027_103045.avi"

# Codec: Prova prima 'MJPG', poi 'XVID' se il primo non va o i file sono troppo grandi
# FOURCC = cv2.VideoWriter_fourcc(*'XVID')
FOURCC = cv2.VideoWriter_fourcc(*'MJPG')
VIDEO_EXTENSION = ".avi" # Usa .mp4 se usi codec come 'mp4v' o 'X264'

# Parametri di controllo PicarX
DRIVE_SPEED = 25  # Velocità di avanzamento/retromarcia
TURN_ANGLE = 25   # Angolo di sterzata (positivo per destra, negativo per sinistra)
# --- Fine Parametri di Configurazione ---

def get_timestamp_filename(base_name, extension):
    """Genera un nome file con timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"

def main():
    # Inizializza PicarX
    px = Picarx()
    px.stop() # Assicurati che sia ferma all'inizio

    # Inizializza Picamera2
    picam2 = Picamera2()
    camera_config = picam2.create_video_configuration(
        main={"size": (VIDEO_WIDTH, VIDEO_HEIGHT), "format": "RGB888"}, # Usa RGB888
        controls={"FrameRate": VIDEO_FPS} # Richiedi un framerate specifico alla camera
    )
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(1) # Dai tempo alla camera di avviarsi
    print(f"Camera avviata. Risoluzione: {VIDEO_WIDTH}x{VIDEO_HEIGHT}, FPS target: {VIDEO_FPS}")

    # Prepara VideoWriter
    output_filename = get_timestamp_filename(OUTPUT_FILENAME_BASE, VIDEO_EXTENSION)
    video_writer = cv2.VideoWriter(output_filename, FOURCC, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    if not video_writer.isOpened():
        print(f"Errore: Impossibile aprire VideoWriter per il file {output_filename}. Controlla i codec e i permessi.")
        picam2.stop()
        return

    print(f"Registrazione avviata. Salvataggio in: {output_filename}")
    print("Controlli PicarX:")
    print("  W: Avanti")
    print("  S: Indietro")
    print("  A: Sinistra")
    print("  D: Destra")
    print("  X: Stop motori (ma continua a registrare)")
    print("  Q: Esci e salva video")
    print("Premi un tasto...")

    try:
        while True:
            # Cattura frame dalla camera
            frame_rgb = picam2.capture_array() # Cattura come array RGB

            # OpenCV si aspetta BGR per la scrittura di default con alcuni codec,
            # ma VideoWriter con RGB888 dovrebbe gestire frame RGB direttamente.
            # Se il video salvato ha colori strani, decommenta la conversione:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Scrivi il frame nel file video
            video_writer.write(frame_bgr) # Scrivi il frame BGR

            # Mostra il frame (opzionale, ma utile per vedere cosa si sta registrando)
            # Ridimensiona per visualizzazione se necessario per adattarsi allo schermo
            # display_frame = cv2.resize(frame_bgr, (VIDEO_WIDTH // 2, VIDEO_HEIGHT // 2))
            cv2.imshow("PicarX Recording - Press Q to Quit", frame_bgr) # Mostra il frame a dimensione intera

            # Leggi l'input da tastiera in modo non bloccante se possibile,
            # o con un piccolo timeout per cv2.waitKey
            key_code = cv2.waitKey(1) & 0xFF # Attende 1ms per un tasto

            # Controllo con i tasti WASD
            if key_code != 255: # 255 significa nessun tasto premuto con waitKey(1)
                char_key = chr(key_code).lower() # Converti in carattere minuscolo
                print(f"Tasto premuto: {char_key}")

                if char_key == 'w':
                    px.forward(DRIVE_SPEED)
                elif char_key == 's':
                    px.backward(DRIVE_SPEED)
                elif char_key == 'a':
                    px.set_dir_servo_angle(-TURN_ANGLE) # Sinistra
                    # Potrebbe essere necessario un piccolo forward per vedere l'effetto della sterzata
                    # px.forward(DRIVE_SPEED / 2) 
                elif char_key == 'd':
                    px.set_dir_servo_angle(TURN_ANGLE)  # Destra
                    # px.forward(DRIVE_SPEED / 2)
                elif char_key == 'x': # Tasto per fermare i motori ma continuare a registrare
                    px.stop()
                elif char_key == 'q': # Tasto per uscire
                    print("Uscita richiesta...")
                    break
            # Se nessun tasto viene premuto, potresti voler fermare la macchina o mantenere l'ultimo stato.
            # Per semplicità, qui non facciamo nulla, quindi la macchina continua con l'ultimo comando
            # finché non ne arriva uno nuovo o 'x' o 'q'.
            # Se vuoi che si fermi quando non premi nulla, aggiungi:
            # else:
            #    px.stop() # O solo il movimento, non la sterzata
            #    px.stop_motors() # Metodo ipotetico, controlla la libreria Picarx

    except KeyboardInterrupt:
        print("Interruzione da tastiera rilevata (Ctrl+C).")
    finally:
        print("Fermata PicarX e chiusura risorse...")
        px.stop()
        if picam2.started:
            picam2.stop()
        if video_writer.isOpened():
            video_writer.release()
            print(f"Video salvato: {output_filename}")
        cv2.destroyAllWindows()
        print("Programma terminato.")

if _name_ == '_main_':
    main()