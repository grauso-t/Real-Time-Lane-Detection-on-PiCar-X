import cv2
import numpy as np
from picamera2 import Picamera2
from picarx import Picarx
import time

# Inizializza Picar-X
px = Picarx()
speed = 10
px.forward(speed)

# Inizializza la PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()
time.sleep(1)  # Stabilizzazione della camera

delay = 33  # ~30 FPS

# Funzione nulla per le trackbar
def nothing(x):
    pass

# Trackbar per selezionare l'intervallo HSV
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    # Cattura immagine dalla PiCamera2
    image = picam2.capture_array()
    frame = cv2.resize(image, (640, 480))

    # Trasformazione prospettica (Top view)
    tl = (70, 220); bl = (0, 472); tr = (570, 220); br = (640, 472)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    topview = cv2.warpPerspective(frame, matrix, (640, 480))

    # Conversione HSV
    hsv = cv2.cvtColor(topview, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv)

    # Maschera in base ai valori HSV selezionati
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow("Mask", mask)

    # Calcolo istogramma
    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Istogramma visivo
    hist_img = np.zeros((300, 640, 3), dtype=np.uint8)
    if histogram.max() > 0:
        hist_norm = (histogram / histogram.max()) * 300
    else:
        hist_norm = histogram
    for x, val in enumerate(hist_norm):
        cv2.line(hist_img, (x, 300), (x, 300 - int(val)), (255, 255, 255), 1)
    cv2.imshow("Histogram", hist_img)

    # Sliding window
    y = 472
    lx = []
    rx = []
    while y > 0:
        img_l = mask[y - 40:y, max(0, left_base - 50):min(640, left_base + 50)]
        contours_l, _ = cv2.findContours(img_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_l:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                lx.append(left_base - 50 + cx)
                left_base = left_base - 50 + cx

        img_r = mask[y - 40:y, max(0, right_base - 50):min(640, right_base + 50)]
        contours_r, _ = cv2.findContours(img_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_r:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                rx.append(right_base - 50 + cx)
                right_base = right_base - 50 + cx

        y -= 40

    # Overlay con linee rilevate
    line_overlay = topview.copy()
    for x in lx:
        cv2.circle(line_overlay, (int(x), y), 5, (0, 0, 255), -1)  # sinistra
    for x in rx:
        cv2.circle(line_overlay, (int(x), y), 5, (255, 0, 0), -1)  # destra
    cv2.imshow("Detected Lines", line_overlay)

    # Controllo dello sterzo
    if lx and rx:
        left_mean = np.mean(lx)
        right_mean = np.mean(rx)
        lane_center = (left_mean + right_mean) / 2
        frame_center = 640 / 2
        deviation = lane_center - frame_center
        threshold = 20

        if abs(deviation) < threshold:
            px.set_dir_servo_angle(0)
        else:
            angle = -int((deviation / frame_center) * 30)
            angle = max(-30, min(30, angle))
            px.set_dir_servo_angle(angle)
    else:
        px.stop()
        print("Linee non trovate, fermo.")

    # Mostra immagini
    cv2.imshow("Original", frame)
    cv2.imshow("Top View", topview)

    # Esci con ESC
    if cv2.waitKey(delay) == 27:
        break

# Pulizia finale
picam2.close()
cv2.destroyAllWindows()
px.stop()
