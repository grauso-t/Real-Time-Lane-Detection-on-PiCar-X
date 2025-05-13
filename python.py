import cv2
import numpy as np
from picarx import Picarx
import time

px = Picarx()

# Imposta la velocità iniziale
speed = 10
px.forward(speed)

# Video in tempo reale dalla camera (usa 0 o il path corretto)
video = cv2.VideoCapture(0)  # Usa 0 se è la camera integrata

fps = video.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback
delay = int(1000 / fps)

# Trackbar per l'HLS
def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    success, image = video.read()
    if not success:
        break

    frame = cv2.resize(image, (640, 480))

    # ROI e trasformazione prospettica
    tl = (70,220); bl = (0,472); tr = (570,220); br = (640,472)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    topview = cv2.warpPerspective(frame, matrix, (640, 480))

    # HSV conversione e maschera
    hsv = cv2.cvtColor(topview, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower, upper)

    # Istogramma
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding Window
    y = 472
    lx = []
    rx = []
    msk = mask.copy()

    while y > 0:
        img_l = mask[y-40:y, left_base-50:left_base+50]
        contours_l, _ = cv2.findContours(img_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_l:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                lx.append(left_base - 50 + cx)
                left_base = left_base - 50 + cx

        img_r = mask[y-40:y, right_base-50:right_base+50]
        contours_r, _ = cv2.findContours(img_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_r:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                rx.append(right_base - 50 + cx)
                right_base = right_base - 50 + cx

        y -= 40

    # Controllo dello sterzo
    if lx and rx:
        left_mean = np.mean(lx)
        right_mean = np.mean(rx)
        lane_center = (left_mean + right_mean) / 2
        frame_center = 640 / 2

        deviation = lane_center - frame_center
        threshold = 20  # tolleranza

        # Applica la sterzata in base allo scarto
        if abs(deviation) < threshold:
            px.set_dir_servo_angle(0)  # dritto
        else:
            # angolo massimo +/-30°
            angle = -int((deviation / frame_center) * 30)
            angle = max(-30, min(30, angle))
            px.set_dir_servo_angle(angle)

    else:
        # se una o entrambe le linee non sono visibili, ferma
        px.stop()
        print("Linee non trovate, fermo.")

    # Mostra le finestre
    cv2.imshow('Original', frame)
    cv2.imshow("Top View", topview)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(delay) == 27:
        break

video.release()
cv2.destroyAllWindows()
px.stop()