from ximea import xiapi
import cv2
import os
import numpy as np

# Inicializácia kamery XIMEA
cam = xiapi.Camera()
try:
    cam.open_device()
    print("Kamera XIMEA bola úspešne otvorená.")
except Exception as e:
    print(f"Chyba pri otváraní kamery: {e}")
    exit(1)

# Nastavenie kamery
cam.set_exposure(100000)
cam.set_param('imgdataformat', 'XI_RGB24')
cam.set_param('auto_wb', 1)
print(f'Expozícia bola nastavená na {cam.get_exposure()} us')

# Inicializácia snímky
img = xiapi.Image()
cam.start_acquisition()
print('Začiatok snímania...')

# Inicializácia OpenCV okna
cv2.namedWindow("Detekcia kružníc", cv2.WINDOW_NORMAL)

# cv2.createTrackbar(name, window_name, value, max_value, callback_function)

# Trackbary na úpravu parametrov
cv2.createTrackbar('Param1', 'Detekcia kružníc', 115, 200, lambda x: None)
cv2.createTrackbar('Param2', 'Detekcia kružníc', 58, 100, lambda x: None)
cv2.createTrackbar('MinRadius', 'Detekcia kružníc', 0, 100, lambda x: None)
cv2.createTrackbar('MaxRadius', 'Detekcia kružníc', 300, 300, lambda x: None)
cv2.createTrackbar('Canny1', 'Detekcia kružníc', 0, 255, lambda x: None)
cv2.createTrackbar('Canny2', 'Detekcia kružníc', 50, 255, lambda x: None)

while True:
    key = cv2.waitKey(1) & 0xFF  # Čakanie na kláves
    if key == ord('q'):
        print("Ukončujem snímanie...")
        break  # Ukončí cyklus

    # Získanie snímky z kamery
    cam.get_image(img)
    frame = img.get_image_data_numpy()

    # Prevod do odtieňov sivej
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 2)  # Silnejšie rozmazanie

    # Načítanie hodnôt z trackbarov
    param1 = cv2.getTrackbarPos('Param1', 'Detekcia kružníc')
    param2 = cv2.getTrackbarPos('Param2', 'Detekcia kružníc')
    minRadius = cv2.getTrackbarPos('MinRadius', 'Detekcia kružníc')
    maxRadius = cv2.getTrackbarPos('MaxRadius', 'Detekcia kružníc')
    canny1 = cv2.getTrackbarPos('Canny1', 'Detekcia kružníc')
    canny2 = cv2.getTrackbarPos('Canny2', 'Detekcia kružníc')
    

    # Cannyho detektor hrán
    edges = cv2.Canny(gray, canny1, canny2)  # 50 150
    # canny 1 = 0
    # canny 2 = 25
    # Detekcia kružníc pomocou Houghovej transformácie
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius
    )

    # Ak sa našli kružnice, vykresli ich
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Zaokrúhlenie hodnôt
        for circle in circles[0, :]:
            x, y, r = circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # Kružnica
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Stred
            cv2.putText(frame, f"Priemer: {2*r}px", (x - 40, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Zobrazenie výsledku
    combined = np.hstack((cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), frame))  # Spojenie hrán a pôvodného obrazu
    cv2.imshow("Detekcia kružníc", combined)

# Ukončenie
print('Zastavujem snímanie...')
cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()
print('Hotovo.')
# todo canny sa nemeni pri posuvani trackbaru