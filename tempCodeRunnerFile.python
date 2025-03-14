import cv2
import numpy as np

# Inicializácia webkamery
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Chyba: Nepodarilo sa otvoriť webkameru.")
    exit(1)

# Nastavenie rozlíšenia
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

# Inicializácia OpenCV okna
cv2.namedWindow("Detekcia kružníc", cv2.WINDOW_NORMAL)

# Trackbary na úpravu parametrov
cv2.createTrackbar('Param1', 'Detekcia kružníc', 30, 200, lambda x: None)
cv2.createTrackbar('Param2', 'Detekcia kružníc', 15, 100, lambda x: None)
cv2.createTrackbar('MinRadius', 'Detekcia kružníc', 10, 100, lambda x: None)
cv2.createTrackbar('MaxRadius', 'Detekcia kružníc', 150, 300, lambda x: None)
cv2.createTrackbar('Canny1', 'Detekcia kružníc', 50, 255, lambda x: None)
cv2.createTrackbar('Canny2', 'Detekcia kružníc', 150, 255, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Chyba: Nepodarilo sa načítať snímku z kamery.")
        break
    
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
    edges = cv2.Canny(gray, canny1, canny2)
    
    # Detekcia kružníc pomocou Houghovej transformácie
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
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
    
    # Čakanie na kláves
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Ukončujem snímanie...")
        break

# Ukončenie
cap.release()
cv2.destroyAllWindows()
print("Hotovo.")
