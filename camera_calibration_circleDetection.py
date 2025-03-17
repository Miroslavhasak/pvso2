import numpy as np
import cv2 as cv
import glob
from ximea import xiapi

#[][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][] Calibration

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('CalibrationPictures/*.jpg')


#Detection
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow('img', gray)
    cv.waitKey(25)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (5,7), cv.CALIB_CB_ADAPTIVE_THRESH)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (5,7), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(25)

cv.destroyAllWindows()

gray = cv.cvtColor(cv.imread("CalibrationPictures/calibration_saved_image_0.jpg"), cv.COLOR_BGR2GRAY)

#Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread('CalibrationPictures/calibration_saved_image_0.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

print(newcameramtx)
print(roi)

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult1.png', dst)

# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)\

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult2.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

#[][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][] Calibration END



#[][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][] Circle Detection


# Inicializácia webkamery
cameraXimea=False

if cameraXimea:
    cam = xiapi.Camera()

    #start communication
    #to open specific device, use:
    #cam.open_device_by_SN('41305651')
    #(open by serial number)
    print('Opening first camera...')
    cam.open_device()

    #settings
    #cam.set_exposure(100000)
    cam.set_exposure(50000)
    cam.set_param('imgdataformat','XI_RGB32')
    cam.set_param('auto_wb', 1)
    print('Exposure was set to %i us' %cam.get_exposure())

    #create instance of Image to store image data and metadata
    img = xiapi.Image()

    #start data acquisition
    print('Starting data acquisition...')
    cam.start_acquisition()


# Inicializácia OpenCV okna
cv.namedWindow("Detekcia kružníc", cv.WINDOW_NORMAL)

# Trackbary na úpravu parametrov
cv.createTrackbar('Param1', 'Detekcia kružníc', 115, 200, lambda x: None)
cv.createTrackbar('Param2', 'Detekcia kružníc', 58, 100, lambda x: None)
cv.createTrackbar('MinRadius', 'Detekcia kružníc', 0, 100, lambda x: None)
cv.createTrackbar('MaxRadius', 'Detekcia kružníc', 500, 500, lambda x: None)
cv.createTrackbar('Canny1', 'Detekcia kružníc', 0, 255, lambda x: None)
cv.createTrackbar('Canny2', 'Detekcia kružníc', 50, 255, lambda x: None)

images = glob.glob('CircleDetectionTestingPictures/*.jpg')

while True:
    for fname in images:
        if cameraXimea:
            cam.get_image(img)
            image = img.get_image_data_numpy()
            image = cv.resize(image, (700, 700))
        else:
            image = cv.imread(fname)

        image = cv.undistort(image, mtx, dist, None, newcameramtx)

        # Prevod do odtieňov sivej
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (11, 11), 2)  # Silnejšie rozmazanie

        # Načítanie hodnôt z trackbarov
        param1 = cv.getTrackbarPos('Param1', 'Detekcia kružníc')
        param2 = cv.getTrackbarPos('Param2', 'Detekcia kružníc')
        minRadius = cv.getTrackbarPos('MinRadius', 'Detekcia kružníc')
        maxRadius = cv.getTrackbarPos('MaxRadius', 'Detekcia kružníc')
        canny1 = cv.getTrackbarPos('Canny1', 'Detekcia kružníc')
        canny2 = cv.getTrackbarPos('Canny2', 'Detekcia kružníc')

        # Cannyho detektor hrán
        edges = cv.Canny(gray, canny1, canny2)

        # Detekcia kružníc pomocou Houghovej transformácie
        circles = cv.HoughCircles(
            gray, cv.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius
        )

        # Ak sa našli kružnice, vykresli ich
        if circles is not None:
            circles = np.uint16(np.around(circles))  # Zaokrúhlenie hodnôt
            for circle in circles[0, :]:
                x, y, r = circle
                cv.circle(image, (x, y), r, (0, 255, 0), 2)  # Kružnica
                cv.circle(image, (x, y), 2, (0, 0, 255), 3)  # Stred
                cv.putText(image, f"Priemer: {2 * int(r)}px", (int(x) - 40, int(y) - int(r) - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Zobrazenie výsledku
        combined = np.hstack((cv.cvtColor(edges, cv.COLOR_GRAY2BGR), image))  # Spojenie hrán a pôvodného obrazu
        cv.imshow("Detekcia kružníc", combined)

        # Čakanie na kláves
        key = cv.waitKey(500) & 0xFF
        if key == ord('q'):
            print("Ukončujem snímanie...")
            break

# Ukončenie
cam.stop_acquisition()
cam.close_device()

cv.destroyAllWindows()
print("Hotovo.")

#[][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][] Circle Detection