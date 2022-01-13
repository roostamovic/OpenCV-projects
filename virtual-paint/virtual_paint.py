import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
# WEBCAM
cap = cv2.VideoCapture(0)   # 0 means ID of 1st webcam in laptop
cap.set(3, frameWidth)   # 3 is ID of width
cap.set(4, frameHeight)   # 4 is ID of height
cap.set(10, 150)  # 10 is ID of brightness

myColors = [[5,107,0,19,255,255],
            [133,56,0,159,156,255],
            [57,76,0,100,255,255],
            [90,48,0,118,255,255]]

myColorValues = [[51, 153, 255],    # BGR
                 [255, 0, 255],
                 [0, 255, 0],
                 [255, 0, 0]]

myPoints = []  # [x, y, colorIdx]

def findColor(img, myColors, myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        #cv2.imshow(str(color[0]), mask)
        x,y = getContours(mask)
        cv2.circle(imgResult, (x,y), 15, myColorValues[count], cv2.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count += 1
    return newPoints

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            #cv2.drawContours(imgResult, cnt, contourIdx=-1, color=(200, 150, 100), thickness=3)
            perimeter = cv2.arcLength(cnt, closed=True)   # calculates the perimeter of the shape
            #print(f'{perimeter:.2f}')
            approx = cv2.approxPolyDP(cnt, epsilon=0.02*perimeter, closed=True) # finds the coordinates of the edges
            #print(len(approx))  # every result above 4 is a circle
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2, y

def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)

while True:
    success, img = cap.read()
    imgResult = img.copy()
    newPoints = findColor(img, myColors, myColorValues)
    
    if len(newPoints) != 0:
        for points in newPoints:
            myPoints.append(points)
    
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorValues)

    cv2.imshow('Virtual Paint', imgResult)
    if cv2.waitKey(1) & 0xFF  == ord('q'):
        break