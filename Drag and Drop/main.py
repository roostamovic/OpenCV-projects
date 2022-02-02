import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np
import random

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.77)
colorRec = (255,0,255)


class DragRect():
    def __init__(self, posCenter,  text, size=[150,150]):
        self.posCenter = posCenter
        self.size = size
        self.text = text
    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        if cx-w//2<cursor[0]<cx+w//2 and cy-h//2<cursor[1]<cy+h//2:
                self.posCenter = cursor
rectList = []
textList = ['Jarvis, repair my costume', "Jarvis, speak out today's news", 'Jarvis, clear the table',
            'Jarvis, find the best way', 'Jarvis, make a robust decision', 'Jarvis, save energy']

for i in range(3):
    text = random.choice(textList)
    rectList.append(DragRect([i*200+120,100], text))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img)
    if lmList:
        l, _, _ = detector.findDistance(8, 12, img, draw=False)
        #print(l)
        if l < 30:
            cursor = lmList[8]
            for rect in rectList:
                rect.update(cursor)
    # Draw Solid
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(img, (cx-w//2,cy-h//2), (cx+w//2,cy+h//2), colorRec, cv2.FILLED)
        cv2.putText(img, rect.text, (cx-w//2, cy), cv2.FONT_HERSHEY_COMPLEX, 0.33, (0,255,0), 1)
        #cvzone.cornerRect(img, (cx-w//2,cy-h//2,w,h), 20, rt=0, colorC=(0,255,255))
    
    # Draw Transparency
    imgNew = np.zeros_like(img, np.uint8)
    
    #for rect in rectList:
    #    cx, cy = rect.posCenter
    #    w, h = rect.size
    #    cv2.rectangle(imgNew, (cx-w//2,cy-h//2), (cx+w//2,cy+h//2), colorRec, cv2.FILLED)
    #    cv2.putText(imgNew, rect.text, (cx-w//2, cy), cv2.FONT_HERSHEY_COMPLEX, 0.33, (255,0,0), 1)
    #imgOut = img.copy()
    #alpha = 0.1
    #mask = imgNew.astype(bool)
    #imgOut[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]
    cv2.imshow('CAP', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break