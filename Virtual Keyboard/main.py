import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
from pynput.keyboard import Controller
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.77)
keys = [['Q','W','E','R','T','Y','U','I','O','P'],
        ['A','S','D','F','G','H','J','K','L',';'],
        ['Z','X','C','V','B','N','M',',','.','/']]

finalText = ''
keyboard = Controller()

def drawALL(img, buttonList):
    for button in buttonList: 
        x,y = button.pos
        w,h = button.size
        cvzone.cornerRect(img, (x,y,w,h), 20, rt=0, colorC=(0,255,255))
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), cv2.FILLED)
        cv2.putText(img, button.text, (x+13, y+33), cv2.FONT_HERSHEY_COMPLEX, 1.1, (255,255,255), 2)
    return img

class Button():
    def __init__(self, pos, text, size=[50,50]):
        self.pos = pos
        self.text = text
        self.size = size
        
buttonList = []
for i in range(len(keys)):
        for x, key in enumerate(keys[i]):
            buttonList.append(Button([60*x+25, 60*i+50], key))

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img)  
    img = drawALL(img, buttonList)

    if lmList:
        for button in buttonList:
            x,y = button.pos
            w,h = button.size

            if x < lmList[8][0] < x+w and y < lmList[8][1] < y+h:
                cv2.rectangle(img, (x,y), (x+w,y+h), (175,0,175), cv2.FILLED)
                cv2.putText(img, button.text, (x+13, y+33), cv2.FONT_HERSHEY_COMPLEX, 1.1, (255,255,255), 2)
                l,_,_ = detector.findDistance(8, 12, img, draw=False)
                if l < 22:
                    keyboard.press(button.text)
                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), cv2.FILLED)
                    cv2.putText(img, button.text, (x+13, y+33), cv2.FONT_HERSHEY_COMPLEX, 1.1, (255,255,255), 2)
                    finalText += button.text
                    sleep(0.15)
                

    cv2.rectangle(img, (25,275), (615,375), (175,0,175), cv2.FILLED)
    cv2.putText(img, finalText, (50,325), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255,255,255), 2)
    
    cv2.imshow('CAP', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break