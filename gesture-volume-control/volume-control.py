from os import curdir
import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(detectionConf=0.7, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]
vol, volBar, volPer = 0, 400, 0
area = 0
prevTime = 0
colorVol = (255,55,55)

while True:
    success, img = cap.read()
    # Find Hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        # Filter based on size
        #print(bbox)
        area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])//100
        print(area)
        if 200 < area < 1000:

            # Find Distance between index and Thumb
            length, img, lineInfo = detector.findDistance(4,8,img)

            # Convert Volume
            volBar = np.interp(length, [50, 250], [400, 150])
            volPer = np.interp(length, [50, 250], [0, 100])
            # print(vol)
            # Reduce resolution to make it smoother
            smoothness = 5
            volPer = smoothness * round(volPer/smoothness)
            # Check fingers up
            fingers = detector.fingersUp()
            #print(fingers)
            # If pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer/100, None)
                cv2.circle(img, (lineInfo[4],lineInfo[5]), 10, (255, 255, 55), cv2.FILLED)
                colorVol = (55,255,55)
            else:
                colorVol = (255,55,55)
            
    #else:
    #   volume.SetMasterVolumeLevel(-22, None)

    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), (255,55,55), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (55,55,255), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (15, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,55,55), 2)
    curVol = int(volume.GetMasterVolumeLevelScalar()*100)
    cv2.putText(img, f'Volume: {int(curVol)}', (415, 50), cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 2)
    
    # Frame rate
    curTime = time.time()
    fps = 1 / (curTime-prevTime)
    prevTime = curTime

    cv2.putText(img, f'FPS: {int(fps)}', (15, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,55,55), 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    