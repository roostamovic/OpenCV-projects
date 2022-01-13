import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
prevTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                ih,iw,ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)


    curTime = time.time()
    fps = 1 / (curTime-prevTime)
    prevTime = curTime

    cv2.putText(img, f'FPS: {int(fps)}', (15, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,55,55), 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break