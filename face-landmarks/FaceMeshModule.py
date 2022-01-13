import cv2
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionConf=0.5, minTrackConf=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionConf = minDetectionConf
        self.minTrackConf = minTrackConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionConf, self.minTrackConf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                     #           0.7, (0, 255, 0), 1)

                    #print(id,x,y)
                    face.append([x,y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    prevTime = 0
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces)!= 0:
            print(faces[0])
        curTime = time.time()
        fps = 1 / (curTime-prevTime)
        prevTime = curTime
        cv2.putText(img, f'FPS: {int(fps)}', (15, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()