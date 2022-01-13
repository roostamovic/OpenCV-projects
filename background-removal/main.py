import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)


segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()


listImg = os.listdir('Images')
#print(listImg)

imgList = []
for imgPath in listImg:
    img = cv2.imread(f'Images/{imgPath}')
    imgList.append(img)
#print(len(imgList))

indexImg = 0


while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.2)    # if threshold > 0.95, it cuts everything in the image
                                                                    # even objects that are located in the image
    
    #imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    #_, imgStacked = fpsReader.update(imgStacked, color=(0,0,255))
    cv2.imshow('Webcam', imgOut)
    
    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImg > 0:
            indexImg -= 1
        else:
            indexImg = len(imgList)-1
    elif key == ord('d'):
        if indexImg < len(imgList)-1:
            indexImg += 1
        else:
            indexImg = 0
    elif key == ord('q'):
        break