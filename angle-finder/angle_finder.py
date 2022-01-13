import cv2
import math

img = cv2.imread('test.png')
pointList = []


def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointList)
        if size != 0 and size % 3 != 0:
            cv2.line(img, tuple(pointList[round((size-1)/3)*3]), (x,y), (0,0,255), 2)
        cv2.circle(img, (x,y), 3, (0,0,255), cv2.FILLED)
        pointList.append([x,y])
        print(pointList)
        #print(x, y)


def gradient(pt1, pt2):
    return ((pt2[1] - pt1[1]) / (pt2[0] - pt1[0]))


def getAngle(pointList):
    pt1, pt2, pt3 = pointList[-3:]
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angleRad = abs(math.atan((m2-m1)/(1+m1*m2)))
    angle = round(math.degrees(angleRad))
    #print(angle)
    cv2.putText(img, str(angle), (pt1[0]-30, pt1[1]-20), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,255), 2)


while True:

    if len(pointList) % 3 == 0 and len(pointList) != 0:
        getAngle(pointList)

    cv2.imshow('Angles', img)
    cv2.setMouseCallback('Angles', mousePoints)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        pointList = []
        img = cv2.imread('test.jpg')