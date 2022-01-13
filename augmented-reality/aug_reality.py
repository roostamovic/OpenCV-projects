import numpy as np
import cv2

cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('image.jpg')
myVideo = cv2.VideoCapture('video.mp4')

detection = False
frameCounter = 0

imgTarget = cv2.resize(imgTarget, (550, 690))

success, imgVideo = myVideo.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

orb = cv2.ORB_create(nfeatures=1000)
key_point1, descriptor1 = orb.detectAndCompute(imgTarget, None)
#imgTarget = cv2.drawKeypoints(imgTarget, key_point1, None)

####################################################################################################

def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

#####################################################################################################

while True:
    success, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    key_point2, descriptor2 = orb.detectAndCompute(imgWebcam, None)
    #imgWebcam = cv2.drawKeypoints(imgWebcam, key_point2, None)

    if detection == False:
        myVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        if frameCounter == myVideo.get(cv2.CAP_PROP_FRAME_COUNT):
            myVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, imgVideo = myVideo.read()
        imgVideo = cv2.resize(imgVideo, (wT, hT))
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    #print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget, key_point1, imgWebcam, key_point2, good, None, flags=2)

    if len(good) > 20:
        detection = True
        srcPoints = np.float32([key_point1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPoints = np.float32([key_point2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        matrix, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5)

        pts = np.float32([[0,0], [0,hT], [wT,hT], [wT,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(src=pts, m=matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (0,255,255), 3)

        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255,255,255))
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)
        #imgStacked = stackImages(([imgWebcam, imgVideo, imgTarget], [imgFeatures, imgWarp, imgAug]), 0.5)

    #cv2.imshow('STACKED', stackImages)
    #cv2.imshow('Warp', imgWarp)
    #cv2.imshow('PloyLines', img2)
    #cv2.imshow('Features', imgFeatures)
    #cv2.imshow('Image', imgTarget)
    #cv2.imshow('Video', imgVideo)
    #cv2.imshow('Webcam', imgWebcam)
    cv2.imshow('Augmented', imgAug)
    cv2.waitKey(1)
    frameCounter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
