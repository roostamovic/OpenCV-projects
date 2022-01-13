import cv2
import pytesseract
import easyocr

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
plateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
img = cv2.imread('PASSPORT.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

numberPlates = plateCascade.detectMultiScale(img, 1.1, 10)
#print(pytesseract.image_to_string(img))

text = ' '
reader = easyocr.Reader(['en'])
result = reader.readtext(img)
for i in range(len(result)):
    text = result[i][-2]
    print(result[i][0][1][0])
    x,y,w,h = result[i][0][0][0], result[i][0][0][1], result[i][0][1][0], result[i][0][2][1]
    cv2.rectangle(img, (x,y), (w,h), (255,0,0), 3)
    cv2.putText(img, text, (x+1, y-1), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255), 1)
    
#for x,y,w,h in numberPlates: 
#    imgRoi = img[y:y+h, x:x+w]
#    cv2.imshow('RoI', imgRoi)
#    result = reader.readtext(imgRoi)
#    text = result[0][-2]
#    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 3)
#    cv2.putText(img, text, (x+10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)



#cv2.imshow('Image', img)
#cv2.waitKey(0)


## Detecting Characters
#h_Img, w_Img, _ = img.shape
#boxes = pytesseract.pytesseract.image_to_boxes(img)
#for b in boxes.splitlines():
    #print(b)
#    b = b.split(' ')
#    #print(b)
#    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
#    cv2.rectangle(img, (x,h_Img-y), (w,h_Img-h), (0,0,255), 2)
#    cv2.putText(img, b[0], (x+5, h_Img-y+33), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)


# Detecting Words
#h_Img, w_Img, _ = img.shape
#data = pytesseract.image_to_data(img)
#print(data)
#for a, b in enumerate(data.splitlines()):
#    if a != 0:
#        b = b.split()
#        #print(b)
#        if len(b) == 12:
#            x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])    
#            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
#            cv2.putText(img, b[11], (x+10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
#            print(b[11])


#cv2.imshow('Image', img)
#cv2.waitKey(0)


# Detecting Digits
#h_Img, w_Img, _ = img.shape
#conf = r'--oem 3 --psm 6 outputbase digits'
#digits = pytesseract.image_to_boxes(img, config=conf)
#print(data)
#for b in digits.splitlines():
    #print(b)
#    b = b.split(' ')
    #print(b)
#    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
#    cv2.rectangle(img, (x,h_Img-y), (w,h_Img-h), (0,0,255), 2)
#    cv2.putText(img, b[0], (x+5, h_Img-y+33), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)


cv2.imshow('Image', img)
cv2.waitKey(0)


