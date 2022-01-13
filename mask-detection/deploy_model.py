from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

	
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)



	faces = []
	locations = []
	preds = []

	
	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]

		
		if confidence > 0.5:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(start_X, start_Y, end_X, end_Y) = box.astype("int")

			
			(start_X, start_Y) = (max(0, start_X), max(0, start_Y))
			(end_X, end_Y) = (min(w-1, end_X), min(h-1, end_Y))

			
			face = frame[start_Y:end_Y, start_X:end_X]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			
			faces.append(face)
			locations.append((start_X, start_Y, end_X, end_Y))

	
	if len(faces) > 0:
		
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)


	return (locations, preds)


prototxt_Path = r"face_detector\\deploy.prototxt"
weights_Path = r"face_detector\\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxt_Path, weights_Path)


maskNet = load_model("mask_detector.model")


print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while True:
	
	success, frame = cap.read()
	#frame = imutils.resize(frame, width=400)

	
	
	(locations, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	
	for (box, pred) in zip(locations, preds):
		
		(start_X, start_Y, end_X, end_Y) = box
		(mask, withoutMask) = pred

		
		label = "MASKED" if mask > withoutMask else "NO MASK"
		color = (0, 255, 0) if label == "MASKED" else (0, 0, 255)

		
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		
		cv2.putText(frame, label, (start_X, start_Y-10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (start_X, start_Y), (end_X, end_Y), color, 2)

	
	cv2.imshow("Frame", frame)


	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cv2.destroyAllWindows()