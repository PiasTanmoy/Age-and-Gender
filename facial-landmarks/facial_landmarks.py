# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
'''

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
image = cv2.imread('images/example_01.jpg')
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

ratio1 = 0
ratio2 = 0
ratio3 = 0
ratio4 = 0
ratio5 = 0
Point46_x = 0
Point46_y = 0
Point43_x = 0
Point43_y = 0
Point40_x = 0
Point40_y = 0
Point37_x = 0
Point37_y = 0
Point34_x = 0
Point34_y = 0
Point49_x = 0
Point49_y = 0
Point55_x = 0
Point55_y = 0
Point9_x = 0
Point9_y = 0

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	i=0
	for (x, y) in shape:
		#print(i)
		#i++
		cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
		print("Point ", i+1," : ",x,y)

		#finding first ratio
		if(i+1 == 46):
			Point46_x = x

		if (i+1 == 43):
			Point43_x = x
			
		if(i+1 == 40):
			Point40_x = x
			Point40_y = y
			
		if(i+1 == 37):
			Point37_x = x

		if(i+1 == 34):
			Point34_y = y

		if(i+1 == 49):
			Point49_x = x
			Point49_y = y 

		if(i+1 == 55):
			Point55_x = x
			Point55_y = y 

		if(i+1 == 9):
			Point9_x = x
			Point9_y = y

			
			
		#print (x,y)
		i += 1
		#print (y)

left_eye_middle_point_x = Point37_x + ((Point40_x - Point37_x)/2)
left_eye_middle_point_y = Point40_y
right_eye_middle_point_x = Point43_x + ((Point46_x - Point43_x)/2)
right_eye_middle_point_y = Point43_y
lefteye_righteye = right_eye_middle_point_x - left_eye_middle_point_x
eye_nose = Point34_y - Point40_y
ratio1 = lefteye_righteye / eye_nose
#print(eye_nose)
print("Ratio 1 : ",ratio1) 

#Ratio2 finding

mouth_middle_x = Point49_x + ((Point55_x - Point49_x)/2)
mouth_middle_y = Point49_y + ((Point55_y - Point49_y)/2)
eye_mouth = mouth_middle_y - Point40_y

ratio2 = lefteye_righteye / eye_mouth
print("Ratio 2 : ",ratio2)

#Ratio3 finding
eye_chin = Point9_y - Point40_y
ratio3 = lefteye_righteye/eye_chin
print("Ratio 3 : ",ratio3)

ratio4 = eye_nose/eye_mouth
print("Ratio 4 : ",ratio4)

ratio5 = eye_mouth/eye_chin
print("Ratio 5 : ",ratio5)

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
