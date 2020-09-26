import cv2
from pygame import mixer
mixer.init()
mixer.music.load('swiftly.mp3')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# capture frames from a camera
cap = cv2.VideoCapture(0)
img_counter = 0

# loop runs if capturing has been initialized.
while True:
	# reads frames from a camera
	ret, img = cap.read()
	# convert to gray scale of each frames
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detects faces of different sizes in the input image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		# To draw a rectangle in a face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		mixer.music.play()

		# Detects eyes of different sizes in the input image
		eyes = eye_cascade.detectMultiScale(roi_gray)

		# To draw a rectangle in eyes
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

	x = 0
	y = 20
	text_color = (0, 255, 0)
	cv2.putText(img, "Press Space To Save", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, thickness=2)
	a = 0
	b = 40
	text_color = (255, 0, 0)
	cv2.putText(img, "Press Esc To Exit", (a, b), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, thickness=2)

	# Display an image in a window
	cv2.imshow('img',img)


	k = cv2.waitKey(1)
	if k == 27:
		# ESC pressed
		cv2.imwrite('face_detected.png', img)
		print("Escape hit, closing...")
		break
	elif k == 32:
		# SPACE pressed
		img_name = "opencv_frame_{}.png".format(img_counter)
		cv2.imwrite(img_name, img)
		print("{} written!".format(img_name))
		img_counter += 1

# Close the window
cap.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()