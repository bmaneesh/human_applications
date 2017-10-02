import cv2
import numpy as np
import imutils
import sys


class human_cum_face:
	def __init__(self,name):
		self.filename = name

	def extract_edge(self):
		im = cv2.imread(self.filename)
		cv2.namedWindow("preview")
		# im_edgy = cv2.Canny(im,100,200)
		cv2.imshow("preview",im)

		face_cascade = cv2.CascadeClassifier()
		face_cascade.load('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
		# print face_cascade.empty()
		faces = face_cascade.detectMultiScale(im, 3, 10)
		cv2.imshow("preview",im)
		cv2.waitKey()
		for face in faces:
			cv2.rectangle(im,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(0,255,0),2)
		cv2.imshow("preview",im)
		cv2.waitKey()
		cv2.destroyAllWindows()
		return 0

	def human_det(self,filename='temp.jpeg'):
		winStride = (8,8)
		padding = (16,16)
		meanShift = 0
		scale = 1.05

		im = cv2.imread(filename)
		im = imutils.resize(im,400,max(im.shape[1],700))
		cv2.namedWindow("input")
		cv2.imshow("input",im)
		cv2.waitKey()
		hog_detector = cv2.HOGDescriptor()
		hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		(rects, weights) = hog_detector.detectMultiScale(im, winStride=winStride,
		padding=padding, scale=scale, useMeanshiftGrouping=meanShift)
		print rects

		if len(rects) > 0:
			for win in rects:
				if len(win) > 0:
					cv2.rectangle(im,(win[0],win[1]),(win[0]+win[2],win[1]+win[3]), (0, 255, 0), 2)
					cv2.imshow("input",im)
					cv2.waitKey(30)
		# cv2.destroyAllWindows()
		return 0

	def human_from_vid(self):
		cap =cv2.VideoCapture(self.filename)
		cap.open(self.filename)
		print self.filename+' Opened'
		# cv2.namedWindow('frame')
		while(cap.isOpened()):
			ret, frame = cap.read()
			cv2.imwrite('temp.jpeg',frame)
			if ret == True:
				# cv2.imshow('frame',frame)
				self.human_det('./temp.jpeg')
			else:
				break
			# cv2.waitKey(30)
		cap.release()
		cv2.destroyAllWindows()
		return 0

if __name__ =="__main__":
	filename = sys.argv[1]
	im = human_cum_face(filename)
	# print im
	# im.human_det()
	# im.extract_edge()
	vid = human_cum_face(filename)
	vid.human_from_vid()