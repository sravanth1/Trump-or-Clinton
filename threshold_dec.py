import numpy
import cv2
from PIL import Image
import os
import pprint
import math
import simplejson as json
from hillary_trump import models as m
import pickle

def test_image(person,V,immean):
	face_cascade_path = "/home/sravanth/Desktop/precog_protos/haarcascade_frontalface_default.xml"
	face_cascade = cv2.CascadeClassifier(os.path.expanduser(face_cascade_path))
	scale_factor = 1.1
	min_neighbors = 5
	min_size = (30, 30)
	flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	li=["hillary","trump"]
	for i in xrange(len(li)):
		z=[]
		image_path = os.path.expanduser("/home/sravanth/Desktop/precog_protos/thres_"+li[i]+"_pics")
		for filename in os.listdir(image_path):
			image_init = os.path.expanduser(image_path+'/'+filename)
			image = cv2.imread(image_init,cv2.CV_LOAD_IMAGE_GRAYSCALE)
			faces = face_cascade.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors,
			minSize = min_size, flags = flags)
			for( x, y, w, h ) in faces:
				crop_img = Image.fromarray(image[y:y+h,x:x+w] ) # crop image and convert it to PIL from numpy
				new_img=crop_img.resize((100,100), Image.ANTIALIAS)#resizing
				#mode=numpy.dot(V,numpy.array(new_img-immean).flatten())
				a=numpy.dot(V,(numpy.array(new_img).flatten()-immean)).T
				mini=99999999999
				for j in xrange(len(person)):
					mini=min(numpy.linalg.norm(person[j]-a),mini)
				z.append(mini)
		print "i"
		facial_features=m.Facial_Vectors(name=li[i],vector=pickle.dumps(z),matrix=pickle.dumps(V),mean=pickle.dumps(immean))
		facial_features.save()
