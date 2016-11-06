import cv2
import os
import sys
from string import Template
import pprint
from PIL import Image
import numpy
# first argument is the haarcascades path
face_cascade_path = "/home/sravanth/Desktop/precog_protos/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(os.path.expanduser(face_cascade_path))
scale_factor = 1.1
min_neighbors = 5
min_size = (30, 30)
flags = cv2.cv.CV_HAAR_SCALE_IMAGE
path="/home/sravanth/Desktop/precog_protos/"
width= 100
height= 100
li=["hillary_train_pics","trump_train_pics"]
for i in xrange(len(li)):
	cnt=0
	cnt_mod=0
	for filename in os.listdir(path+li[i]):
		image_path = os.path.expanduser(path+li[i]+'/'+filename)
		image = cv2.imread(image_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)
		#pprint.pprint(image)
		cnt+=1
		faces = face_cascade.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors,
		minSize = min_size, flags = flags)
		if len(faces) >0:
			cnt_mod+=1
		for( x, y, w, h ) in faces:
			crop_img = Image.fromarray(image[y:y+h,x:x+w] ) # crop image and convert it to PIL from numpy
			new_img=crop_img.resize((width,height), Image.ANTIALIAS)#resizing image
			#cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
			outfname = "/home/sravanth/Desktop/precog_protos/mongodb_"+li[i]+"/%s_face.jpg" % os.path.basename(filename)
			cv2.imwrite(os.path.expanduser(outfname), numpy.array(new_img)) #only numpy arrays can be written
	print cnt
	print cnt_mod


# im2 = im1.resize((width, height), Image.NEAREST)      # use nearest neighbour
# im3 = im1.resize((width, height), Image.BILINEAR)     # linear interpolation in a 2x2 environment
# im4 = im1.resize((width, height), Image.BICUBIC)      # cubic spline interpolation in a 4x4 environment
# im5 = im1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter