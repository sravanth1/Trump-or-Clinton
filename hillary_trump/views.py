from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from django.shortcuts import render_to_response

import numpy
import cv2
from PIL import Image
import os
import pprint
import math
import simplejson as json
import pickle
import models
import forms 
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings


@csrf_exempt
def test_image(request):
	models.ImageAttachment.objects.all().delete()
	if request.method == 'POST':
		newdoc = models.ImageAttachment(file=request.FILES['myfile'],path=settings.IMAGE_UPLOAD_PATH,name=request.FILES['myfile'].name)
		newdoc.save()
	documents = models.ImageAttachment.objects.all()
	di={}
	di['hillary']=False
	di['trump']=False
	if documents:
		face_cascade_path = "/home/sravanth/Desktop/precog_protos/haarcascade_frontalface_default.xml"
		face_cascade = cv2.CascadeClassifier(os.path.expanduser(face_cascade_path))
		scale_factor = 1.1
		min_neighbors = 5
		min_size = (30, 30)
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		image_path=os.path.expanduser(documents[0].path+'/'+documents[0].name)
		print "##########################################"
		print image_path
		image = cv2.imread(image_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)
		faces = face_cascade.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors,minSize = min_size, flags = flags)
		people=models.Facial_Vectors.objects.all()
		minimum=[{'hillary':999999999,'trump':999999999}]*len(faces)
		for i in xrange(len(people)):
			name=people[i].name
			person=pickle.loads(people[i].vector)
			V=pickle.loads(people[i].matrix)
			immean=pickle.loads(people[i].mean)
			#pprint.pprint(pickle.loads(people[i].vector))
			cnt=0
			#print len(faces)
			for( x, y, w, h ) in faces:
				crop_img = Image.fromarray(image[y:y+h,x:x+w] ) # crop image and convert it to PIL from numpy
				new_img=crop_img.resize((100,100), Image.ANTIALIAS)#resizing
				#########     AFTER    DATABASE CREATION ###############
				a=numpy.dot(V,(numpy.array(new_img).flatten()-immean)).T
				mini=99999999999
				for j in xrange(len(person)):
					mini=min(numpy.linalg.norm(person[j]-a),mini)
				print mini
				minimum[cnt][name]=mini
				cnt+=1
			##########################################################
		for i in xrange(len(minimum)):
			if minimum[i]['hillary'] < minimum[i]['trump']:
				if minimum[i]['hillary']<1650:
					print minimum[i]['hillary']
					di['hillary']=True
			if minimum[i]['trump'] < minimum[i]['hillary']:
				if minimum[i]['trump']<1650:
					print minimum[i]['trump']
					di['trump']=True

		pprint.pprint(di)

	return render_to_response('basic.html',di)
	#print "out of testing-------------------------"