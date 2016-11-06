from PIL import Image
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pprint
import math
import cv2
from sklearn.decomposition import PCA
import threshold_dec as td
from hillary_trump import models as mod
import pickle

def pca1(X):
	# Principal Component Analysis
	# inumpyut: X, matrix with training data as flattened arrays in rows
	# return: projection matrix (with important dimensions first),
	# variance and mean

	#get dimensions
	num_data,dim= X.shape
	#center data
	X=numpy.array(X,dtype=numpy.float64)
	#X=cl.clustering(X)
	mean_X = numpy.array(X.mean(axis=0))
	stand_dev=numpy.array(X.std(axis=0))
	X-=mean_X
	#print X
	if dim>100:
		print 'PCA - compact trick used'
		M = numpy.dot(X,X.T)#covariance matrix
		e,EV = numpy.linalg.eigh(M) #eigenvalues and eigenvectors
		tmp = numpy.dot(X.T,EV).T #this is the compact trick
		V = tmp[::-1] #reverse since last eigenvectors are the ones we want
		S = numpy.sqrt(e)[::-1] #reverse since eigenvalues are in increasing order
		for i in range(V.shape[1]):
			V[:,i] /= S
	else:
		print 'PCA - SVD used'
		U,S,V = numpy.linalg.svd(X)
		V = V[:num_data] #only makes sense to return the first num_data
  	#return the projection matrix, the variance and the mean
	return V,S,mean_X,X

def main():
	li=["mongodb_hillary_train_pics","mongodb_trump_train_pics"]
	hillary=[]
	trump=[]
	for i in xrange(len(li)):
		images_path="/home/sravanth/Desktop/precog_protos/"+li[i]
		imlist=[]
		for file in os.listdir(images_path):
			imlist.append(images_path+'/' +file)
		im = numpy.array(Image.open(imlist[0])) #open one image to get the size
		m,n = im.shape[0:2] #get the size of the images
		#print m,n
		imnbr = len(imlist) #get the number of images
		#create matrix to store all flattened images
		immatrix = numpy.array([numpy.array(Image.open(imlist[j])).flatten() for j in xrange(0,imnbr)],'f')
		#perform PCA
		#immatrix=cl.clustering(immatrix)
		V,S,immean,immatrix = pca1(immatrix)
		#print immatrix.shape
		num=20
		if(i==0):
			hillary=numpy.dot(V[0:5],(immatrix[0:20]).T).T
			facial_features=mod.Facial_Vectors(name="hillary",vector=pickle.dumps(hillary),matrix=pickle.dumps(V[0:5]),mean=pickle.dumps(immean))
			facial_features.save()
			#td.test_image(hillary,V[0:5],immean)
		else:
			trump=numpy.dot(V[0:5],(immatrix[0:20]).T).T
			facial_features=mod.Facial_Vectors(name="trump",vector=pickle.dumps(trump),matrix=pickle.dumps(V[0:5]),mean=pickle.dumps(immean))
			facial_features.save()
			#td.test_image(trump,V[0:5],immean)
		