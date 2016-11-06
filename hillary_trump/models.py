from django.db import models
from django.conf import settings
#from __future__ import unicode_literals

# Create your models here.
class Facial_Vectors(models.Model):
	name                = models.CharField(max_length=200)
	vector				= models.TextField(null=True)
	matrix              = models.TextField(null=True)
	mean                = models.TextField(null=True)
	date                = models.DateTimeField(auto_now_add=True)

class ImageAttachment(models.Model):
	file=models.ImageField(upload_to=settings.IMAGE_UPLOAD_PATH,blank=True, null=True)
	path=models.CharField(max_length=200000,null=True)
	name=models.CharField(max_length=200000,null=True)