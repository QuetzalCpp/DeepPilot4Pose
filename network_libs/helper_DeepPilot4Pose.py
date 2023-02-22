#!/usr/bin/env python3
__author__ = "L. Oyuki Rojas-Perez and Dr. Jose Martinez-Carranza"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "L. Oyuki Rojas-Perez"
__email__ = "carranza@inaoep.mx"

from tqdm import tqdm
import numpy as np
import os.path
import sys
import random
import math
import cv2
import gc

#warehouse_Lshape
directory = '/home/rafaga22630/Documents/INAOE/Github/DeepPilot4Pose/dataset/Warehouse_PoseNet/data_Lshape_220310/'
dataset_train = 'Warehouse_train_Lshape_Data_17-20_final.txt'
dataset_eval = 'Warehouse_eval_data_Lshape.txt'

# ~ # 7 Scenes - Redkitchen
# ~ directory = '/home/rafaga22630/datasets/RGB-D_Dataset_7-Scenes_Microsoft/redkitchen/'
# ~ dataset_train = 'train-data.txt'
# ~ dataset_train = 'train-datax4.txt'
# ~ dataset_eval = 'kitchen_eval.txt'

# ~ # 7 Scenes - Heads
# ~ directory = '/home/rafaga22630/datasets/RGB-D_Dataset_7-Scenes_Microsoft/heads/'
# ~ dataset_train = 'train-data.txt'
# ~ dataset_train = 'train-data_x4.txt'
# ~ dataset_eval = 'evalData.txt'

class datasource(object):
	def __init__(self, images, poses):
		self.images = images
		self.poses = poses

def preprocess(images):
	images_out = [] #final result
	#Resize input images
	for i in tqdm(range(len(images))):
		# ~ print( " ")
		# ~ print( "going to read i ",i," fn: ",images[i])
		X = cv2.imread(images[i])
		# ~ print( i," was read, now: img size: ",X.shape)
		X = cv2.resize(X, (224, 224))
		X = np.transpose(X,(2,0,1))
		X = np.squeeze(X)
		X = np.transpose(X, (1,2,0))
		Y = np.expand_dims(X, axis=0)
		images_out.append(Y)
	del X, i
	gc.collect()
	return images_out

def get_data(dataset):
	poses = []
	images = []

	with open(directory+dataset) as f:
		# skip the 3 header lines
		next(f)  
		next(f)
		next(f)
		for line in f:
			fname, p0,p1,p2,_,_,p3,_,_,_,_ = line.split()
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = float(p3)
			poses.append((p0,p1,p2,p3))
			images.append(directory+fname)
	images_out = preprocess(images)
	return datasource(images_out, poses)

def getEvalSource():
	datasource_test = get_data(dataset_eval)
	
	images_test = []
	poses_test = []

	for i in range(len(datasource_test.images)):
		images_test.append(datasource_test.images[i])
		poses_test.append(datasource_test.poses[i])

	return datasource(images_test, poses_test)	

### #img_name PoseX PoseY PoseZ EMx EMy EMz Qw Qx Qy Qz roll pitch yaw altitude
def get_TrainSources():
	poses = []
	images = []
	
	train_img = []
	train_poses = []
	
	valid_img = []
	valid_poses = []

	with open(directory+dataset_train) as f:
		# skip the 3 header lines
		next(f)  
		next(f)
		next(f)
		for line in f:
			fname, p0,p1,p2,_,_,p3,_,_,_,_ = line.split()
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = float(p3)
			train_poses.append((p0,p1,p2,p3))
			train_img.append(directory+fname)
	# ~ print ("" )
	# ~ print (len(train_img), " Train images" )
	# ~ print (train_img)
	# ~ print (train_poses)
	
	for i in range(0,len(train_img)):
		# ~ if i%4==0:
		if i%5==0:
			valid_img.append(train_img[i])
			valid_poses.append(train_poses[i])
	
	for i in valid_img:
		train_img.remove(i)
	
	for i in valid_poses:
		train_poses.remove(i)
	
	# ~ print( "" )
	# ~ print( len(valid_img), " subimages" )
	# ~ print( valid_img)
	# ~ print( valid_poses)
	# ~ print( "" )
	# ~ print( len(train_img), " images" )
	# ~ print( train_img)
	# ~ print( train_poses)
	
	train_img_out = preprocess(train_img)
	valid_img_out = preprocess(valid_img)

	return datasource(train_img_out, train_poses), datasource(valid_img_out, valid_poses)
