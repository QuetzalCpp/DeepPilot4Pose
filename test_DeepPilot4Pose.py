#!/usr/bin/env python3
__author__ = "L. Oyuki Rojas-Perez and Dr. Jose Martinez-Carranza"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "L. Oyuki Rojas-Perez"
__email__ = "carranza@inaoep.mx"

import network_libs.helper_DeepPilot4Pose as helper_net
import network_libs.DeepPilot4Pose_net as net
from numpy import *
import numpy as np
from tensorflow.keras.optimizers import Adam
import math
import time
from scipy.spatial import distance

if __name__ == "__main__":
	model = net.create_DeepPilot4Pose()

	print(" ")
	print("Load model")
	
	model_name = "models/Test_Warehouse_model10_100ep.h5"
	
	model.load_weights(model_name)
	
	dataset_eval = helper_net.getEvalSource()

	x_eval = np.squeeze(np.array(dataset_eval.images))
	y_eval = np.squeeze(np.array(dataset_eval.poses))

	y_eval_pose = y_eval[:,0:3]
	y_eval_expmap = y_eval[:,3:4]

	star = time.time()
	eval_est = model.predict(x_eval)
	elapse = time.time() - star

	# Get results... :/

	N = len(dataset_eval.images)
	
	dTw = []
	dEM = []

	for i in range(N):
		
		Px = eval_est[0][i][0]
		Py = eval_est[1][i][0]
		Pz = eval_est[2][i][0]
		
		Tg = y_eval_pose[i]
		Te = np.array([Px,Py,Pz])
		
		EMg = y_eval_expmap[i][0]
		EMe = eval_est[3][i][0]

		# Traslation
		dist = distance.euclidean(Tg, Te)
		
		diffEM = np.absolute(EMg - EMe)
		
		dTw.append(dist)
		dEM.append(diffEM)
	
	dTwMean = np.mean(dTw)
	dTwStd = np.std(dTw)
	
	print( " ")
	print( "Model Evaluated: ", model_name)
	print( " ")
	print( "Total Imgs: ",N)
	print( "-----------------")
	print( "Tw Mean (m): ",dTwMean)
	print( "Tw Std (m): ",dTwStd)
	print( "-----------------")

	dEMMean = np.mean(dEM, axis=0)
	dEMStd = np.std(dEM, axis=0)
	print("EMz Mean (deg): ",np.transpose(dEMMean)*180/np.pi)
	print("EMz Std (deg): ",np.transpose(dEMStd)*180/np.pi)
	print("-----------------")

	print("time: ",elapse)
	print("Freq (Hz): ",1/(elapse/N))
