#!/usr/bin/env python3
__author__ = "L. Oyuki Rojas-Perez and Dr. Jose Martinez-Carranza"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "L. Oyuki Rojas-Perez"
__email__ = "carranza@inaoep.mx"

import network_libs.helper_DeepPilot4Pose as helper_net
import network_libs.DeepPilot4Pose_net as net
import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from keras import backend as K 

if __name__ == "__main__":
	K.clear_session()
	# Variables
	#change batch_size in function of your VRAM capacity, or for the sample size you want
	batch_size = 60

	# Model creation for dedicated branch
	model = net.create_DeepPilot4Pose('posenet.npy', True)

	adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=1.5)

	#Same compilation for both nets, dense layers have sema name.
	model.compile(optimizer=adam, loss={'pose_x': net.euc_lossx,
										'pose_y': net.euc_lossy,
										'pose_z': net.euc_lossz,
										'EMz': net.euc_lossEMz})



	dataset_train, dataset_test = helper_net.get_TrainSources()

	X_train = np.squeeze(np.array(dataset_train.images))
	y_train = np.squeeze(np.array(dataset_train.poses))

	y_train_x = y_train[:,0:1]
	y_train_y = y_train[:,1:2]
	y_train_z = y_train[:,2:3]
	y_train_emz = y_train[:,3:4]

	X_test = np.squeeze(np.array(dataset_test.images))
	y_test = np.squeeze(np.array(dataset_test.poses))

	y_test_x = y_test[:,0:1]
	y_test_y = y_test[:,1:2]
	y_test_z = y_test[:,2:3]
	y_test_emz = y_test[:,3:4]

	# Setup checkpointing
	print(" " )
	modelName_Out= "models/Test_Warehouse_model10_10ep.h5"

	print("model Name Out: ", modelName_Out)
	print(" " )
	#file_name.h5 -> the file where to save data
	#save_best_only -> not to save all the weights just the ones who improves
	#save_weights_only -> if true save the weights, if false save the entire model +  weights
	checkpointer = ModelCheckpoint(filepath=modelName_Out, verbose=3, save_best_only=True, save_weights_only=True)

	#In this part you specify the labels for training learning and for validation
	history = model.fit(X_train, [y_train_x, y_train_y, y_train_z, y_train_emz],
		  batch_size=batch_size,
		  epochs=300,
		  validation_data=(X_test, [y_test_x, y_test_y, y_test_z, y_test_emz]),
		  callbacks=[checkpointer])

	#For plot the training and validation losses 
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(loss) + 1)
	plt.plot(epochs, loss, color='red', label='Training loss')
	plt.plot(epochs, val_loss, color='green', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.grid(True)
	plt.show()
