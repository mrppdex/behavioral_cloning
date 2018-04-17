import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Conv2D, Lambda, Dense, Input, Flatten, Dropout
from keras import backend as K
from keras.optimizers import SGD, Adam
import tensorflow as tf
import csv
from random import shuffle
import sklearn
from sklearn.model_selection import train_test_split
import os

# locations of the training data
log_file = "../data/driving_log.csv"
img_dir = "../data/IMG"

# input 320x160 RGB (BGR in training)
input_shape = (160, 320, 3) 

# some of the hyperparameters
batch_size = 64
nb_epochs = 5 

def read_log_file(log_file):
	''' 
	Opens log_file containing file names and driving data.
	
	Input:
		log_file - csv file containing file names for camera images and associated data.
	Returns: 
		img_files - file names with image data
		measurements - steering angles associated with img_files
	'''
	with open(log_file, 'r') as f:
		reader = csv.reader(f)
		img_files = []
		measurements = []
		# uncomment next line if the csv file contains a header
		#next(reader, None)  		
		for line in reader:
			# center camera image
			c_img_f = line[0].split('/')[-1]
			# left camera image
			l_img_f = line[1].split('/')[-1]
			# right camera image
			r_img_f = line[2].split('/')[-1]
			
			img_files.append(os.path.join(img_dir, c_img_f))
			img_files.append(os.path.join(img_dir, l_img_f))
			img_files.append(os.path.join(img_dir, r_img_f))
			
			# perspective correction for the side cameras (2.5 degrees)
			perspective_correction_angle = 2.5
			
			center_angle = float(line[3])
			left_angle = center_angle + perspective_correction_angle
			right_angle = center_angle - perspective_correction_angle
			measurements.extend([center_angle, left_angle, right_angle])
		return img_files, measurements
		
def img_generator(img_files, measurements, batch_size):
	'''
	Transforms input images to and generates batches of data.

	Input:
		img_files - file names (string) with camera images
		measurements - steering wheel angles associated with img_files
		batch_size - batch size

	Output:
		X_train - training data
		y_train - labels for training data
	'''
	# random idices used to shuffle the trainig data and labels
	rand_idx = list(range(len(img_files)))
	shuffle(rand_idx)

	# used to output sample image from the training set
	once_off = False
	
	# data generator
	while 1:
		for batch in range(0, len(rand_idx), batch_size):
			X_train = []
			y_train = []
		
			# fetch a batch of random indices and use it to open image data from the function argument
			for nr in rand_idx[batch: batch+batch_size]:
				# open trainig image
				image = cv2.imread(img_files[nr])
				# converts OpenCV's BGR format to RGB format
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				# get label
				label = measurements[nr]
				# augment the data. flip every 5th image.
				if np.random.rand() < 0.2:
					image = cv2.flip(image, 1)
					label = -label
				# debug. save an example of the camera data.
				if not once_off:
					cv2.imwrite('screenshot.png', image)
					once_off = True
				# add data/label to the data sets
				X_train.append(image)
				y_train.append(label)
			# shuffle and yield trainig data/label
			yield sklearn.utils.shuffle(np.asarray(X_train), np.asarray(y_train))
			
def covnet_model(input_shape):
	'''
	Creates covnet model based on NVIDIA behavioral cloning paper.
	
	Inputs:
		input_shape - the shape of an input image
	Outputs:
		model
	'''
	
	# resizes input image to 200x66x3 (by NVIDIA)
	# uses tenforflow backend's resize_images function
	def resize_normalize(img):
		import cv2
		from keras.backend import tf as ktf
		image = ktf.image.resize_images(img, (66, 200))
		return image


	inputs = Input(input_shape)
	
	net = Cropping2D(((50, 20), (0, 0)), input_shape=(160, 320, 3))(inputsa)

	# the following line prevents the model from being saved. 
	# Using the resize_normalize function solves the problem
	#net = Lambda(lambda x: tf.image.resize_images(net, (66, 200)))(inputs)

	# normalize the input data
	net = Lambda(lambda x: (x/255.) - 0.5)(inputs)
	# resize from 320x160x3 to 200x66x3
	net = Lambda(resize_normalize, input_shape=(160, 320, 3), output_shape=(66, 200, 3))(inputs)
	net = Conv2D(24, 5, 5, subsample=(2, 2), activation='relu')(net)
	net = Conv2D(36, 5, 5, subsample=(2, 2), activation='relu')(net)
	net = Conv2D(48, 5, 5, subsample=(2, 2), activation='relu')(net)
	# dropout can be used although model works well without it
	#net = Dropout(0.5)(net)
	net = Conv2D(64, 3, 3, subsample=(1, 1), activation='relu')(net)
	net = Conv2D(64, 3, 3, subsample=(1, 1), activation='relu')(net)
	
	net = Flatten()(net)
	net = Dense(1164, activation='relu')(net)
	#net = Dropout(0.4)(net)
	net = Dense(100, activation='relu')(net)
	#net = Dropout(0.2)(net)
	net = Dense(50, activation='relu')(net)
	net = Dense(10, activation='relu')(net)
	
	# an output is a scalar with the steering wheel angle.
	logits = Dense(1)(net)
	
	# return the model
	return Model(input=inputs, output=logits)
	

if __name__ == '__main__':
	# read training data and labels from the log_file
	files, measurements = read_log_file(log_file)

	# split training data into trainig and validation subsets
	X_train, X_valid, y_train, y_valid = train_test_split(files, measurements)
	
	# initialize generators
	train_gen = img_generator(X_train, y_train, batch_size=batch_size)
	valid_gen = img_generator(X_valid, y_valid, batch_size=batch_size)
	
	# create the model
	model=covnet_model(input_shape)

	# use Adam optimizer with learning rate = 0.001
	optimizer = Adam(lr=0.001)

	# model's loss is Mean Squared Error, as it is a regression problem
	model.compile(loss='mse', optimizer=optimizer)

	# train the model
	model.fit_generator(train_gen, samples_per_epoch = len(X_train), \
	validation_data=valid_gen, nb_val_samples=len(X_valid), nb_epoch=nb_epochs, \
	verbose=1)
	
	# save the model
	model.save('model.h5')

