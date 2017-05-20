## Train
# Main script to train cat detector in the wild
#
# Author: Fahim Dalvi

import cPickle as cp
import keras
import numpy as np
import os
import sys

from data_utils import *
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential

def evaluate_accuracy(model, X, y):
	"""
	Given a model, and a test set, evaluate overall accuracy and
	per-class accuracy.
	
	Parameters
	----------
	model : Keras Model
	Trained model to be used for inference
	X : numpy tensor [N x 128 x 128 x 3]
	Samples to perform inference on.
	y : numpy vector [N x 1]
	Vector with labels for samples.

	Returns
	-------
	None
	"""
	y_pred = model.predict(X, verbose=0)
	acc = 100.0 * np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))/y.shape[0]

	positive_examples = np.where(y[:, 1]==1)[0]
	positive_acc = 100.0 * np.sum(np.argmax(y_pred[positive_examples], axis=1) == np.argmax(y[positive_examples], axis=1))/y[positive_examples].shape[0]

	negative_examples = np.where(y[:, 0]==1)[0]
	negative_acc = 100.0 * np.sum(np.argmax(y_pred[negative_examples], axis=1) == np.argmax(y[negative_examples], axis=1))/y[negative_examples].shape[0]
	print("%s: %.2f%%" % ("Accuracy", acc))
	print("%s: %.2f%%" % ("Cat Accuracy", positive_acc))
	print("%s: %.2f%%" % ("Non Cat Accuracy", negative_acc))

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

def main():
	# Check if model saving path is specified and create
	# relevant directories
	model_save_path = ""
	if len(sys.argv) > 1:
		model_save_path = sys.argv[1]
		if not os.path.exists(model_save_path):
			os.mkdir(model_save_path)

	# Network parameters
	batch_size = 128
	epochs = 1
	total_mining_iterations = 0
	bbox_reg_epochs = 1
	num_classes = 2

	# Load dataset (from cache if possible, or else process it first)
	x_train, y_train, x_val, y_val, x_test, y_test, \
	x_train_bbox, y_train_bbox, x_val_bbox, y_val_bbox, \
	x_test_bbox, y_test_bbox, x_extra_negatives = load_voc_data_cached()
	x_train_positive = x_train_bbox

	# Create model
	shared_model = Sequential()

	# Shared model layers
	shared_model.add(Conv2D(32, (3, 3), padding='same',
					 input_shape=x_train.shape[1:], name='conv1'))
	shared_model.add(Activation('relu', name='conv1_act'))
	shared_model.add(Conv2D(32, (3, 3), name='conv2'))
	shared_model.add(Activation('relu', name='conv2_act'))
	shared_model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
	shared_model.add(Dropout(0.25, name='pool1_drop'))

	shared_model.add(Conv2D(64, (3, 3), padding='same', name='conv3'))
	shared_model.add(Activation('relu', name='conv3_act'))
	shared_model.add(Conv2D(64, (3, 3), name='conv4'))
	shared_model.add(Activation('relu', name='conv4_act'))
	shared_model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
	shared_model.add(Dropout(0.25, name='pool2_drop'))

	shared_model.add(Flatten(name='flat1'))

	# Create normal model for classification
	model = Sequential()
	model.add(shared_model)
	model.add(Dense(512, name='fc5'))
	model.add(Activation('relu', name='fc5_act'))
	model.add(Dropout(0.5, name='fc5_drop'))
	model.add(Dense(512, name='fc6'))
	model.add(Activation('relu', name='fc6_act'))
	model.add(Dropout(0.5, name='fc6_drop'))
	model.add(Dense(num_classes, name='out'))
	model.add(Activation('softmax', name='out_act'))

	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Compile classifier
	model.compile(loss='categorical_crossentropy',
				  optimizer=opt,
				  metrics=['accuracy'])

	# Create fork at final fc layer for bounding box regression
	for l in shared_model.layers:
		l.trainable = False

	# Create model for regression
	bbox_reg_model = Sequential()
	bbox_reg_model.add(shared_model)
	bbox_reg_model.add(Dense(512, name='bbox_fc5'))
	bbox_reg_model.add(Activation('relu', name='bbox_fc5_act'))
	bbox_reg_model.add(Dropout(0.5, name='bbox_fc5_drop'))
	bbox_reg_model.add(Dense(128, name='bbox_fc6'))
	bbox_reg_model.add(Activation('linear', name='bbox_fc6_act'))
	bbox_reg_model.add(Dense(4, name='bbox_out'))
	bbox_reg_model.compile(loss='mean_squared_error', optimizer='adam')

	# Start training
	train_history = []

	# Train classifier
	for mine_idx in xrange(total_mining_iterations):
		# For each mining period, train for `epochs` epochs.
		# First period is based on balanced classes
		for epoch in xrange(epochs):
			print "Epoch",(mine_idx*epochs + epoch+1)
			# Perform 1 epoch of training
			historyLogger = LossHistory()
			history = model.fit(x_train, y_train,
					  batch_size=batch_size,
					  epochs=1,
					  validation_data=(x_val, y_val),
					  shuffle=True, callbacks=[historyLogger])
			train_history.append(historyLogger.losses)

			# Save current model
			model_name = "model_%d_%0.2f.h5"%(mine_idx*epochs + epoch+1, history.history['val_acc'][0]*100)
			model_name = os.path.join(model_save_path, model_name)
			model.save(model_name)

		# Evaluate model after period
		evaluate_accuracy(model, x_val, y_val)

		# Hard negative mining
		num_hard_negatives = x_train_bbox.shape[0] * 2
		potential_hard_negatives_idx = np.random.permutation(x_extra_negatives.shape[0])
		potential_hard_negatives = x_extra_negatives[potential_hard_negatives_idx[:num_hard_negatives]]
		y_pred = model.predict(potential_hard_negatives)
		y_pred = np.argmax(y_pred, axis=1)
		hard_negatives_idx = np.where(y_pred==1)[0]
		print "%d/%d Hard negatives found."%(hard_negatives_idx.shape[0], potential_hard_negatives.shape[0])
		remaining_negatives = num_hard_negatives - hard_negatives_idx.shape[0]
		remaining_negatives_idx = np.arange(remaining_negatives)+num_hard_negatives
		hard_negatives = potential_hard_negatives[hard_negatives_idx]
		hard_negatives = np.concatenate((hard_negatives, x_extra_negatives[remaining_negatives_idx]), 0)
		print "%d/%d Hard negatives found."%(hard_negatives.shape[0], potential_hard_negatives.shape[0])
		if hard_negatives.shape[0] == 0:
			hard_negatives = None

		# Create new train set
		x_train = np.concatenate((x_train_bbox, hard_negatives), 0)
		y_train = np.zeros((x_train.shape[0], 2),dtype=np.int)
		y_train[0:x_train_bbox.shape[0], 1] = 1
		y_train[x_train_bbox.shape[0]:, 0] = 1

		idx = np.random.permutation(x_train.shape[0])
		x_train = x_train[idx]
		y_train = y_train[idx]

	# Train regressor
	for epoch in xrange(bbox_reg_epochs):
		print "Epoch",(epoch+1)
		history = bbox_reg_model.fit(x_train_bbox, y_train_bbox,
				  batch_size=batch_size,
				  epochs=1,
				  validation_data=(x_val_bbox, y_val_bbox),
				  shuffle=True)

		# Save current model
		model_name = "bbox_model_%d_%0.4f.h5"%(epoch+1, history.history['loss'][0])
		model_name = os.path.join(model_save_path, model_name)
		model.save(model_name)

		y_pred = bbox_reg_model.predict(x_test_bbox[:10])

		print "Samples..."
		for i in xrange(10):
			print y_pred[i],y_test_bbox[i]

	# Evaluate the model on the test set
	evaluate_accuracy(model, x_test, y_test)

	# Save final models and history
	model.save(os.path.join(model_save_path, "model_final.h5"))
	with open(os.path.join(model_save_path, "history.pkl"), 'w') as fp:
		cp.dump(train_history, fp)
	bbox_reg_model.save(os.path.join(model_save_path, "bbox_model_final.h5"))
	print("Saved model to disk")

if __name__ == '__main__':
	main()
