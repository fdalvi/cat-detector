import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from cifar_processor import load_cifar_data
from voc_processor import *

import numpy as np
import sys
import os
import cPickle as cp

def evaluate_accuracy(model, X, y):
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

model_save_path = ""
if len(sys.argv) > 1:
	model_save_path = sys.argv[1]
	if not os.path.exists(model_save_path):
		os.mkdir(model_save_path)

batch_size = 128
epochs = 15
total_mining_iterations = 3
bbox_reg_epochs = 60
num_classes = 2
# x_train, y_train, x_val, y_val, x_test, y_test = load_cifar_data(single_class=3, balance_classes=True)
x_train, y_train, x_val, y_val, x_test, y_test, \
x_train_bbox, y_train_bbox, x_val_bbox, y_val_bbox, \
	x_test_bbox, y_test_bbox, x_extra_negatives = load_voc_data_cached()
x_train_positive = x_train_bbox
# Create model
shared_model = Sequential()

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

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
			  optimizer=opt,
			  metrics=['accuracy'])

# Create fork at final fc layer for bounding box regression
for l in shared_model.layers:
	l.trainable = False

bbox_reg_model = Sequential()
bbox_reg_model.add(shared_model)
bbox_reg_model.add(Dense(512, name='bbox_fc5'))
bbox_reg_model.add(Activation('relu', name='bbox_fc5_act'))
bbox_reg_model.add(Dropout(0.5, name='bbox_fc5_drop'))
bbox_reg_model.add(Dense(128, name='bbox_fc6'))
bbox_reg_model.add(Activation('linear', name='bbox_fc6_act'))
bbox_reg_model.add(Dense(4, name='bbox_out'))
bbox_reg_model.compile(loss='mean_squared_error', optimizer='adam')

train_history = []
for mine_idx in xrange(total_mining_iterations):
	for epoch in xrange(epochs):
		print "Epoch",(mine_idx*epochs + epoch+1)
		historyLogger = LossHistory()
		history = model.fit(x_train, y_train,
				  batch_size=batch_size,
				  epochs=1,
				  validation_data=(x_val, y_val),
				  shuffle=True, callbacks=[historyLogger])
		train_history.append(historyLogger.losses)
		model_name = "model_%d_%0.2f.h5"%(mine_idx*epochs + epoch+1, history.history['val_acc'][0]*100)
		model_name = os.path.join(model_save_path, model_name)
		model.save(model_name)

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

	x_train = np.concatenate((x_train_bbox, hard_negatives), 0)
	y_train = np.zeros((x_train.shape[0], 2),dtype=np.int)
	y_train[0:x_train_bbox.shape[0], 1] = 1
	y_train[x_train_bbox.shape[0]:, 0] = 1

	idx = np.random.permutation(x_train.shape[0])
	x_train = x_train[idx]
	y_train = y_train[idx]
	# print y_train

for epoch in xrange(bbox_reg_epochs):
	print "Epoch",(epoch+1)
	history = bbox_reg_model.fit(x_train_bbox, y_train_bbox,
			  batch_size=batch_size,
			  epochs=1,
			  validation_data=(x_val_bbox, y_val_bbox),
			  shuffle=True)

	model_name = "bbox_model_%d_%0.4f.h5"%(epoch+1, history.history['loss'][0])
	model_name = os.path.join(model_save_path, model_name)
	model.save(model_name)

	y_pred = bbox_reg_model.predict(x_test_bbox[:10])

	print "Samples..."
	for i in xrange(10):
		print y_pred[i],y_test_bbox[i]

# evaluate the model
y_pred = model.predict(x_test, verbose=0)
acc = 100.0 * np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))/y_test.shape[0]

positive_examples = np.where(y_test[:, 1]==1)[0]
positive_acc = 100.0 * np.sum(np.argmax(y_pred[positive_examples], axis=1) == np.argmax(y_test[positive_examples], axis=1))/y_test[positive_examples].shape[0]

negative_examples = np.where(y_test[:, 0]==1)[0]
negative_acc = 100.0 * np.sum(np.argmax(y_pred[negative_examples], axis=1) == np.argmax(y_test[negative_examples], axis=1))/y_test[negative_examples].shape[0]
print("%s: %.2f%%" % ("Accuracy", acc))
print("%s: %.2f%%" % ("Cat Accuracy", positive_acc))
print("%s: %.2f%%" % ("Non Cat Accuracy", negative_acc)) 

# processed_file = "processed_images.npz"
# all_samples = load_processed_samples(processed_file)
# x_positive = all_samples[get_class_idx('cat')]

# y_pred = model.predict(x_positive, batch_size=64, verbose=True)
# # print y_pred
# correct = 0
# for i in xrange(y_pred.shape[0]):
# 	# print y_pred[i]
# 	if y_pred[i,1] > 0.5:
# 		correct += 1
# print("%s: %0.2f"%("Cat Accuracy", 100.0*correct/y_pred.shape[0]))


# Save model
model.save(os.path.join(model_save_path, "model_final.h5"))
with open(os.path.join(model_save_path, "history.pkl"), 'w') as fp:
	cp.dump(train_history, fp)
bbox_reg_model.save(os.path.join(model_save_path, "bbox_model_final.h5"))
print("Saved model to disk")
