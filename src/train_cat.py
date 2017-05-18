import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from cifar_processor import load_cifar_data
from voc_processor import *

import numpy as np

batch_size = 128
epochs = 30
num_classes = 2
# x_train, y_train, x_val, y_val, x_test, y_test = load_cifar_data(single_class=3, balance_classes=True)
x_train, y_train, x_val, y_val, x_test, y_test, \
x_train_bbox, y_train_bbox, x_val_bbox, y_val_bbox, \
			x_test_bbox, y_test_bbox = load_voc_data()

# Create model
shared_model = Sequential()

shared_model.add(Conv2D(32, (3, 3), padding='same',
				 input_shape=x_train.shape[1:]))
shared_model.add(Activation('relu'))
shared_model.add(Conv2D(32, (3, 3)))
shared_model.add(Activation('relu'))
shared_model.add(MaxPooling2D(pool_size=(2, 2)))
shared_model.add(Dropout(0.25))

shared_model.add(Conv2D(64, (3, 3), padding='same'))
shared_model.add(Activation('relu'))
shared_model.add(Conv2D(64, (3, 3)))
shared_model.add(Activation('relu'))
shared_model.add(MaxPooling2D(pool_size=(2, 2)))
shared_model.add(Dropout(0.25))

shared_model.add(Flatten())
shared_model.add(Dense(512))
shared_model.add(Activation('relu'))
shared_model.add(Dropout(0.5))
shared_model.add(Dense(512))
shared_model.add(Activation('relu'))
shared_model.add(Dropout(0.5))

# Create normal model for classification
model = Sequential()
model.add(shared_model)
model.add(Dense(num_classes))
model.add(Activation('softmax'))

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
bbox_reg_model.add(Dense(128))
bbox_reg_model.add(Activation('relu'))
bbox_reg_model.add(Dense(4))
bbox_reg_model.compile(loss='mean_squared_error', optimizer='adam')

for epoch in xrange(epochs):
	history = model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=1,
			  validation_data=(x_val, y_val),
			  shuffle=True)
	model_name = "model_%d_%0.2f.h5"%(epoch+1, history.history['val_acc'][0]*100)
	model.save(model_name)

bbox_reg_model.fit(x_train_bbox, y_train_bbox,
			  batch_size=batch_size,
			  epochs=30,
			  validation_data=(x_val_bbox, y_val_bbox),
			  shuffle=True)

# evaluate the model
y_pred = model.predict(x_test, verbose=0)
acc = 100.0 * np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))/y_test.shape[0]

positive_examples = y_test[:, 1].astype(np.int)
positive_acc = 100.0 * np.sum(np.argmax(y_pred[positive_examples,:], axis=1) == np.argmax(y_test[positive_examples,:], axis=1))/y_test[positive_examples,:].shape[0]
print("%s: %.2f%%" % ("Accuracy", acc))
print("%s: %.2f%%" % ("Cat Accuracy", positive_acc)) 

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
model.save("model_final.h5")
bbox_reg_model.save("bbox_model_final.h5")
print("Saved model to disk")
