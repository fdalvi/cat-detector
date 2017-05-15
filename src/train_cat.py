import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import numpy as np

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def transform_to_3d(flattened_array):
	num_examples = flattened_array.shape[0]
	return np.swapaxes(flattened_array.reshape((num_examples,32,32,3), order='F'),1,2)

def transform_to_2d(label_array):
	num_examples = label_array.shape[0]
	result = np.zeros((num_examples, 2))
	result[np.arange(num_examples),label_array] = 1
	return result

def balance_classes(x_train, y_train):
  num_examples = np.sum(y_train)
  negative_examples = np.where(y_train == 0)[0]
  np.random.shuffle(negative_examples)
  negative_examples = negative_examples[0:num_examples]
  positive_examples = np.where(y_train == 1)[0]
  
  new_x_train_negative = x_train[negative_examples,:]
  new_x_train_positive = x_train[positive_examples,:]
  new_x_train = np.concatenate((new_x_train_negative,new_x_train_positive), 0)
  new_y_train = np.concatenate((np.zeros((num_examples,), dtype=np.int), np.ones((num_examples,), dtype=np.int)),0)

  idx = np.random.permutation(num_examples*2)
  return new_x_train[idx], new_y_train[idx]

data_batches = [unpickle('cifar-10-batches-py/data_batch_' + str(i)) for i in xrange(1,6)]
test_batch = unpickle('cifar-10-batches-py/test_batch')

num_classes = 2
batch_size = 32
epochs = 10

x_train = np.concatenate(tuple([data_batches[i]['data'] for i in xrange(4)]), 0)
y_train = np.concatenate(tuple([data_batches[i]['labels'] for i in xrange(4)]), 0)
y_train[y_train!=3] = 0
y_train[y_train==3] = 1
x_train, y_train = balance_classes(x_train, y_train)

x_val = data_batches[4]['data']
y_val = np.array(data_batches[4]['labels'])
y_val[y_val!=3] = 0
y_val[y_val==3] = 1
x_val, y_val = balance_classes(x_val, y_val)


x_test = test_batch['data']
y_test = np.array(test_batch['labels'])
y_test[y_test!=3] = 0
y_test[y_test==3] = 1
x_test, y_test = balance_classes(x_test, y_test)

x_train = transform_to_3d(x_train)
x_val = transform_to_3d(x_val)
x_test = transform_to_3d(x_test)

y_train = transform_to_2d(y_train)
y_val = transform_to_2d(y_val)
y_test = transform_to_2d(y_test)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_val /= 255
x_test /= 255

# print x_test[0,0:10]
# print x_test[0,1024:1034]
# print x_test[0,2048:2058]

# print(x_test.shape)
# print(x_test[0][0][0][1])
# print(x_test[0,:,:,0])
# print(x_test[0,:,:,1])
# print(x_test[0,:,:,2])

# Create model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val),
              shuffle=True)

# evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
