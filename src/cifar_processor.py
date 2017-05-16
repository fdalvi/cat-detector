## CIFAR-10 processor
# Loads CIFAR-10 data and returns the data as [H*W*D]
#
# Author: Fahim Dalvi

import cPickle
import numpy as np

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

def transform_to_3d(flattened_array):
	num_examples = flattened_array.shape[0]
	return np.swapaxes(flattened_array.reshape((num_examples,32,32,3), order='F'),1,2)

def transform_to_2d(label_array, num_classes=10):
	num_examples = label_array.shape[0]
	result = np.zeros((num_examples, num_classes))
	result[np.arange(num_examples),label_array] = 1
	return result

def balance_set(x_train, y_train):
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

## Loads CIFAR-data from disk
# If single_class=-1, then all 10 classes are considered. If not,
# examples are divided into two classes, one of the `single_class`
# and one of all other classes. set balance_classes to True if
# you are only considering one class and want to randomly sample
# from the other classes and make both classes equal in the number
# of examples
def load_cifar_data(single_class=-1, balance_classes=False):
	data_batches = [unpickle('../data/cifar10/cifar-10-batches-py/data_batch_' + str(i)) for i in xrange(1,6)]
	test_batch = unpickle('../data/cifar10/cifar-10-batches-py/test_batch')

	# Set Number of classes to 2 if we are only interested in a single class, like cat
	num_classes = 10
	if single_class != -1:
		num_classes = 2

	x_train = np.concatenate(tuple([data_batches[i]['data'] for i in xrange(4)]), 0)
	y_train = np.concatenate(tuple([data_batches[i]['labels'] for i in xrange(4)]), 0)
	if single_class != -1:
		y_train[y_train!=single_class] = 0
		y_train[y_train==single_class] = 1
		if balance_classes:
			x_train, y_train = balance_set(x_train, y_train)


	x_val = data_batches[4]['data']
	y_val = np.array(data_batches[4]['labels'])
	if single_class != -1:
		y_val[y_val!=single_class] = 0
		y_val[y_val==single_class] = 1
		if balance_classes:
			x_val, y_val = balance_set(x_val, y_val)


	x_test = test_batch['data']
	y_test = np.array(test_batch['labels'])
	if single_class != -1:
		y_test[y_test!=single_class] = 0
		y_test[y_test==single_class] = 1
		if balance_classes:
			x_test, y_test = balance_set(x_test, y_test)

	x_train = transform_to_3d(x_train)
	x_val = transform_to_3d(x_val)
	x_test = transform_to_3d(x_test)

	y_train = transform_to_2d(y_train, num_classes=num_classes)
	y_val = transform_to_2d(y_val, num_classes=num_classes)
	y_test = transform_to_2d(y_test, num_classes=num_classes)

	x_train = x_train.astype('float32')
	x_val = x_val.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_val /= 255
	x_test /= 255

	return x_train, y_train, x_val, y_val, x_test, y_test

def visualize_image(image, image_name="image.png"):
	from PIL import Image
	im = Image.fromarray((image*255).astype('int8'), mode='RGB')
	im.save(image_name)

def main():
	x_train, _, _, _, _, _ = load_cifar_data()
	visualize_image(x_train[0], image_name="1.png")
	visualize_image(x_train[1], image_name="2.png")

if __name__ == '__main__':
	main()