## VOC2012 processor
# Loads VOC2012 data and returns the data as [H*W*D]
#
# Author: Fahim Dalvi

import numpy as np
import os
import xml.etree.ElementTree

from PIL import Image

DATA_PATH = '../data/voc2012/VOC2012/'
classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','tvmonitor']

def get_class_idx(class_name):
	return classes.index(class_name)

def get_class_name(class_idx):
	return classes[class_idx]

def transform_to_2d(label_array, num_classes=2):
	num_examples = label_array.shape[0]
	result = np.zeros((num_examples, num_classes))
	result[np.arange(num_examples),label_array] = 1
	return result

def get_class_image_names(class_name, include_difficult=False):
	class_images = []
	with open(DATA_PATH + 'ImageSets/Main/' + class_name + '_trainval.txt') as fp:
		for line in fp:
			image_name, label = line.strip().split()
			label = int(label)
			if label == 1 or (include_difficult and label == 0):
				class_images.append(image_name)
	return class_images

def get_bounding_boxes(image_name, class_name):
	bounding_boxes = []
	e = xml.etree.ElementTree.parse(DATA_PATH + 'Annotations/' + image_name + '.xml').getroot()
	for obj in e.findall("object"):
		current_obj = obj.find("name").text
		if current_obj == class_name:
			box = [0,0,0,0]
			for coord in obj.find("bndbox"):
				if coord.tag == 'xmin': box[0] = int(coord.text)
				if coord.tag == 'ymin': box[1] = int(coord.text)
				if coord.tag == 'xmax': box[2] = int(coord.text)
				if coord.tag == 'ymax': box[3] = int(coord.text)
			bounding_boxes.append(box)
	return bounding_boxes

def extract_image(image_name, bounding_box):
	im = Image.open(DATA_PATH + 'JPEGImages/' + image_name + ".jpg")
	cropped_im = im.crop(bounding_box)
	cropped_im = cropped_im.resize((64,64), resample=Image.BILINEAR)
	return np.array(cropped_im)

def process_voc_data(save_file):
	image_sets = [get_class_image_names(c) for c in classes]
	bounding_boxes = []
	for c in classes:
		print "Extracting boxes for",c
		bounding_boxes.append([get_bounding_boxes(image_sets[get_class_idx(c)][i], c) for i in xrange(len(image_sets[get_class_idx(c)]))])

	all_samples = {}
	for c in classes:
		print "Extracting images for",c
		class_idx = get_class_idx(c)
		num_images = len(image_sets[class_idx])
		num_samples = sum([len(bboxes) for bboxes in bounding_boxes[class_idx]])
		samples = np.zeros((num_samples, 64, 64, 3), dtype=np.int8)
		sample_idx = 0

		progress_meter = int(0.10 * num_images)

		for image_idx, image_name in enumerate(image_sets[class_idx]):
			if (image_idx+1) % progress_meter == 0:
				print int(100.0 * (image_idx+1)/num_images),"% done"
			for bbox in bounding_boxes[class_idx][image_idx]:
				samples[sample_idx,:,:,:] = extract_image(image_name, bbox)
				sample_idx += 1
		all_samples[c] = samples

	np.savez_compressed("processed_images", **all_samples)

def load_processed_samples(save_file):
	samples = np.load(save_file)
	all_samples = []
	for c in classes:
		all_samples.append(samples[c])
	return all_samples

def load_voc_data():
	NEGATIVE_SAMPLES = 3
	processed_file = "processed_images.npz"
	if not os.path.isfile(processed_file):
		process_voc_data(processed_file)
	all_samples = load_processed_samples(processed_file)

	negative_classes = [c for c in classes if c != 'cat']

	# Collect positive samples
	x_positive = all_samples[get_class_idx('cat')]
	num_positive = x_positive.shape[0]
	idx = np.random.permutation(num_positive)
	train_start, train_end = 0, int(0.7*num_positive)
	val_start, val_end = train_end+1, int(0.9*num_positive)
	test_start, test_end = val_end, num_positive
	x_train_positive = x_positive[idx[train_start:train_end]]
	x_val_positive = x_positive[idx[val_start:val_end]]
	x_test_positive = x_positive[idx[test_start:test_end]]

	# Collect negative samples
	x_negatives = []
	for c in negative_classes:
		x_negatives.append(all_samples[get_class_idx(c)])
	x_negative = np.concatenate(tuple(x_negatives), 0)
	idx = np.random.permutation(x_negative.shape[0])
	num_negative = NEGATIVE_SAMPLES*x_positive.shape[0]
	x_negative = x_negative[idx[:num_negative]]
	train_start, train_end = 0, int(0.7*num_negative)
	val_start, val_end = train_end+1, int(0.9*num_negative)
	test_start, test_end = val_end, num_negative
	x_train_negative = x_negative[train_start:train_end]
	x_val_negative = x_negative[val_start:val_end]
	x_test_negative = x_negative[test_start:test_end]

	print x_train_negative.shape
	print x_val_negative.shape
	print x_test_negative.shape

	x_train = np.concatenate((x_train_positive, x_train_negative),0)
	y_train = np.concatenate((np.ones(x_train_positive.shape[0], dtype=np.int), np.zeros(x_train_negative.shape[0], dtype=np.int)), 0)
	idx = np.random.permutation(x_train.shape[0])
	x_train = x_train[idx]
	y_train = transform_to_2d(y_train[idx])

	x_val = np.concatenate((x_val_positive, x_val_negative),0)
	y_val = np.concatenate((np.ones(x_val_positive.shape[0], dtype=np.int), np.zeros(x_val_negative.shape[0], dtype=np.int)), 0)
	idx = np.random.permutation(x_val.shape[0])
	x_val = x_val[idx]
	y_val = transform_to_2d(y_val[idx])

	x_test = np.concatenate((x_test_positive, x_test_negative),0)
	y_test = np.concatenate((np.ones(x_test_positive.shape[0], dtype=np.int), np.zeros(x_test_negative.shape[0], dtype=np.int)), 0)
	idx = np.random.permutation(x_test.shape[0])
	x_test = x_test[idx]
	y_test = y_test[idx]
	y_test = transform_to_2d(y_test)

	print "Num Train:",x_train.shape[0]
	print "Num Val:",x_val.shape[0]
	print "Num Test:",x_test.shape[0]

	return x_train, y_train, x_val, y_val, x_test, y_test

def main():
	load_voc_data()	



if __name__ == '__main__':
	main()