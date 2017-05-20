## Data processor
# Utilities to load/process/prepare data to train a
# network to detect cats. Contains processors for 
# CIFAR and VOC sets.
#
# Author: Fahim Dalvi

import cPickle
import dlib
import numpy as np
import os
import random
import xml.etree.ElementTree

from PIL import Image, ImageDraw
DATA_PATH = '../data/voc2012/VOC2012/'
classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','tvmonitor']

def get_class_idx(class_name):
	"""
	Given a class name, return its canonical index in the system
	
	Parameters
	----------
	class_name : string
	Class Name.

	Returns
	-------
	class_idx : number
	Index of given class. -1 if class is illegal.
	"""
	return classes.index(class_name)

def get_class_name(class_idx):
	"""
	Given a class index, return its canonical name in the system
	
	Parameters
	----------
	class_idx : number
	Class Index

	Returns
	-------
	class_idx : number
	Name corresponding to the given class index.
	"""
	return classes[class_idx]

def transform_to_2d(label_array, num_classes=2):
	"""
	Given a 1d label array, convert it into a matrix that
	can be fed to a optimization algorithm (i.e each row
	should be a 1-hot vector of the correct class)
	
	Parameters
	----------
	label_array : numpy vector [N]
	Vector of labels
	num_classes : number
	Total number of classes `n_classes`.

	Returns
	-------
	label_matrix : numpy matrix [N x n_classes]
	Label matrix to be given to optimization algorithm
	"""
	num_examples = label_array.shape[0]
	result = np.zeros((num_examples, num_classes))
	result[np.arange(num_examples),label_array] = 1
	return result

def get_class_image_names(class_name, include_difficult=False, data_path=DATA_PATH):
	"""
	Given a class name, extract all images containing objects
	of that class from the VOC dataset.
	
	Parameters
	----------
	class_name : string
	Class name for which objects need to be extracted for.
	include_difficult : boolean
	VOC dataset comes with some "difficult" samples. This flag
	sets if those objects should be in the output as well.
	data_path : string
	Path to VOC dataset

	Returns
	-------
	class_images : list
	List of all images containing objects of the given class
	"""
	class_images = []
	with open(data_path + 'ImageSets/Main/' + class_name + '_trainval.txt') as fp:
		for line in fp:
			image_name, label = line.strip().split()
			label = int(label)
			if label == 1 or (include_difficult and label == 0):
				class_images.append(image_name)
	return class_images

def get_bounding_boxes(image_name, class_name, data_path=DATA_PATH):
	"""
	Given a image and a class, extract the bounding boxes for the given
	class in the given image.
	
	Parameters
	----------
	image_name : string
	Name of the image
	class_name : string
	Class name for which objects need to be extracted for.
	data_path : string
	Path to VOC dataset

	Returns
	-------
	bounding_boxes : list
	List of bounding boxes of the given object in the given image
	"""
	bounding_boxes = []
	e = xml.etree.ElementTree.parse(data_path + 'Annotations/' + image_name + '.xml').getroot()
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

def extract_image(image_name, bounding_box, res=128):
	"""
	Given a image and a bounding box, extract the sub-image.
	
	Parameters
	----------
	image_name : string
	Name of the image
	bounding_box : list [x1 y1 x2 y2]
	Bounding box for the sub-image to be extracted.
	res : number
	Resolution to which the samples must be rescaled
	
	Returns
	-------
	sub_img : np matrix [res x res x 3]
	Sub image extracted from the original image.
	"""
	im = Image.open(DATA_PATH + 'JPEGImages/' + image_name + ".jpg")
	cropped_im = im.crop(bounding_box)
	cropped_im = cropped_im.resize((res,res), resample=Image.BILINEAR)
	return np.array(cropped_im)

def IoU(bbox1, bbox2):
	"""
	Compute Intersection over union of two bounding boxes
	
	Parameters
	----------
	bbox1 : list [x1 y1 x2 y2]
	First bounding box.
	bbox2 : list [x1 y1 x2 y2]
	Second bounding box.
	
	Returns
	-------
	iou : number
	Intersection over Union of given boxes
	"""
	inter_x1 = max(bbox1[0], bbox2[0])
	inter_y1 = max(bbox1[1], bbox2[1])
	inter_x2 = min(bbox1[2], bbox2[2])
	inter_y2 = min(bbox1[3], bbox2[3])


	if inter_y1 > inter_y2 or inter_x1 > inter_x2:
		return 0.0

	interection_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
	bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
	bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

	# Check if boxes are exactly equal
	if (bbox1_area + bbox2_area - interection_area) == 0:
		return 1
	return interection_area / (bbox1_area + bbox2_area - interection_area)

def process_voc_detection_data(save_file, res=128):
	"""
	Extract positive and negative samples from the VOC dataset.
	
	Parameters
	----------
	save_file : string
	path to save file so we don't have to do this over and over again.
	res : number
	Resolution to which the samples must be rescaled
	
	Returns
	-------
	None
	"""
	image_sets = [get_class_image_names(c) for c in classes]
	bounding_boxes = []
	for c in classes:
		print "Extracting boxes for",c
		bounding_boxes.append([get_bounding_boxes(image_sets[get_class_idx(c)][i], c) for i in xrange(len(image_sets[get_class_idx(c)]))])

	all_samples = {}
	background_samples = []
	background_bbox_deltas = []
	for c in classes:
		print "Extracting images for",c
		class_idx = get_class_idx(c)
		num_images = len(image_sets[class_idx])
		num_samples = sum([len(bboxes) for bboxes in bounding_boxes[class_idx]])
		samples = []
		bbox_deltas = []
		ious = []
		sample_idx = 0

		progress_meter = int(0.10 * num_images)

		for image_idx, image_name in enumerate(image_sets[class_idx]):
			if (image_idx+1) % progress_meter == 0:
				print int(100.0 * (image_idx+1)/num_images),"% done"
			for bbox in bounding_boxes[class_idx][image_idx]:
				im = Image.open(DATA_PATH + 'JPEGImages/' + image_name + ".jpg")
				img = np.array(im)
				regions = []
				dlib.find_candidate_object_locations(img, regions, min_size=500)

				positive_bounding_boxes = []
				negative_bounding_boxes = []
				for r in regions:
					if len(positive_bounding_boxes) > 10 and len(negative_bounding_boxes) > 10:
						break
					curr_bbox = [r.left(), r.top(), r.right(), r.bottom()]
					iou = IoU(bbox, curr_bbox)
					if iou > 0.5:
						positive_bounding_boxes.append(curr_bbox + [iou])

					if iou < 0.3:
						negative_bounding_boxes.append(curr_bbox + [iou])

				random.shuffle(positive_bounding_boxes)
				random.shuffle(negative_bounding_boxes)

				positive_bounding_boxes = positive_bounding_boxes[:4]
				negative_bounding_boxes = negative_bounding_boxes[:3]

				# Get ground_truth
				cropped_im = im.crop(bbox)
				cropped_im = cropped_im.resize((res,res), resample=Image.BILINEAR)
				samples.append(np.array(cropped_im))
				bbox_deltas.append([0,0,0,0])
				ious.append(IoU(bbox, bbox))

				# Get positive samples
				for p_bbox in positive_bounding_boxes:
					cropped_im = im.crop(p_bbox[:4])
					cropped_im = cropped_im.resize((res,res), resample=Image.BILINEAR)
					samples.append(np.array(cropped_im))
					bbox_deltas.append([
						bbox[0]-p_bbox[0],
						bbox[1]-p_bbox[1],
						bbox[2]-p_bbox[2],
						bbox[3]-p_bbox[3],
					])
					ious.append(p_bbox[4])

				# Get negative samples
				for n_bbox in negative_bounding_boxes:
					cropped_im = im.crop(n_bbox[:4])
					cropped_im = cropped_im.resize((res,res), resample=Image.BILINEAR)
					background_samples.append(np.array(cropped_im))
					background_bbox_deltas.append([0,0,0,0])
		samples = np.stack(samples, axis=0)
		bbox_deltas = np.stack(bbox_deltas, axis=0)
		print c,samples.shape, bbox_deltas.shape,num_samples
		all_samples["samples_" + c] = samples
		all_samples["bboxdeltas_" + c] = bbox_deltas

	background_samples = np.stack(background_samples, axis=0)
	background_bbox_deltas = np.stack(background_bbox_deltas, axis=0)
	print "background",background_samples.shape, background_bbox_deltas.shape
	all_samples["samples_background"] = background_samples
	all_samples["bboxdeltas_background"] = background_bbox_deltas

	np.savez_compressed(save_file, **all_samples)

def visualize_processed_files(save_file, num_samples=10):
	"""
	Used the processed data to create an image with random 
	samples from each class
	
	Parameters
	----------
	save_file : string
	Path where image should be saved
	num_samples : number
	Number of samples per class
	
	Returns
	-------
	None
	"""
	samples = np.load(save_file)
	print("Loaded data.")
	num_classes = 20
	text_offset = 100
	full_image = np.zeros((64*num_classes, text_offset+64*num_samples, 3), dtype=np.int8)
	print len(classes + ["background"])
	for c_idx, c in enumerate(classes + ["background"]):
		idx = np.random.permutation(samples["samples_" + c].shape[0])
		idx = idx[:num_samples]
		random_samples = samples["samples_" + c][idx]
		random_bboxes = samples["bboxdeltas_" + c][idx]

		for s_idx in xrange(num_samples):
			tmp = random_samples[s_idx,:,:,:]
			full_image[c_idx*64:(c_idx+1)*64,text_offset+(s_idx*64):text_offset+((s_idx+1)*64),:] = tmp
	full_image[:,0:text_offset,:] = 255
	img = Image.fromarray(full_image , mode="RGB")
	draw = ImageDraw.Draw(img)
	for c_idx, c in enumerate(classes + ["background"]):
		draw.text((10, 25+c_idx*64), c, fill='black')
	del draw
	img.save("full_samples.png")

def load_voc_data(processed_file="processed_images_detection_128.npz", negative_ratio = 1):
	"""
	Load VOC processed data and create train/dev/test split.
	
	Parameters
	----------
	processed_file : string
	File saved by `process_voc_detection_data`
	negative_ratio : number
	Ratio of negative samples to positive samples in splits
	
	Returns
	-------
	x_train : numpy matrix [N x 128 x 128 x 3]
	Training samples
	y_train : numpy matrix [N x 1]
	Training labels
	x_val : numpy matrix [M x 128 x 128 x 3]
	Validation samples
	y_val : numpy matrix [M x 1]
	Validation labels
	x_test : numpy matrix [T x 128 x 128 x 3]
	Testing samples
	y_test : numpy matrix [T x 1]
	Testing labels
	x_train_bbox : numpy matrix [A x 128 x 128 x 3]
	Bbox training samples (Also positive classification samples)
	y_train_bbox : numpy matrix [A x 4]
	Bbox training regression labels
	x_val_bbox : numpy matrix [B x 128 x 128 x 3]
	Bbox validation samples 
	y_val_bbox : numpy matrix [B x 4]
	Bbox validation regression labels
	x_test_bbox : numpy matrix [C x 128 x 128 x 3]
	Bbox testing samples 
	y_test_bbox : numpy matrix [C x 4]
	Bbox testing regression labels
	x_extra_negatives : numpy matrix [Z x 128 x 128 x 3]
	Extra negative samples for hard negative mining
	"""
	if not os.path.isfile(processed_file):
		process_voc_detection_data(processed_file)

	all_samples = np.load(processed_file)
	print("Loaded data.")

	negative_classes = [c for c in classes if c != 'cat']
	background_class = "background"

	# Collect positive samples
	x_positive = all_samples["samples_cat"]
	y_bbox = all_samples["bboxdeltas_cat"]
	num_positive = x_positive.shape[0]
	idx = np.random.permutation(num_positive)
	train_start, train_end = 0, int(0.7*num_positive)
	val_start, val_end = train_end+1, int(0.9*num_positive)
	test_start, test_end = val_end, num_positive
	x_train_positive = x_positive[idx[train_start:train_end]]
	x_train_bbox = x_train_positive
	y_train_bbox = y_bbox[idx[train_start:train_end]]
	x_val_positive = x_positive[idx[val_start:val_end]]
	x_val_bbox = x_val_positive
	y_val_bbox = y_bbox[idx[val_start:val_end]]
	x_test_positive = x_positive[idx[test_start:test_end]]
	x_test_bbox = x_test_positive
	y_test_bbox = y_bbox[idx[test_start:test_end]]
	print("Collected positive samples.")

	# Collect negative samples
	x_negatives = []
	x_extra_negatives = None
	for c in negative_classes:
		x_negatives.append(all_samples["samples_" + c])
	x_negative = np.concatenate(tuple(x_negatives), 0)
	idx = np.random.permutation(x_negative.shape[0])
	num_negative_objs = int(negative_ratio*0.5*x_positive.shape[0])
	x_extra_negatives = x_negative[idx[num_negative_objs:num_negative_objs*4]]
	x_negative = x_negative[idx[:num_negative_objs]]
	print("Collected negative samples.")

	x_background = all_samples["samples_background"]
	idx = np.random.permutation(x_background.shape[0])
	num_negative_background = num_negative_objs
	x_extra_negatives = np.concatenate((x_extra_negatives,x_background[idx[num_negative_background:num_negative_background*4]]), 0)
	x_background = x_background[idx[:num_negative_background]]
	print("Collected background samples.")

	x_negative = np.concatenate(tuple([x_negative, x_background]), 0)
	idx = np.random.permutation(x_negative.shape[0])
	x_negative = x_negative[idx]

	num_negative = num_negative_objs + num_negative_background
	train_start, train_end = 0, int(0.7*num_negative)
	val_start, val_end = train_end+1, int(0.9*num_negative)
	test_start, test_end = val_end, num_negative
	x_train_negative = x_negative[train_start:train_end]
	x_val_negative = x_negative[val_start:val_end]
	x_test_negative = x_negative[test_start:test_end]

	print x_train_negative.shape
	print x_val_negative.shape
	print x_test_negative.shape

	print "Num cats:",num_positive
	print "Num other objects:",num_negative_objs
	print "Num random background:",num_negative_background
	print "Num extra for hard mining:", x_extra_negatives.shape[0]

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

	x_train = x_train.astype('float32')
	x_val = x_val.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_val /= 255
	x_test /= 255

	x_train_bbox = x_train_bbox.astype('float32')
	x_val_bbox = x_val_bbox.astype('float32')
	x_test_bbox = x_test_bbox.astype('float32')
	x_train_bbox /= 255
	x_val_bbox /= 255
	x_test_bbox /= 255

	x_extra_negatives = x_extra_negatives.astype('float32')
	x_extra_negatives /= 255

	print "Num Train:",x_train.shape[0]
	print "Num Val:",x_val.shape[0]
	print "Num Test:",x_test.shape[0]

	return x_train, y_train, x_val, y_val, x_test, y_test, \
			x_train_bbox, y_train_bbox, x_val_bbox, y_val_bbox, \
			x_test_bbox, y_test_bbox, x_extra_negatives

def load_voc_data_cached(cached_file = "sets_cached.npz"):
	"""
	Load train/dev/test splits from cache, and if not present,
	process them and save to cache
	
	Parameters
	----------
	cached_file : string
	Path to cache file
	
	Returns
	-------
	x_train : numpy matrix [N x 128 x 128 x 3]
	Training samples
	y_train : numpy matrix [N x 1]
	Training labels
	x_val : numpy matrix [M x 128 x 128 x 3]
	Validation samples
	y_val : numpy matrix [M x 1]
	Validation labels
	x_test : numpy matrix [T x 128 x 128 x 3]
	Testing samples
	y_test : numpy matrix [T x 1]
	Testing labels
	x_train_bbox : numpy matrix [A x 128 x 128 x 3]
	Bbox training samples (Also positive classification samples)
	y_train_bbox : numpy matrix [A x 4]
	Bbox training regression labels
	x_val_bbox : numpy matrix [B x 128 x 128 x 3]
	Bbox validation samples 
	y_val_bbox : numpy matrix [B x 4]
	Bbox validation regression labels
	x_test_bbox : numpy matrix [C x 128 x 128 x 3]
	Bbox testing samples 
	y_test_bbox : numpy matrix [C x 4]
	Bbox testing regression labels
	x_extra_negatives : numpy matrix [Z x 128 x 128 x 3]
	Extra negative samples for hard negative mining
	"""
	
	if not os.path.isfile(cached_file):
		print "Cache not found, computing sets..."
		x_train, y_train, x_val, y_val, x_test, y_test, \
			x_train_bbox, y_train_bbox, x_val_bbox, y_val_bbox, \
			x_test_bbox, y_test_bbox, x_extra_negatives = load_voc_data()
		sets = {
			'x_train': x_train,
			'y_train': y_train,
			'x_val': x_val,
			'y_val': y_val,
			'x_test': x_test,
			'y_test': y_test,
			'x_train_bbox': x_train_bbox,
			'y_train_bbox': y_train_bbox,
			'x_val_bbox': x_val_bbox,
			'y_val_bbox': y_val_bbox,
			'x_test_bbox': x_test_bbox,
			'y_test_bbox': y_test_bbox,
			'x_extra_negatives': x_extra_negatives
		}

		np.savez_compressed("sets_cached", **sets)
	else:
		print "Loading cached sets"
		sets = np.load(cached_file)

	return sets['x_train'], sets['y_train'], sets['x_val'], sets['y_val'], \
		sets['x_test'], sets['y_test'], sets['x_train_bbox'], \
		sets['y_train_bbox'], sets['x_val_bbox'], sets['y_val_bbox'], \
		sets['x_test_bbox'], sets['y_test_bbox'], sets['x_extra_negatives']


def main():
	# process_voc_detection_data('processed_images_detection_128',res=128)
	visualize_processed_files('processed_images_detection.npz')
	# x_train, y_train, x_val, y_val, x_test, y_test, \
	# 		x_train_bbox, y_train_bbox, x_val_bbox, y_val_bbox, \
	# 		x_test_bbox, y_test_bbox, x_extra_negatives = load_voc_data_cached()

if __name__ == '__main__':
	main()


######################### HELPER FUNCTIONS ###########################
############### Not explicitly used in final pipeline ################
def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

def transform_to_3d(flattened_array):
	num_examples = flattened_array.shape[0]
	return np.swapaxes(flattened_array.reshape((num_examples,32,32,3), order='F'),1,2)

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

def process_voc_classification_data(save_file):
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

def load_voc_classification_data():
	NEGATIVE_SAMPLES = 1
	processed_file = "processed_images.npz"
	if not os.path.isfile(processed_file):
		process_voc_classification_data(processed_file)
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

	x_train = x_train.astype('float32')
	x_val = x_val.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_val /= 255
	x_test /= 255

	print "Num Train:",x_train.shape[0]
	print "Num Val:",x_val.shape[0]
	print "Num Test:",x_test.shape[0]

	return x_train, y_train, x_val, y_val, x_test, y_test

def load_processed_samples(save_file):
	samples = np.load(save_file)
	all_samples = []
	for c in classes:
		all_samples.append(samples[c])
	return all_samples
