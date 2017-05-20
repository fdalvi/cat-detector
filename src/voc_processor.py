## VOC2012 processor
# Loads VOC2012 data and returns the data as [H*W*D]
#
# Author: Fahim Dalvi

import dlib
import numpy as np
import os
import random
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

def get_bounding_boxes(image_name, class_name, data_path=DATA_PATH):
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

def extract_image(image_name, bounding_box):
	im = Image.open(DATA_PATH + 'JPEGImages/' + image_name + ".jpg")
	cropped_im = im.crop(bounding_box)
	cropped_im = cropped_im.resize((64,64), resample=Image.BILINEAR)
	return np.array(cropped_im)

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

def IoU(bbox1, bbox2):
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
		# all_samples["ious_" + c] = ious

	background_samples = np.stack(background_samples, axis=0)
	background_bbox_deltas = np.stack(background_bbox_deltas, axis=0)
	print "background",background_samples.shape, background_bbox_deltas.shape
	all_samples["samples_background"] = background_samples
	all_samples["bboxdeltas_background"] = background_bbox_deltas

	np.savez_compressed(save_file, **all_samples)

def load_processed_samples(save_file):
	samples = np.load(save_file)
	all_samples = []
	for c in classes:
		all_samples.append(samples[c])
	return all_samples

def visualize_processed_files(save_file, num_samples=10):
	samples = np.load(save_file)
	print("Loaded data.")
	# all_samples = []
	# for c in classes + ["background"]:
	# 	all_samples.append(samples["samples_" + c])
	# 	all_samples.append(samples["bboxdeltas_" + c])
	num_classes = 20
	full_image = np.zeros((64*num_classes, 64*num_samples, 3), dtype=np.int8)
	print len(classes + ["background"])
	for c_idx, c in enumerate(classes + ["background"]):
		# print samples["samples_" + c].shape
		# print samples["bboxdeltas_" + c].shape
		# print samples["ious_" + c].shape
		idx = np.random.permutation(samples["samples_" + c].shape[0])
		idx = idx[:num_samples]
		random_samples = samples["samples_" + c][idx]
		random_bboxes = samples["bboxdeltas_" + c][idx]
		# random_ious = samples["ious_" + c][idx]

		# print c_idx
		for s_idx in xrange(num_samples):
			tmp = random_samples[s_idx,:,:,:]
			# print tmp.shape, s_idx, full_image[s_idx*64:(s_idx+1)*64,c_idx*64:(c_idx+1)*64,:].shape
			# print full_image.shape
			full_image[c_idx*64:(c_idx+1)*64,s_idx*64:(s_idx+1)*64,:] = tmp
			# print random_bboxes[s_idx], random_ious[s_idx]
		# break
	img = Image.fromarray(full_image , mode="RGB")
	img.save("full_samples.png")

def load_voc_data():
	NEGATIVE_SAMPLES = 1
	processed_file = "processed_images_detection_128.npz"
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
	num_negative_objs = int(NEGATIVE_SAMPLES*0.5*x_positive.shape[0])
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

def load_voc_data_cached():
	cached_file = "sets_cached.npz"
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
	# x_train, y_train, x_val, y_val, x_test, y_test = load_voc_data()
	# print np.sum(y_test, axis=0)
	# process_voc_detection_data('processed_images_detection_128',res=128)
	# visualize_processed_files('processed_images_detection.npz')
	x_train, y_train, x_val, y_val, x_test, y_test, \
			x_train_bbox, y_train_bbox, x_val_bbox, y_val_bbox, \
			x_test_bbox, y_test_bbox, x_extra_negatives = load_voc_data_cached()

	print x_train[0]
	print x_train_bbox[0]
	print x_extra_negatives[0]

if __name__ == '__main__':
	main()