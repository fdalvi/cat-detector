import dlib
import numpy as np

from PIL import Image
from PIL import ImageDraw
from keras.models import load_model

from voc_processor import *
import keras

import os
import cPickle as cp

import sys
import h5py

def rect2lines(bbox):
	point_list = [
		bbox[0], bbox[1],
		bbox[2], bbox[1],
		bbox[2], bbox[3],
		bbox[0], bbox[3],
		bbox[0], bbox[1]
	]
	return point_list

# Malisiewicz et al.
def non_max_suppression_fast(boxes, probs, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(probs)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int"), probs[pick]

def normalize_image(img):
	if len(img.shape) == 2:
		# Black and White Image
		img = np.reshape(img, (img.shape[0],img.shape[1],1))
		img = np.repeat(img, 3, axis=2)
	elif img.shape[2] == 4:
		# With Alpha
		img = img[:,:,0:3]
	return img

def test(model, test_set_images, output_path=None, annotations_path=None):
	if (output_path is not None) and (not os.path.exists(output_path)):
		os.makedirs(output_path)

	all_regions = dict()
	mAP = 0
	mAP_count = 0
	for idx, f in enumerate(os.listdir(test_set_images)):
		ext = f.split('.')[-1]
		if ext not in ['jpg', 'jpeg', 'png']:
			print "Not an Image: " + f
			continue
		print "Image: " + f
		if output_path is not None:
			img_out_path = os.path.join(output_path,f + ".png")
			if os.path.exists(img_out_path):
				print "Already processed."
				continue
		im = Image.open(os.path.join(test_set_images,f))
		img = np.array(im)
		img = normalize_image(img)

		all_regions[f] = []
		dlib.find_candidate_object_locations(img, all_regions[f], min_size=500)
		print len(all_regions[f])
		candidates = list()
		for i, r in enumerate(all_regions[f]):
			candidates.append([r.left(), r.top(), r.right(), r.bottom()])
			if i > 2000:
				# Hard limit number of proposals
				break

		candidates = list(candidates)
		X_test = np.zeros((len(candidates), 128, 128, 3))
		for idx, bbox in enumerate(candidates):
			bounding_box = bbox
			cropped_im = im.crop(bounding_box)
			cropped_im = cropped_im.resize((128,128), resample=Image.BILINEAR)
			X_test[idx,:,:,:] = normalize_image(np.array(cropped_im))

		X_test = X_test.astype('float32')
		X_test /= 255
		y_pred = model.predict(X_test, batch_size=128, verbose=True)
		accepted_bboxes = []
		accepted_bboxes_probs = []
		for i in xrange(y_pred.shape[0]):
			if y_pred[i,1] > 0.5:
				accepted_bboxes.append(candidates[i])
				accepted_bboxes_probs.append(y_pred[i,1])

		if len(accepted_bboxes) == 0:
			final_bboxes = []
			accepted_bboxes = np.array(accepted_bboxes)
			final_bboxes = np.array(final_bboxes)
		else:
			accepted_bboxes = np.stack(accepted_bboxes, axis=0)
			accepted_bboxes_probs = np.array(accepted_bboxes_probs)
			print accepted_bboxes.shape, accepted_bboxes_probs.shape
			final_bboxes, final_probs = non_max_suppression_fast(accepted_bboxes, accepted_bboxes_probs, 0.3)

			# filtered_idx = np.where(final_probs > 0.75)[0]
			# final_bboxes = final_bboxes[filtered_idx]
			# final_probs = final_probs[filtered_idx]

		print "Total candidates: ",y_pred.shape[0]
		print "Positive candidates: ",accepted_bboxes.shape[0]
		print "Final candidates: ",final_bboxes.shape[0]

		if annotations_path is not None:
			img_base = f.split('.')[0]
			gt_bboxes = get_bounding_boxes(img_base, 'cat', data_path=annotations_path)

			for gt_bbox in gt_bboxes:
				tp = 0
				fp = 0
				ious = []
				for pred_bbox_idx in xrange(final_bboxes.shape[0]):
					ious.append(IoU(gt_bbox, final_bboxes[pred_bbox_idx]))
				ious = sorted(ious)
				if len(ious) > 0:
					if ious[-1] > 0.5:
						tp += 1
						fp = len(ious)-1
					else:
						fp = len(ious)

				if (tp + fp) != 0:
					mAP += float(tp)/float(tp+fp)
					mAP_count += 1

		if output_path is not None:
			im = Image.open(os.path.join(test_set_images,f))
			line_width = int(max(im.size) * 0.005)
			draw = ImageDraw.Draw(im)
			for i in xrange(final_bboxes.shape[0]):
				draw.line(rect2lines(final_bboxes[i]), fill="green", width=line_width)
				draw.text((final_bboxes[i][0],final_bboxes[i][1]), str(final_probs[i]))
			del draw
			im.save(img_out_path, "PNG")
	print "mAP score: %0.2f"%(float(mAP)/mAP_count)

def main():
	TEST_SET_PATH = "../../HiringExercise_MLCVEngineer/test_set/"
	MODEL_PATH = "models/model_128_7/model_final.h5"
	OUT_PATH = MODEL_PATH.replace("models/","results/")

	f = h5py.File(MODEL_PATH, 'a')
	if 'optimizer_weights' in f:
		del f['optimizer_weights']
	f.close()

	model = load_model(MODEL_PATH)

	test(model, TEST_SET_PATH, output_path=OUT_PATH)
	# test(model, "../data/voc2012/VOC2012/catOnly", output_path=OUT_PATH, annotations_path="../data/voc2012/VOC2012/")

if __name__ == '__main__':
	main()