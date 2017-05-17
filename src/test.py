import dlib
import numpy as np

from PIL import Image
from PIL import ImageDraw
from keras.models import model_from_json
from keras.models import load_model

from voc_processor import *
import keras

import os
import cPickle as cp

import sys

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
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
	idxs = np.argsort(y2)
 
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
	return boxes[pick].astype("int")

model = load_model('model.h5')

TEST_SET_PATH = "../../HiringExercise_MLCVEngineer/test_set/"
# TEST_SET_PATH = "../data/voc2012/VOC2012/JPEGImages/"

# processed_file = "processed_images.npz"
# all_samples = load_processed_samples(processed_file)
# x_positive = all_samples[get_class_idx('cat')]

# for i in xrange(x_positive.shape[0]):
#     iii = Image.fromarray(x_positive[i], mode='RGB')
#     iii.save("test/results/" + str(i) + ".png")

# # y_pred = model.predict(x_positive, batch_size=64, verbose=True)
# # # print y_pred
# # for i in xrange(y_pred.shape[0]):
# #     print y_pred[i]
# #     if y_pred[i,1] > 0.5:
# #         print 'detected!!!'

# import sys
# sys.exit(1)
all_regions = dict()
for idx, f in enumerate(os.listdir(TEST_SET_PATH)):
	print "Image: " + f
	im = Image.open(TEST_SET_PATH + f)
	# im = Image.open("../data/voc2012/VOC2012/JPEGImages/2007_003778.jpg")
	img = np.array(im)
	if len(img.shape) == 2:
		# Black and White Image
		img = np.tile(img, 2)
	elif img.shape[2] == 4:
		# With Alpha
		img = img[:,:,0:3]

	# img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
	all_regions[f] = []
	dlib.find_candidate_object_locations(img, all_regions[f], min_size=500)
	print len(all_regions[f])
	candidates = list()
	for i, r in enumerate(all_regions[f]):
	#     # excluding same rectangle (with different segments)
	#     # if r['rect'] in candidates:
	#     #     continue
	#     # # excluding regions smaller than 2000 pixels
	#     # if r['size'] < 2000:
	#     #     continue
	#     # # distorted rects
	#     # x, y, w, h = r['rect']
	#     # if w / h > 1.2 or h / w > 1.2:
	#     #     continue
		candidates.append([r.left(), r.top(), r.right(), r.bottom()])
		if i > 10000:
			break

	# # print len(regions)
	candidates = list(candidates)
	print "Num candidates:",len(candidates)

	X_test = np.zeros((len(candidates), 64, 64, 3))

	for idx, bbox in enumerate(candidates):
		# x,y,w,h = bbox
		# bounding_box = [x, y, x+w, y+h]
		bounding_box = bbox
		# print im.size, bbox, bounding_box
		cropped_im = im.crop(bounding_box)
		cropped_im = cropped_im.resize((64,64), resample=Image.BILINEAR)
		# cropped_im.save("test/results/" + str(idx) + ".png")
		# im.save("test/00" + str(idx) + ".png")
		X_test[idx,:,:,:] = np.array(cropped_im)[:,:,0:3]

	X_test = X_test.astype('float32')
	X_test /= 255
	y_pred = model.predict(X_test, batch_size=128, verbose=True)
	accepted_bboxes = []
	for i in xrange(y_pred.shape[0]):
		if y_pred[i,1] > 0.5:
			accepted_bboxes.append(candidates[i])

	print len(accepted_bboxes)
	accepted_bboxes = np.stack(accepted_bboxes, axis=0)
	final_bboxes = non_max_suppression_fast(accepted_bboxes, 0.3)

	print "Total candidates: ",y_pred.shape[0]
	print "Positive candidates: ",accepted_bboxes.shape[0]
	print "Final candidates: ",final_bboxes.shape[0]

	im = Image.open(TEST_SET_PATH + f)
	draw = ImageDraw.Draw(im)
	for i in xrange(final_bboxes.shape[0]):
		draw.rectangle(list(final_bboxes[i]), outline="green")
	del draw
	im.save("test/results/" + f + ".png", "PNG")
	break
	# for idx, bbox in enumerate(candidates):
	#     x,y,w,h = bbox
	#     bounding_box = [x, y, x+w, y+h]
	#     if y_pred[idx,1] > 0:
	#         draw.rectangle(bounding_box, outline="green")
	#     else:
	#         draw.rectangle(bounding_box, outline="red")
	# del draw
	# im.save("a.png", "PNG")

# with open("regions.pkl",'w') as fp:
#     cp.dump(all_regions, fp)