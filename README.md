# Deep Learning for cat detection in the wild!

Table of Contents
-----------------

  * [Deep network architecture](#the-network-catnet)
  * [Data](#the-data-pascal-voc-2012)
  * [Results](#results)
    * [Full Results](#full-results)
  * [Discussion](#discussion)
  * [Instructions to run](#instructions-to-run)

## The network: CatNet
![Deep Network](/report/network.jpg)
**CatNet** is inspired by the famous **AlexNet** and the **FastRCNN** paper. Since all of the computation for this project was done using a CPU, a delicate balance was needed between architecture depth and processing time. The inputs are `128x128` images, which are passed through a total of four convolution layers, before splitting off into two output heads - one for image classification to tell us if the image is that of a cat or not, and the other to tell us if the cat is slightly off-image, and if yes, by how much. _Further details in the [Discussion](#discussion) section._

## The data: PASCAL VOC 2012
We use the PASCAL VOC 2012 dataset - it contains 19 classes of various objects from animals to furniture. It also contains annotations for multiple objects in any given image - _bounding boxes_ that accurately define the location of an object in the image. In our case, we build our data as follows:
 * **Positive images:** Cats
   * For all the images in the dataset, we extract the cats as per the ground truth bounding boxes. We also consider bounding boxes that are _close_ to the ground truth bounding boxes - this process is done using _Selective Search_.
 * **Negative images:** Other objects and background
   * For all other objects in the dataset, we extract similar images as for cats - ground truth bounding boxes as well as other _close_ bounding boxes.
   * For the background class, we extract boxes that are _far away_ from all ground truth bounding boxes, i.e. do not contain any object according to the annotations

_Data selection and splitting is explained further in the [Discussion](#discussion) section._

Samples from the dataset:

![VOC2012](/report/voc_2012_samples.png)
## Results
### The good
### The not-so-good
### Full Results

## Discussion
* What network/model did you select, and why?
* Describe your pipeline and pre-processing steps?
* What steps did you take to get the best accuracy?
* How long did your training and inference take, how could you make these faster?
* If you had more time, how would you expand on this submission?

## Instructions to run
