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
![Good Results](/report/good-lowres.png)
Characteristics of good results were images with unocluded cats, especially faces. The network learned well to recognize cat eyes as well, and possibly used that to make several classifications.
### The not-so-good
![Not so Good Results](/report/not-so-good-lowres.png)
In the images that are mis-classified, the majority contain objects with features similar to cats, like dogs or people. Distinct dogs or people with no occlusion are generally classified correctly, but there are times when the network makes a mistake on these. Random backgrounds are correctly classified most of the times. Some cats that have occluded faces are also misclassified by the network. _Further details in the [Discussion](#discussion) section._
### Full Results
Full results are available in the `results/` folder. For quick viewing, visit the https://github.com/fdalvi/cat-detector/blob/master/results/RESULTS.md page. Images here are normalized to have constant width and small file size, so the page loads fairly quickly.

Full resolution outputs can be found in `results/full-resolution` and the corresponding https://github.com/fdalvi/cat-detector/blob/master/results/full-resolution/RESULTS.md page, but take significantly longer to load.

## Discussion
### What network/model did you select, and why?

The network used in this project was a custom network inspired from *AlexNet* and *FastRCNN*. Deeper networks such as those based on Google's *ResNet* or *VGGNet* were not used because all computation was done on a CPU for this project, and these deeper networks take significantly longer to train.

The *CatNet* consists of four convolutional layers and two output heads, one for classification and one for bounding box regression. The training process was as follows:
1. Train classification network for 15 epochs on balanced positive and negative class data.
2. Use trained network to mine for _hard_ negative samples, i.e. samples we know that are negative but are currently misclassified by the network.
3. Used mined negative samples and positive samples (with a ratio of 2:1) to further train the network for 15 epochs.
4. Repeat steps 2-3 one more time, to have a total of 2 hard-mine iterations.
5. Freeze the convolution layers and train the bounding box regressor for 60 epochs.

The number of epochs was chosen based on training loss, which stabilized by the 45th epoch:

![Training Progress](/report/training_progress.png)

The actual samples chosen are described in the next question. We used negative to positive samples ratio of 2:1 in the second and third training periods to focus more on the negative examples. We still provide the positive examples inorder to avoid _catrastrophic forgetting_ in the network.

The network also takes advantage of multi-task learning, since both the classification and regression are closely related. This idea was introduced in the FastRCNN paper, which was unlike the RCNN paper, where separate SVM's were trained.

###  Describe your pipeline and pre-processing steps?

The dataset we use is the PASCAL VOC 2012 dataset, which has annotations for bounding boxes around various objects. As mentioned earlier, we consider both the exact bounding boxes as per the annotations, but also bounding boxes that are _close_ to the ground truth bounding boxes. The _closeness_ is measured using `Intersection Over Union`, with a value of 0.5 or higher signifying close bounding boxes. Bounding boxes that are _far away_ from all ground truth bounding boxes were chosen to belong to a **background** class. During our preprocessing, we extract all of the images given these bounding boxes and resize them to `128x128`, the size expected by our network.

In the first training period, our training data is divided as follows: 50% `cat` images, 25% `other objects` images and 25% `background` images. In the second and third training period, our training data is divided as follows: 33% `cat` images and 67% `hard negative` images chosen randomly from the set of all negative images.

At the end, we have about ~5000 images per class (including `background` class). We use a 70-20-10 train-dev-test split. We do not do any other preprocessing such as data augmentation by image transformation or colorspace transformation, as this is left for future work. All images were normalized to have pixel color values between 0 and 1, as these networks tend to perform better with normalized images.

The overall pipeline for training is as follows:
1. Extract image samples as described above. Candidates for non ground-truth bounding boxes are generated using _Selective Search_, and `IntersectionOverUnion` is used to define the closeness of these generated bounding boxes.
2. Train the model as described in the previous question.

The overall pipeline for testing is as follows:
1. Given a test image, extract region proposals using _Selective Search_
2. Extract and resize all proposals to `128x128` pixels
3. Perform a Forward pass through the network for all proposals
4. Offset all bounding boxes for positive proposals based on the regressor's output
5. Filter positive predictions with a threshold (0.75) to choose only high-confidence outputs
6. Perform _Non-Maximum suppression_ using probabilities as the ordering to reduce number of proposals
7. Perform _Non-Maximum suppression_ using areas as the ordering to further reduce number of proposals
8. Output image with remaining proposals

### What steps did you take to get the best accuracy?

To obtain the best accuracy, we tried several architectures that were shallower/deeper - and chose the current architecture as the best balance between accuracy and processing time. We used a held out set of 100 images from the PASCAL VOC 2012 validation set to compute our overall mAP on the set, to get a quantitative measure of the overall system's performance.

We also tried various ratios of positive to negative class samples for each period, as well as the ratio of `background` class to the rest of the negative classes. Using a ratio of 1:1 in the first period and 2:1 in the second and third period was crucial to obtaining good accuracy on both positive and negative samples.

The input image size was another dimension that affects the accuracy significantly. On one hand, smaller image sizes help us train deeper networks, but also provide the network fewer pixels to learn from. We tried several image dimensions like 32, 64, 128 - and finally settled on `128x128` as a good balance between processing time and accuracy.

### How long did your training and inference take, how could you make these faster?

All training and inference was performed on a _Quad Core 3.2 GHz Intel Core i5_ with _16GB memory_. Training takes about 7 hours end-to-end including feature extraction, classifier training and bounding box regressor training.

Inference takes around 3s per image to extract regions of interest, and ~27s for classification and ~27 seconds for regression. Currently the classifier and regressor forward-passes are done separately, but since they share the lower layers, a lot of computation can be saved. This is left as future work.

The computation can be made faster by using higher batch sizes (since we propose ~2000 regions per image, processing more of them in parallel will improve runtime). A higher batch size will require more memory. We can also perform the inference on a GPU, which empirically gives a 100x improvement on forward-pass computation. Finally, the computation for the lower layers is currently repeated, but can be avoided since the lower layers are shared.

_On a side note, thanks to this exercise, a contribution was made to the https://github.com/yaroslavvb/tensorflow-community-wheels repository, which is a place for pre-built tensorflow binaries on various systems. The default tensorflow package provided with `pip` does not support advanced instruction sets which improves training time by about 3x. The contributed package is easy to install and provides users on macOS a faster way to train their networks on their CPU's!_

### If you had more time, how would you expand on this submission?

Several potential expansions are possible with more time and/or resources:
1. Use a deeper network such as *VGGNet*: Vision has consistently improved with deeper architectures
2. Use more data: currently we have only used the PASCAL VOC 2012 dataset, but we can also use datasets such as ImageNet, which are significantly larger
3. Use data augmentation: We can see the our network does not do well on different color spaces such as black and white images - we can augment our data for various color spaces. Cats specifically are also known to be found in all kinds of orientations, so data augmentation using image transformation would also potentially help with the accuracy.
4. Train a multi-class network: Currently we have only used two classes - cats and non-cats. Including more classes can help the network perform better, especially with false-negatives. Since we are currently bundling dogs, people, furniture and random backgrounds all into one `non-cat` class, we are potentially making it harder for the network to keep track of all of these separately.

## Instructions to run
There are two main scripts to this project, both in the `src/` folder. The following pre-requisites need to be satisified for the project to run:
* Python 2.7
* `virtualenv`
* Python build with boost bindings (and subsequently `boost` as well)
* `tar` and `unzip` utilities

On MacOS, you can use the following to install boost and boost-bindings for python:
```
brew install boost --with-python
brew install boost-python
```

On Linux, you can use the following:
```
sudo apt-get install libboost-all-dev
```

Once you have these pre-requisites loaded, use the following to download the datasets (VOC and testdata):
```
cd data/voc2012
curl -o vocdata.tar https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
tar -xvf vocdata.tar
cd ../..

cd data/test
curl -o test_set.zip https://raw.githubusercontent.com/CMIST/HiringExercise_MLCVEngineer/master/test_set.zip
unzip test_set.zip
cd ../..
```

Now, create a virtual environment for the python requirements:
```
virtualenv .env
. .env/bin/activate
pip install -r requirements.txt
```

The environment should be all setup now! Its time to train the network:
```
cd src
python train.py models/
python test.py models/
```

The models will be saved in the `models/` directory, while the results in `results/model_final.h5/` directory.

