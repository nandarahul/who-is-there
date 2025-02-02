# Object detection using YOLOv3 and Face Recognition using FaceNet
  ### Object Detection
Image understanding is a hard problem in computer science, even though humans are good at segmenting, detecting the objects and understanding the scene in real time. Object detection is inherently harder than a general image classification due to the possibility of multiple objects in a single image. Earlier methods dealt with running classifiers in different regions of an image and hence were inefficient and hardly of any use in real life.   

You Only Look Once (YOLO) (https://arxiv.org/abs/1506.02640) is a state of the art object detection algorithm. It introduces an innovative approach of viewing object detection as a regression problem to get bounding boxes of objects (and their class) in the image. A single neural network predicts bounding boxes and class probabilities directly in a single evaluation. This repository serves as an implementation of YOLOv3 (https://arxiv.org/abs/1804.02767) using PyTorch. We used transfer learning to use the pre-trained weights made available by the original authors. The original network was trained on the MS COCO dataset. We also integrated the object detection workflow with Face Recognition using FaceNet (https://arxiv.org/abs/1503.03832) that we'll talk about soon. 

### Network Design
The network has 53 convolutional layers and is also called Darknet-53.
#### Yolo Layers:
*Route Layer* : Concatenates the feature maps from the list of layers specified  
*Shortcut Layer* : It is a skip connection similar to ResNet, which adds the feature map from a past layer to the current one  
*Convolution Layer* : It is a convolutional layer with batch norm and leaky Relu  
*Upsample Layer* : Upsample the feature map by a striding factor  
*YOLO Layer* : A detection layer that outputs bounding boxes and corresponding object predictions, done at three different scales of grid over the image.

### Output
We ran our implementation on images and videos. Here are some sample outputs:

<img src="/output/det_hicks.jpg" width="65%" />

#### Video:
(View in 1080p)  
<a href="http://www.youtube.com/watch?v=WlvFTBsUcKk
" target="_blank"><img src="http://img.youtube.com/vi/WlvFTBsUcKk/0.jpg" 
alt="Object Detection on a video" width="440" height="340" border="10" /></a>

# Face Recognition using FaceNet
There have been various techniques developed in Computer Vision for face recognition, but recent advances in machine learning have greatly improved the accuracy and performance of neural network based methods like FaceNet (https://arxiv.org/abs/1503.03832) which achieved 99.63% accuracy on LFW (Labeled Faces in the Wild) dataset outperforming most humans.   
FaceNet learns to map a face to a 128 bit vector embedding using a triplet loss function that maximizes the distance between vectors of non matching faces and minimizes the distance between similar ones, using *Siamese Network* setup.

### FaceNet Architecture
We have used ResNet-18 as the core architecture for FaceNet and trained it using triplet loss function on the deep-funneled LFW (Labeled Faces in the Wild) dataset. We modify the final fully connected layer in resnet to map to 128 output features which will be the embedding for the face in the image. The training dataset consisted of 4913 images belonging to 426 classes (people), while the test set contained 1072 images.   
For training, we only used classes in LFW which had at least 5 images to allow for a 80/20 split of the data into training and testing dataset.

### Face Recognition
We tried two methods for using the embeddings to recognize a face:   
1) Compute the embedding of the test face, find the class of the training image to which the image has the minimum l2 distance in the embeddings space
2) Train a Multiclass classifier with the embeddings as the input features and the class labels.

# YOLO + FaceNet !
After implementing YOLO and FaceNet independently, we integrated them such that if any “person” is detected in an image, we pass it through FaceNet. Since FaceNet expects only the face of a person, we use HOG detector from dlib library to extract a face from the image. After passing the face image through FaceNet, if the face is recognized, we display the corresponding name/label of the person in the bounding box instead of the generic label “person”. In order to prevent misclassification of unseen faces, we threshold the distance to a particular class's image before classifying them.   
Here are some sample outputs:

<img src="/output/image3.jpg" width="55%" />   
(We had some misclassifications too, like the second person from left in the image above, the distance the the shown class was very close to the distance of a true positive.)   

</br>
<img src="/output/image6.png" width="55%" />    
The Flash needs a better mask!

# Discussion
YOLO only takes around 1.4 seconds to process a 4032 x 3024 image on a regular PC without a GPU which is quite efficient and thus can be conveniently run on videos as well.   
We trained the Facenet model for about 72 hours on a cuda cluster with 6 cores of 12GB TITAN Xp GPUs, (some initial training was done on lesser hardware ~2GB in our laptop) and were able to achieve a considerable accuracy of 53.26% out of 1072 test images LFW dataset (with 4913 training images over 426 classes), which is much better than random chance, i.e. 1/426 x 100 = 0.2 %. We believe training on all possible triplets and more data will improve the test accuracy even more.


You can download the weights file for YOLO v3 from [here](https://pjreddie.com/media/files/yolov3.weights)
