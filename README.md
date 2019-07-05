# Object detection using YOLO v3 
You Only Look Once (YOLO) (https://arxiv.org/abs/1506.02640) is a state of the art object detection algorithm. This repository serves as implementation of YOLOv3 (https://arxiv.org/abs/1804.02767) using PyTorch. We used transfer learning to use the pre-trained weights made available by the original authors. The original network was trained on the MS COCO dataset. We also integrated the object detection workflow with Face Recognition using FaceNet (https://arxiv.org/abs/1503.03832) that we'll talk about soon. 

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
(Watch in HD)

<a href="http://www.youtube.com/watch?v=WlvFTBsUcKk
" target="_blank"><img src="http://img.youtube.com/vi/WlvFTBsUcKk/0.jpg" 
alt="Object Detection on a video" width="440" height="340" border="10" /></a>


Please download the weights file for YOLO v3 from [here](https://pjreddie.com/media/files/yolov3.weights)
