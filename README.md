# Object detection using YOLO v3 
You Only Look Once (YOLO) (https://arxiv.org/abs/1506.02640) is a state of the art object detection algorithm. This repository serves as implementation of YOLOv3 (https://arxiv.org/abs/1804.02767) using PyTorch. We also integrated the object detection workflow with Face Recognition using FaceNet (https://arxiv.org/abs/1503.03832) that we'll talk about soon. 

### Network Design
The network has 53 convolutional layers and is also called Darknet-53. Apart from *convolutional*, there are *shortcut (skip connections)*, *upsample* and *route* layers in the network. 

### Output
We ran our implementation on images and videos. Here are some sample outputs:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=WlvFTBsUcKk
" target="_blank"><img src="http://img.youtube.com/vi/WlvFTBsUcKk/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

Please download the weights file for YOLO v3 from [here](https://pjreddie.com/media/files/yolov3.weights)
