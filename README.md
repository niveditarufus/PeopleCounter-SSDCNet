# PeopleCounter with SS-DCNet

## Introduction
This is a people counter module  that is implemented using SS-DCNet. This module is equipped to to give the count of people present in a frame captured from a video.

## Dependencies
Python==3.6  
torch==1.4.0  
torchvision==0.5.0  
opencv==4.2.0.34  
Pillow==7.1.1  
numpy==1.18.2  
scipy==1.4.1  

### Inputs
1. **Model:** There are three pretrained models available [here](https://drive.google.com/drive/folders/1i7oVrxz8w4m7t0zQI7-qtv2__M0OSVp3?usp=sharing). They were trained by the authors of the the [paper](https://arxiv.org/abs/2001.01886) on Shanghai Dataset(A, B) and QRNF dataset.[Note]: download the models to a folder called `model` in the repository.
2. **Video:** It can be a video file stored in the videos folder or a URL to the video.

### Quick Start

##### 1. Clone the repository:  
`git clone https://github.com/niveditarufus/PeopleCounter-SSDCNet.git`  
##### 2. Run:  
`cd PeopleCounter-SSDCNet`
##### 3. Install all dependencies required, run:  
`pip3 install -r requirements.txt`  
##### 4. Run Demo:      
**Usage:**   
python3 Run.py [--model MODEL] [--video LIST OF VIDEO FILES/URL] [--filter METHOD]      
###### Example:  
`python3 Run.py --model model3 --video m1.mp4 --filter kf --skip_frames 30`  
if a video(file/URL) was not supplied, a reference to the webcam will be grabbed.  

You can also supply the a list of videos which have overlapping views. This might cause some delay in relaying the count of people, so change the `skip_frames` parameter accordingly. This employs an [image stitching](http://matthewalunbrown.com/papers/ijcv2007.pdf) on the images before returning the value of the count of people.
###### Example:  
`python3 Run.py --model model3 --video f1.mp4 f2.mp4 --filter kf --skip_frames 30`  

If the videos are just from different perspectives(can't be stitched) and not really overlapping you will have to set the `stitch` parameter to `False` while parsing. This employs Boyer-Moore's Majority Voting algorithm [(Link)](https://www.cs.utexas.edu/~moore/best-ideas/mjrty/) for the count of people.  
###### Example:  
`python3 Run.py --model model3 --video f1.mp4 f2.mp4 --filter kf --skip_frames 30 --stitch False`  

### References
1. [SS-DCNet](https://arxiv.org/abs/2001.01886)  
2. [SS-DCNET code](https://github.com/xhp-hust-2018-2011/SS-DCNet) 
3. [Image stitching](http://matthewalunbrown.com/papers/ijcv2007.pdf)
4. [Boyer-Moore's Majority Voting algorithm](https://www.cs.utexas.edu/~moore/best-ideas/mjrty/)
