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
1. **Model:** There are three pretrained models available [here](https://drive.google.com/drive/folders/1i7oVrxz8w4m7t0zQI7-qtv2__M0OSVp3?usp=sharing). They were trained by the authors of the the [paper](https://arxiv.org/abs/2001.01886) on Shanghai Dataset(A, B) and QRNF dataset.
2. **Video:** It can be a video file stored in the videos folder or a URL to the video.

### Quick Start

##### 1. Clone the repository:  
`git clone https://github.com/niveditarufus/PeopleCounter-SSDCNet.git`  
##### 2. Run:  
`cd PeopleCounter-SSDCNet`
##### 3. Install all dependencies required, run:  
`pip3 install -r requirements.txt`  
##### 4. Run Demo:      
**Usage:** python3 Run.py [--model MODEL] [--video VIDEO FILE/URL]
###### Example:  
`python3 Run.py --model model1 --video m1.mp4`

### References
[SS-DCNet](https://arxiv.org/abs/2001.01886)  
[code](https://github.com/xhp-hust-2018-2011/SS-DCNet)
