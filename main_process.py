import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader
import torch.optim as optim
import cv2
from PIL import Image
import os
import numpy as np
from time import time
import math
import pandas as pd
import csv
import cv2
import requests
import imutils
from IOtools import txt_write,get_config_str
from load_data_V2 import Countmap_Dataset
from Network.SSDCNet import SSDCNet_classify
from Val import test_phase

# class ReadIPStream:
#     def __init__(self, url):
#         self.stream = requests.get(url, stream=True)

#     def read_stream(self):
#         msg = bytes('', encoding = 'UTF-8')
#         for chunk in self.stream.iter_content(chunk_size=1024):
#             msg += chunk
#             a = msg.find(b'\xff\xd8')
#             b = msg.find(b'\xff\xd9')
#             if a != -1 and b != -1:
#                 jpg = msg[a:b + 2]
#                 msg = msg[b + 2:]
#                 if len(jpg) > 0:
#                     img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
#                     img = imutils.resize(img, width=500)
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#                     return True, img

def getFrame(vidcap, sec = 0, frameRate = 1):
    
    sec = round(sec, 2)
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return hasFrames, image, sec

def test(frame, opt, rgb_dir, transform_test, num_workers, label_indice, model_path):
    img = Image.fromarray(frame)
    testset = Countmap_Dataset(img,rgb_dir,transform=transform_test,\
    if_test=True, IF_loadmem=opt['IF_savemem_test'])
    testloader = DataLoader(testset, batch_size=opt['test_batch_size'],
                        shuffle=False, num_workers=num_workers)
    # init networks
    label_indice = torch.Tensor(label_indice)
    class_num = len(label_indice)+1
    div_times = opt['div_times']
    net = SSDCNet_classify(class_num,label_indice,div_times=div_times,\
            frontend_name='VGG16',block_num=5,\
            IF_pre_bn=False,IF_freeze_bn=False,load_weights=True,\
            psize=opt['psize'],pstride = opt['pstride'],parse_method ='maxp').cuda()
    # test the min epoch
    mod_path='best_epoch.pth' 
    mod_path=os.path.join(opt['model_path'] ,mod_path)
    if os.path.exists(mod_path):
        all_state_dict = torch.load(mod_path)
        net.load_state_dict(all_state_dict['net_state_dict'])
        tmp_epoch_num = all_state_dict['tmp_epoch_num']
        log_save_path = os.path.join(model_path,'log-epoch-min[%d]-%s.txt'%(tmp_epoch_num+1,opt['parse_method']) )
        # test
        test_log = test_phase(opt,net,testloader,log_save_path=log_save_path)
    cv2.imshow('frames captured ',frame)
    

def main(opt):
    # path to model
    model_path = opt['model_path'] 
    # =============================================================================
    # inital setting
    # =============================================================================
    # Initial setting
    read_ipstream = opt['read_ipstream']
    video = opt['video']
    num_workers = opt['num_workers']
    transform_test = []
    
    # set label_indice
    label_indice = np.arange(opt['step'],opt['max_num']+opt['step'],opt['step'])
    add = np.array([1e-6,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]) 
    label_indice = np.concatenate( (add,label_indice) )
    opt['label_indice'] = label_indice
    opt['class_num'] = label_indice.size+1
    
    #test settings
    rgb_dir = os.path.join(model_path,'rgbstate.mat')
    vidcap = cv2.VideoCapture(video)
    if not read_ipstream:
        sec = 0
        frameRate = 1
        print('Loading video from file...')
    else:
        skip_frames = opt['skip_frames']
        print('Loading from the given URL...')
    success = True
    total_frames = 0

    while success:
        if not read_ipstream:
            success, frame = getFrame(vidcap, sec)
            sec = sec + frameRate
            if success:
                test(frame, opt, rgb_dir, transform_test, num_workers, label_indice, model_path)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        else:
            success,frame = vidcap.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if(total_frames % skip_frames == 0):
                    test(frame, opt, rgb_dir, transform_test, num_workers, label_indice, model_path)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

                total_frames += 1
            else:
                print("End of Video feed or Error in streaming")
                break

        
    vidcap.release()
    cv2.destroyAllWindows()