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
from load_data_V2 import Countmap_Dataset
from Network.SSDCNet import SSDCNet_classify
from Val import test_phase


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
    skip_frames = opt['skip_frames']

    if not read_ipstream:
        print('Loading video from file...')
    else:
        print('Loading from the given URL...')

    success = True
    total_frames = 0
    start = time()
    while success:
        success,frame = vidcap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if(total_frames % skip_frames == 0):
                test(frame, opt, rgb_dir, transform_test, num_workers, label_indice, model_path)
                cv2.imshow('frames captured ',frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                
            total_frames += 1
        else:
            print("End of Video feed or Error in streaming")
            break

    end = time()
    print(end - start)   
    vidcap.release()
    cv2.destroyAllWindows()