import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader
import torch.optim as optim
import cv2
from PIL import Image
import os
import numpy as np
from time import time, sleep
import math
import pandas as pd
import csv
from load_data import Countmap_Dataset
from Network.SSDCNet import SSDCNet_classify
from Val import test_phase
import queue, threading


def test(frame, opt, rgb, transform_test, num_workers, label_indice, model_path):
    img = Image.fromarray(frame)
    testset = Countmap_Dataset(img,rgb,transform=transform_test,\
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
        test_log, count = test_phase(opt,net,testloader,log_save_path=log_save_path)
        return count
    

def main(opt):
    # path to model
    model_path = opt['model_path'] 
    # =============================================================================
    # inital setting
    # =============================================================================
    # Initial setting
    read_ipstream = opt['read_ipstream']
    num_workers = opt['num_workers']
    transform_test = []
    
    # set label_indice
    label_indice = np.arange(opt['step'],opt['max_num']+opt['step'],opt['step'])
    add = np.array([1e-6,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]) 
    label_indice = np.concatenate( (add,label_indice) )
    opt['label_indice'] = label_indice
    opt['class_num'] = label_indice.size+1
    skip_frames = opt['skip_frames']
    
    start = time()
    if not opt['start_webcam']:
        video = opt['video']
        vidcap = cv2.VideoCapture(video)
        if not read_ipstream:
            print('[INFO]Loading video from file...')
        else:
            print('[INFO]Loading from the given URL...')
    else:
        print("[INFO] starting video stream...")
        vidcap = cv2.VideoCapture(0)

    total_frames = 1
    rgb = np.zeros(3)
    while True:
        frame = vidcap.read()[1]
        
        if frame is not None:
            color = cv2.mean(frame)
            rgb += np.array([color[2], color[1], color[0]])
            if(total_frames % 1 == 0):
                rgb = rgb/(skip_frames * 256)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                (H, W) = frame.shape[:2]
                count = test(frame, opt, rgb, transform_test, num_workers, label_indice, model_path)
                cv2.putText(frame, 'No.of People: %.0f' % count, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow('frame ',frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                rgb = np.zeros(3)
            total_frames += 1

        else:
            print("[INFO]End of Video feed or Error in streaming")
            print("[INFO]Exiting...")
            vidcap.release()
            break
    end = time()
    print(end - start)
    cv2.destroyAllWindows()