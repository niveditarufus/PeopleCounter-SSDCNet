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
import matplotlib.pyplot as plt
import imutils


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
        # count = torch.round(count)
        return count

def KalmanFilter(count_k_1, P_k_1, count, R=0.1):
    count_prior_k = count_k_1
    P_prior_k = P_k_1
    K_k = P_prior_k / (P_prior_k + R)
    count_k = count_prior_k + K_k * (count - count_prior_k)
    P_k = (1 - K_k) * P_prior_k
    return count_k, P_k

def Moving_avg(count, c_queue, window=20):
    if(len(c_queue)<=window):
        count = mean(c_queue)
    else:
        c_queue.pop(0)
        count = mean(c_queue)
    return round(count)

def majorityElement(count_list):
    x = np.unique(np.round(count_list))
    if(len(x)==len(count_list) and np.count_nonzero(x)==len(count_list)):
        return np.round(np.mean(x))
    m = -1
    i = 0
    ind = -1
    for j in range(len(count_list)):
        if i == 0:
            m = count_list[j]
            i = 1
            ind = j
        elif m == count_list[j]:
            i = i + 1
        else:
            i = i - 1
    return m
    
def mean(nums):
    return float(sum(nums)) / max(len(nums), 1)

def main(opt):
    # path to model
    model_path = opt['model_path'] 
    # =============================================================================
    # inital setting
    # =============================================================================
    # Initial setting
    # read_ipstream = opt['read_ipstream']
    num_workers = opt['num_workers']
    transform_test = []
    stitch = opt['stitch']
    filter_method = opt['filter']
    
    # set label_indice
    label_indice = np.arange(opt['step'],opt['max_num']+opt['step'],opt['step'])
    add = np.array([1e-6,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]) 
    label_indice = np.concatenate( (add,label_indice) )
    opt['label_indice'] = label_indice
    opt['class_num'] = label_indice.size+1
    skip_frames = opt['skip_frames']
    shape = (1024, 768)
    writer = None
    exit_flag = True

    ## Uncomment these lines if you want to save the video to your system
    # fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    # writer = cv2.VideoWriter('output.avi',fourcc, 2, shape)
        
    start = time()
    vidcap = []
    if not opt['start_webcam']:
        
        for read_ipstream, video in zip(opt['read_ipstream'], opt['video']):
            vidcap.append(cv2.VideoCapture(video))
            if not read_ipstream:
                print('[INFO]Loading video from file...')
            else:
                print('[INFO]Loading from the given URL...')
    else:
        print("[INFO] starting video stream...")
        vidcap.append(cv2.VideoCapture(0))

    if(stitch):
        total_frames = 1
        rgb = np.zeros(3)
        start_flag = True
        num_views = len(vidcap)

        while exit_flag:
            for i in range(num_views):
                image = vidcap[i].read()[1]
                if image is not None:
                    color = cv2.mean(image)
                    rgb += np.array([color[2], color[1], color[0]])
            if(total_frames % skip_frames == 0):
                if(num_views==1):
                    frame = vidcap[0].read()[1]
                else:
                    frames = []
                    for i in range(num_views):
                        image = vidcap[i].read()[1]
                        if image is not None:
                            frames.append(image)
                    if(len(frames)==num_views):
                        if imutils.is_cv3() :
                            stitcher = cv2.createStitcher() 
                        else:
                            stitcher = cv2.Stitcher_create()
                        (status, frame) = stitcher.stitch(frames)
                if frame is not None:
                    rgb = rgb/(skip_frames * 256 * num_views)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, shape)
                    count = test(frame, opt, rgb, transform_test, num_workers, label_indice, model_path)
                    if filter_method=='kf':
                        if not start_flag:
                            count, P_k = KalmanFilter(count_k_1, P_k_1, count, R)
                            count_k_1, P_k_1 = count, P_k
                        else:
                            count_k_1 = count
                            P_k_1 = 1
                            R = 0.1
                            start_flag = False
                    if filter_method=='mavg':
                        if not start_flag:
                            c_queue.append(count)
                            count = Moving_avg(count, c_queue)
                        else:
                            c_queue = []
                            c_queue.append(count)
                            start_flag = False
                    cv2.putText(frame, 'No.of People: '+str(round(count)), (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow('frame ',frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        exit_flag=False
                        break
                    if writer is not None:
                        writer.write(frame)
                    rgb = np.zeros(3)
                else:
                    exit_flag = False
                    print("[INFO]End of Video feed or Error in streaming")
                    print("[INFO]Exiting...")
                    for vc in vidcap:
                        vc.release()
            total_frames += 1

    else:
        total_frames = 1
        rgb = np.zeros(3)
        num_views = len(vidcap)
        start_flag = np.ones((num_views), dtype=bool)
        count = np.zeros(num_views)
        if filter_method=='kf':
            count_k_1 = np.zeros(num_views)
            P_k_1 = np.ones(num_views)
            P_k = np.zeros(num_views)
        elif filter_method=='mavg':
            c_queue = np.empty((num_views,),dtype=object)

        while exit_flag:
            for i in range(num_views):
                image = vidcap[i].read()[1]
                if image is not None:
                    color = cv2.mean(image)
                    rgb += np.array([color[2], color[1], color[0]])
            if(total_frames % skip_frames == 0):
                for i in range(num_views):
                    frame = vidcap[i].read()[1]
                    if frame is not None:
                        rgb = rgb/(skip_frames * 256 * num_views)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, shape)
                        if(i==0):
                            img = frame
                        count[i] = test(frame, opt, rgb, transform_test, num_workers, label_indice, model_path)
                        if filter_method=='kf':
                            if not start_flag[i]:
                                count[i], P_k[i] = KalmanFilter(count_k_1[i], P_k_1[i], count[i], R)
                                count_k_1[i], P_k_1[i] = count[i], P_k[i]
                            else:
                                count_k_1[i]=count[i]
                                P_k_1[i] = 1
                                R = 0.1
                                start_flag[i] = False
                        elif filter_method=='mavg':
                            if not start_flag[i]:
                                c_queue[i].append(count[i])
                                count[i] = Moving_avg(count[i], c_queue[i])
                            else:
                                c_queue[i] =[]
                                c_queue[i].append(count[i])
                                start_flag[i] = False
                        final_count = majorityElement(count)
                        
                        cv2.putText(img, 'No.of People: '+str(round(final_count)), (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.imshow('frame ',img)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            exit_flag=False
                            break
                        if writer is not None:
                            writer.write(frame) 
                    else:
                        exit_flag = False
                        print("[INFO]End of Video feed or Error in streaming")
                        print("[INFO]Exiting...")
                        for vc in vidcap:
                            vc.release()
                        break
                rgb = np.zeros(3)
            total_frames += 1

    end = time()
    print(end - start)
    cv2.destroyAllWindows()
