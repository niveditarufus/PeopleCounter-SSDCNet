# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import  DataLoader
import torch.nn.functional as F
import torch

import os
import numpy as np
from time import time
import math
import math
from Network.class_func import get_local_count


def test_phase(opt,net,testloader,log_save_path=None):
    with torch.no_grad():
        net.eval()
        start = time()

        for data in testloader:
            inputs = data['image']
            inputs = inputs.type(torch.float32)
            inputs = inputs.cuda()

            # process with SSDCNet
            features = net(inputs)
            div_res = net.resample(features)
            merge_res = net.parse_merge(div_res)
            outputs = merge_res['div'+str(net.div_times)]
            del merge_res

            pre =  (outputs).sum()
            end = time()
            running_frame_rate = opt['test_batch_size'] * float( 1 / (end - start))
        
    im_num = len(testloader)
    test_dict=dict()
    
    return test_dict, pre.cpu().item()





    






 



