import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np 
import argparse
from main_process import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model_setting')
    parser.add_argument('--model', default='model1', help='choose model: model1, model2, model3')
    parser.add_argument('--video', help='Specify a video path')
    parser.add_argument('--filter', default = None, help='specify a filtering method if required: kf, mavg ')
    args = parser.parse_args()
    model_idxlist = {'model1':0,'model2':1,'model3':2}
    model_list = ['model1', 'model2', 'model3']    
    model_max = [[22],
                    [7],
                    [8]]
    model_choose = [model_idxlist[args.model] ]
    for di in model_choose:
        opt = dict()
        opt['model'] = model_list[di]
        opt['max_list'] = model_max[di]
        opt['skip_frames'] =  30
        opt['start_webcam'] = False
        opt['read_ipstream'] = None
        opt['filter'] = args.filter

        if args.video==None:
            opt['start_webcam'] = True

        # step1: Create root path for dataset
        elif args.video.startswith('http'):
            opt['read_ipstream'] = True
            opt['video'] = args.video

        else:
            opt['read_ipstream'] = False
            opt['video'] = os.path.join('videos', args.video )

           
        opt['num_workers'] = 0
        opt['IF_savemem_train'] = False
        opt['IF_savemem_test'] = False
        # -- test setting
        opt['test_batch_size'] = 1
        # --Network settinng    
        opt['psize'],opt['pstride'] = 64,64
        opt['div_times'] = 2
        # -- parse class to count setting
        parse_method_dict = {0:'maxp'}
        opt['parse_method'] = parse_method_dict[0]
        #step2: set the max number and partition method
        opt['max_num'] = opt['max_list'][0]
        opt['partition'] = 'two_linear'
        opt['step'] = 0.5
        # here create model path
        opt['model_path'] = os.path.join('model',args.model)

        main(opt)