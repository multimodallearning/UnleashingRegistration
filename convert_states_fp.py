
#!/usr/bin/env python
# coding: utf-8

import argparse
import nibabel as nib
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange,tqdm
from monai.networks.nets import AutoEncoder
from unleashing_utils import VectorQuantizer,Transformer,synthesise_fn, optim_warmup,dice_coeff,weighted_elbo_loss

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

def main(filename):
    states = torch.load(filename)

    transformer = True if('sampler' in filename) else False
    if(transformer):
        denoise_fn = Transformer().cuda()

        state_list0 = list(states)[0]
        if('_orig_mod' in state_list0):
            denoise_fn = torch.compile(denoise_fn)
        half = False if (states[state_list0].dtype==torch.float32) else True
        if(half): #convert to the other
            denoise_fn.to(torch.float32)
            torch.save(denoise_fn.state_dict(),filename)
        else:
            denoise_fn.to(torch.float16)
            torch.save(denoise_fn.state_dict(),filename)
        print('done converting sampler states')

    else: #AE (two state_dicts but only AE is relevant)
    #              torch.save([net.state_dict(),quantise.embedding.state_dict()],'unleashing_3d_vqae_paired_states.pth')

        net = AutoEncoder(spatial_dims=3,in_channels=2,out_channels=4,channels=(48,48,64,64,128,192,256,384),strides=(1,2,1,2,1,2,1,2)).cuda()
        states1 = states[0]
        state_list0 = list(states1)[0]
        if('_orig_mod' in state_list0):
            net = torch.compile(net)
        half = False if (states1[state_list0].dtype==torch.float32) else True
        if(half): #convert to the other
            net.to(torch.float32)
            torch.save([net.state_dict(),states[1]],filename)
        else:
            net.to(torch.float16)
            torch.save([net.state_dict(),states[1]],filename)
        print('done converting AE states')


            
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train_sampler args')
    parser.add_argument('filename', help='e.g. unleashing_3d_diffusion_sampler_states.pth')
    args = parser.parse_args()
    main(args.filename)
