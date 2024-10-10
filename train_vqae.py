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
import numpy as np
from tqdm.auto import trange,tqdm
from monai.networks.nets import AutoEncoder
from unleashing_utils import VectorQuantizer,optim_warmup,dice_coeff

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

def main(batch_size):

    
    #load vessels for fixed and moving (training scans)
    folder = 'dilated_vessels/'
    files = sorted(os.listdir(folder))
    cases = torch.tensor([int(f.split('_')[1]) for f in files]).unique()

    vessel_fix = []; vessel_mov = [];
    for i in trange(len(cases)):
        vessel_fix.append(torch.from_numpy(nib.load(folder+'case_'+str(int(cases[i])).zfill(3)+'_1.nii.gz').get_fdata()>0).float().contiguous())
        vessel_mov.append(torch.from_numpy(nib.load(folder+'case_'+str(int(cases[i])).zfill(3)+'_2.nii.gz').get_fdata()>0).float().contiguous())
    #for some reason case-id 33 is much too big
    vessel_fix[33] = vessel_fix[33][:,:,:280]
    vessel_mov[33] = vessel_mov[33][:,:,:280]

    #create model
    net = AutoEncoder(spatial_dims=3,in_channels=2,out_channels=4,channels=(48,48,64,64,128,192,256,384),\
            strides=(1, 2, 1, 2, 1, 2,1,2)).cuda()
    net = torch.compile(net)
    quantise = VectorQuantizer(emb_dim=384).cuda()

    #our default settings
    lr_base = 0.0005
    warmup_iters = 1500
    #batch_size = 8 #set by args
    optimizer = torch.optim.Adam(list(net.parameters())+list(quantise.parameters()),lr=lr_base)
    num_iterations = 336000//batch_size
    run_loss = torch.zeros(num_iterations)
    run_dice = torch.zeros(num_iterations)
    run_codeloss = torch.zeros(num_iterations)
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,num_iterations//7,2)
    scaler = torch.cuda.amp.GradScaler()
    batch = torch.randn(batch_size,2,128,128,128).cuda()
    weight = torch.tensor([1,2.5]).cuda()
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        for i in range(num_iterations):
        
            optimizer.zero_grad()
            idx = torch.randperm(len(vessel_fix))[:batch_size]
            #create two-channel batch with fixed and moving vessels
            for j in range(batch_size):
                H,W,D = vessel_fix[idx[j]].shape[-3:]
                h1,w1,d1 = int(torch.randint(H-128,(1,))),int(torch.randint(W-128,(1,))),int(torch.randint(D-128,(1,)))
                batch[j,0] = vessel_fix[idx[j]][h1:h1+128,w1:w1+128,d1:d1+128].cuda()
                batch[j,1] = vessel_mov[idx[j]][h1:h1+128,w1:w1+128,d1:d1+128].cuda()
            #affine augmentation
            with torch.no_grad():
                A = F.affine_grid((torch.eye(3,4).unsqueeze(0)+torch.randn(batch_size,3,4)*.05).cuda(),(batch_size,1,128,128,128),align_corners=False)
                batch = F.grid_sample(batch,A,mode='nearest',align_corners=False)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                #forward
                z = net.encode(batch)
                z_q,loss_codebook,_ = quantise(z)
                output = net.decode(z_q)
                #dual CE loss for fixed and moving vessels separately
                loss = .5*nn.CrossEntropyLoss(weight)(output[:,:2],batch[:,0].long())+\
                            .5*nn.CrossEntropyLoss(weight)(output[:,2:],batch[:,1].long())
            run_loss[i] = loss.item()
            run_codeloss[i] = loss_codebook.item()
            run_dice[i] = .5*dice_coeff(output[:,:2].argmax(1).contiguous(),batch[:,0].long().contiguous(),2)+\
                            .5*dice_coeff(output[:,2:].argmax(1).contiguous(),batch[:,1].long().contiguous(),2)
            if(i%50==49):
                np.savetxt('unleashing_3d_vqae_paired_log.txt',torch.stack((run_loss[:i],run_codeloss[:i],run_dice[:i]),1).numpy())

            scaler.scale((loss+loss_codebook*.1)).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-28:i-1].mean())}, Dice: {'%0.3f'%(run_dice[i-28:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t0)} sec, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
            pbar.set_description(str1)
            pbar.update(1)
            if(i%3500==3499):
                torch.save([net.state_dict(),quantise.embedding.state_dict()],'unleashing_3d_vqae_paired_states.pth')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train_vqae args')
    parser.add_argument('batch_size', help='our default 8')
    args = parser.parse_args()
    main(int(args.batch_size))
