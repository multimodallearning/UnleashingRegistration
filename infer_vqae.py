
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

def main(chkpt_vqae):

    
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
    
    #reload vqae
    states = torch.load(chkpt_vqae)
    net = AutoEncoder(spatial_dims=3,in_channels=2,out_channels=4,channels=(48,48,64,64,128,192,256,384),strides=(1, 2, 1, 2, 1, 2,1,2)).cuda()
    net = torch.compile(net)
    net.load_state_dict(states[0])
    quantise = VectorQuantizer(emb_dim=384).cuda()
    quantise.embedding.load_state_dict(states[1])

    #inference and quantisation
    quants = []
    val_dice = torch.zeros(len(vessel_fix))
    for i in trange(len(vessel_fix)):
        with torch.no_grad():
            #pad crop to 256x208x256
            H,W,D = vessel_fix[i].shape[-3:]
            h1,w1,d1 = (22*16-H)//2-48,(16*16-W)//2-32,(22*16-D)//2-48
            h2,w2,d2 = 22*16-H-h1-96, 16*16-W-w1-48, 22*16-D-d1-96
            batch_f = F.pad(vessel_fix[i],(d1,d2,w1,w2,h1,h2))
            batch_m = F.pad(vessel_mov[i],(d1,d2,w1,w2,h1,h2))

            batch = torch.stack((batch_f,batch_m),0).cuda().unsqueeze(0)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                z = net.encode(batch)
                z_q,loss_codebook,d_argmin = quantise(z)
                quants.append(d_argmin)
                output = net.decode(z_q)
            val_dice[i] = dice_coeff(output[0,:2].argmax(0).contiguous(),batch[0,0].long().contiguous(),2)
        if(i%25==20):
            output1 = output[:,:2].argmax(1).float()#.unsqueeze(1)
            output2 = output[:,2:4].argmax(1).float()#
            f,ax = plt.subplots(1,2,figsize=(12,12))
            ax[0].imshow(batch[0,0,:H//2].mean(0).cpu().data.t().flip(0),'Blues')
            ax[0].imshow(batch[0,1,:H//2].mean(0).cpu().data.t().flip(0),'Oranges',alpha=.5); ax[0].axis('off')
            ax[1].imshow(output1[0,:H//2].mean(0).cpu().data.t().flip(0),'Blues')
            ax[1].imshow(output2[0,:H//2].mean(0).cpu().data.t().flip(0),'Oranges',alpha=.5); ax[1].axis('off')
            plt.savefig('visual_results/vqae_lung250_'+files[i][:-9]+'.png')
            plt.show()

    print('validation Dice',val_dice.mean())
    quants_s = [q.short() for q in quants]
    torch.save(quants_s,'unleashing_3d_vqae_paired_quants.pth')

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'infer_vqae args')
    parser.add_argument('chkpt_vqae', help='file from VQAE training default unleashing_3d_vqae_paired_states.pth')
    args = parser.parse_args()
    main(args.chkpt_vqae)
