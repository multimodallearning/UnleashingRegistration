
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

def main(total_samples,chkpt_sampler,chkpt_gen):

    
    #reload vqae / quantiser and generator
    states = torch.load(chkpt_gen)
    net = AutoEncoder(spatial_dims=3,in_channels=2,out_channels=4,channels=(48,48,64,64,128,192,256,384),strides=(1, 2, 1, 2, 1, 2,1,2)).cuda()
    net = torch.compile(net)
    net.load_state_dict(states[0])
    quantise = VectorQuantizer(emb_dim=384).cuda()
    quantise.embedding.load_state_dict(states[1])
    states_sampler = torch.load(chkpt_sampler)
    denoise_fn = Transformer().cuda()
    compiled_states = '_orig_mod' in list(states_sampler.keys())[0]
    if(compiled_states):
        denoise_fn = torch.compile(denoise_fn)
        denoise_fn.load_state_dict(states_sampler)
    else:
        denoise_fn.load_state_dict(states_sampler)
        denoise_fn = torch.compile(denoise_fn)

    for i in range(total_samples//2):
        denoise_fn.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                latents = synthesise_fn(denoise_fn, temp=.9)

                q = F.one_hot(latents.view(-1),1024).float().mm(quantise.embedding.weight).reshape(-1,16,13,16,384).permute(0,4,1,2,3)
                images = net.decode(q.float())
        if(i%20==1):
            f,ax = plt.subplots(1,2,figsize=(14,6))
            ax[0].imshow(images[0,:2].argmax(0).float().mean(0).data.cpu(),'Blues'); ax[0].axis('off')
            ax[0].imshow(images[0,2:4].argmax(0).float().mean(0).data.cpu(),'Oranges',alpha=.5);
            ax[1].imshow(images[1,:2].argmax(0).float().mean(0).data.cpu(),'Blues'); ax[1].axis('off')
            ax[1].imshow(images[1,2:4].argmax(0).float().mean(0).data.cpu(),'Oranges',alpha=.5);
            plt.show()
            plt.savefig('visual_results/sampler_lung250_'+str(i).zfill(3)+'.png')
            plt.close()

        for j in range(2):
            recon_f = torch.softmax(images[j,:2].float(),0).cpu()
            recon_m = torch.softmax(images[j,2:4].float(),0).cpu()
            
            np.savez_compressed('synth/case'+str(i*2+j+1).zfill(4)+'.npz',\
                            fix=recon_f.half().numpy(),mov=recon_m.half().numpy())
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'infer_sampler args')
    parser.add_argument('total_samples', help='our default 256')
    parser.add_argument('chkpt_sampler', help='file from VQAE inference default unleashing_3d_diffusion_sampler_states.pth')
    parser.add_argument('chkpt_gen', help='file from VQAE trainer default unleashing_3d_vqae_paired_states.pth')
    args = parser.parse_args()
    main(int(args.total_samples),args.chkpt_sampler,args.chkpt_gen)
