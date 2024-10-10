
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

def main(batch_size,chkpt_quants,chkpt_gen):

    
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
    
    #reload vqae / only quantiser and generator needed for visualisation
    visualise = False
    if(os.path.exists(chkpt_gen)):
        states = torch.load(chkpt_gen)
        net = AutoEncoder(spatial_dims=3,in_channels=2,out_channels=4,channels=(48,48,64,64,128,192,256,384),strides=(1, 2, 1, 2, 1, 2,1,2)).cuda()
        net = torch.compile(net)
        net.load_state_dict(states[0])
        quantise = VectorQuantizer(emb_dim=384).cuda()
        quantise.embedding.load_state_dict(states[1])
        visualise = True
    
    quants_s = torch.load(chkpt_quants)

    lr_base = 1e-4
    warmup_iters = 500
    num_timesteps = 256
    mask_id = 1024
    
    denoise_fn = torch.compile(Transformer().cuda())
    
    num_iterations = 15000*8//batch_size
    batch_q = torch.zeros(batch_size,16,13,16).long().cuda()
    t0 = time.time()

    optimizer = torch.optim.Adam(denoise_fn.parameters(),lr=lr_base)
    scaler = torch.cuda.amp.GradScaler()
    run_loss = torch.zeros(num_iterations)
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        for i in range(num_iterations):
            denoise_fn.train()

            if i <= warmup_iters:
                optim_warmup(lr_base, i, optimizer, warmup_iters)


            optimizer.zero_grad()
            
            idx = torch.randperm(len(quants_s))[:batch_size]
            with torch.no_grad():
                for j in range(batch_size):
                    batch_q[j] = quants_s[idx[j]].view(-1,16,13,16)

            x_0 = batch_q.view(batch_size,-1)#quants_all[idx].view(len(idx),-1).cuda()
            # sample random time point for each batch element
            t = torch.randint(1, num_timesteps+1, (x_0.shape[0],)).to(x_0).float() / num_timesteps
            # create and apply random mask
            mask = torch.rand_like(x_0.float()) < (t.float()).unsqueeze(-1)
            # replace masked tokens with the undefined ID (1024)
            x_t = torch.where(mask,mask_id,x_0.clone())
            # ground-truth with unchangable tokens set to ignore
            target = torch.where(mask,x_0.clone(),-1)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # perform one denoising step
                x_logits = denoise_fn(x_t).permute(0,2,1)
                loss = weighted_elbo_loss(x_logits,target,t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            run_loss[i] = 100*loss.item()
            
            str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-28:i-1].mean())}, runtime: {'%0.3f'%(time.time()-t0)} sec, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
            pbar.set_description(str1)
            pbar.update(1)

            #visualisation
            if(i%1500==199):
                if(visualise):
                    denoise_fn.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            latents = synthesise_fn(denoise_fn, temp=.9)

                            q = F.one_hot(latents.view(-1),1024).float().mm(quantise.embedding.weight).reshape(-1,16,13,16,384).permute(0,4,1,2,3)
                            images = net.decode(q.float())
                    f,ax = plt.subplots(1,2,figsize=(14,6))

                    ax[0].imshow(images[0,:2].argmax(0).float().mean(0).data.t().flip(0).cpu(),'Blues'); ax[0].axis('off')
                    ax[0].imshow(images[0,2:4].argmax(0).float().mean(0).data.t().flip(0).cpu(),'Oranges',alpha=.5);
                    ax[1].imshow(images[1,:2].argmax(0).float().mean(0).data.t().flip(0).cpu(),'Blues'); ax[1].axis('off')
                    ax[1].imshow(images[1,2:4].argmax(0).float().mean(0).data.t().flip(0).cpu(),'Oranges',alpha=.5);
            
                    plt.savefig('visual_results/sampler_lung250_'+str(i+1)+'.png')
                    plt.show()
                    plt.close()
                torch.save(denoise_fn.state_dict(),'unleashing_3d_diffusion_sampler_states.pth')

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train_sampler args')
    parser.add_argument('batch_size', help='our default 8')
    parser.add_argument('chkpt_quants', help='file from VQAE inference default unleashing_3d_vqae_paired_quants.pth')
    parser.add_argument('chkpt_gen', help='file from VQAE trainer default unleashing_3d_vqae_paired_states.pth')
    args = parser.parse_args()
    main(int(args.batch_size),args.chkpt_quants,args.chkpt_gen)
