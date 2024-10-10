
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
from monai.networks.nets.unet import UNet
#from monai.networks.nets import AutoEncoder
from unleashing_utils import warp_sym_step,disp_square,compose

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

def main(batch_size,data_folder):
    
    #load vessels for fixed and moving (training scans)
    folder = data_folder #'dilated_vessels/'
    files = sorted(os.listdir(folder))
    #check whether data are raw nii.gz files or synthetic npz (latter requires no
    niigz = np.array(['nii.gz' in f for f in files]).sum()>np.array(['npz' in f for f in files]).sum()

    vessel_fix = []; vessel_mov = [];

    if(niigz):
        cases = torch.tensor([int(f.split('_')[1]) for f in files]).unique()

        for i in trange(len(cases)):
            vessel_fix.append(torch.from_numpy(nib.load(folder+'/case_'+str(int(cases[i])).zfill(3)+'_1.nii.gz').get_fdata()>0).float().contiguous())
            vessel_mov.append(torch.from_numpy(nib.load(folder+'/case_'+str(int(cases[i])).zfill(3)+'_2.nii.gz').get_fdata()>0).float().contiguous())
            if(int(cases[i]) == 35): #for some reason id 33 (that is case_035) is much too big
                vessel_fix[i] = vessel_fix[i][:,:,:280]
                vessel_mov[i] = vessel_mov[i][:,:,:280]
        
        #pad crop to 256x208x256
        for i in trange(len(vessel_fix)):
            with torch.no_grad():
                H,W,D = vessel_fix[i].shape[-3:]
                h1,w1,d1 = (22*16-H)//2-48,(16*16-W)//2-32,(22*16-D)//2-48
                h2,w2,d2 = 22*16-H-h1-96, 16*16-W-w1-48, 22*16-D-d1-96
                vessel_fix[i] = F.pad(vessel_fix[i],(d1,d2,w1,w2,h1,h2))
                vessel_mov[i] = F.pad(vessel_mov[i],(d1,d2,w1,w2,h1,h2))
    else:
        for i in trange(min(256,len(files))):#limit to 256 pairs (as in paper)
            with torch.no_grad():
                data = np.load(folder+'/case'+str(i+1).zfill(4)+'.npz')
                vessel_fix.append(torch.from_numpy(data['fix'])[1])
                vessel_mov.append(torch.from_numpy(data['mov'])[1])
    vessel_fix = torch.stack(vessel_fix)
    vessel_mov = torch.stack(vessel_mov)
    print('done',vessel_fix.shape)
    print('done',vessel_mov.shape)

    ##setting up registration training routine
    sym = True #always true
    #initialise UNets and optimisers
    unet1_ = []; unets = [];
    channels = (8,16,32,64,64,64)
    strides = (2,2,1,2,1)
    for i in range(2):
        unet = UNet(spatial_dims=3,in_channels=2,out_channels=3,channels=channels,strides=strides).cuda()
        unet1_.append(unet)
        unets.append(torch.compile(unet1_[i]))
    optimizers = []
    for i in range(len(unets)):
        optimizers.append(torch.optim.Adam(unets[i].parameters(), lr=0.015))
    scaler = torch.cuda.amp.GradScaler()

    fields = []; fields_bw = []
    lambda_ = .5 #weight for regulariser
    batch = batch_size
    num_iters = 2500//batch_size*4
    train = True; use_synth = True
    
    H,W,D = (torch.tensor(vessel_fix.shape[-3:])//2).tolist()
    grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda().repeat(batch_size,1,1),(batch_size,1,H,W,D),align_corners=False)

    run_loss = torch.zeros(num_iters)

    with tqdm(total=num_iters, file=sys.stdout) as pbar:
        for iter in range(num_iters):

            idx = torch.randperm(len(vessel_fix))[:batch_size]

            #reduce spatial size by factor of 2 (to enable faster training)
            splat_fix = F.avg_pool3d(vessel_fix[idx].cuda().unsqueeze(1).float(),2)
            splat_mov = F.avg_pool3d(vessel_mov[idx].cuda().unsqueeze(1).float(),2)
            
            for i in range(len(unets)):
                optimizers[i].zero_grad()

                warped_fix,warped_mov,field_fwd,field_bwd,hr_fwd,hr_bwd = warp_sym_step(splat_fix,splat_mov,unet1_[i])
                loss = nn.L1Loss()(splat_mov,warped_fix)
                loss += nn.L1Loss()(splat_fix,warped_mov)
                
                #additional regulariser
                regular = (hr_fwd[:,:,1:]-hr_fwd[:,:,:-1]).square().mean()+(hr_fwd[:,:,:,1:]-hr_fwd[:,:,:,:-1]).square().mean()\
                +(hr_fwd[:,:,:,:,1:]-hr_fwd[:,:,:,:,:-1]).square().mean()
                regular += (hr_bwd[:,:,1:]-hr_bwd[:,:,:-1]).square().mean()+(hr_bwd[:,:,:,1:]-hr_bwd[:,:,:,:-1]).square().mean()\
                +(hr_bwd[:,:,:,:,1:]-hr_bwd[:,:,:,:,:-1]).square().mean()
                #(loss+regular*lambda_).backward()
                scaler.scale((loss+regular*lambda_)).backward()
                scaler.step(optimizers[i])
                scaler.update()


                run_loss[iter] += (loss+regular*lambda_).item()*100
                #midstep warp, see https://doi.org/10.1007/978-3-031-43999-5_65
                splat_fix = F.grid_sample(splat_fix.data.clone(),disp_square(field_fwd/2).permute(0,2,3,4,1)+grid0,align_corners=False).data
                splat_mov = F.grid_sample(splat_mov.data.clone(),disp_square(field_bwd/2).permute(0,2,3,4,1)+grid0,align_corners=False).data

                #optimizers[i].step()
                if(i==0):
                    field1_fwd = field_fwd.clone().detach().data
                    field1_bwd = field_bwd.clone().detach().data

            str1 = f"iter: {iter}, run_loss: {'%0.3f'%(run_loss[iter-5:iter-1].mean())}, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
            pbar.set_description(str1)
            pbar.update(1)
            #if(iter%50==49):
                #np.savetxt('monai_qae_log/reg_log_orig.txt',run_tre[:iter].numpy())

            if(iter%500==499):
                torch.save([unet1_[0].state_dict(),unet1_[1].state_dict()],'regnets_state_'+folder.strip('/')+'.pth')

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train_registration args')
    parser.add_argument('batch_size', help='our default 8')
    parser.add_argument('data_folder', help='folder could be dilated_vessels or synth or dilated_vessels_10')
    args = parser.parse_args()
    main(int(args.batch_size),args.data_folder)
