
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

def main(iter_tta,chkpt_reg):
    
    keypts_f,keypts_m = torch.load('lms_validation_cropped.pth')

    #load vessels for fixed and moving (test scans)
    folder = 'dilated_vessels_ts/'
    files = sorted(os.listdir(folder))#[:10]
    #check whether data are raw nii.gz files or synthetic npz (latter requires no
    vessel_fix = []; vessel_mov = [];
    cases = torch.tensor([int(f.split('_')[1]) for f in files]).unique()

    for i in trange(len(cases)):
        vessel_fix.append(torch.from_numpy(nib.load(folder+'/case_'+str(int(cases[i])).zfill(3)+'_1.nii.gz').get_fdata()>0).float().contiguous())
        vessel_mov.append(torch.from_numpy(nib.load(folder+'/case_'+str(int(cases[i])).zfill(3)+'_2.nii.gz').get_fdata()>0).float().contiguous())
    
    #pad crop to 256x208x256
    for i in trange(len(vessel_fix)):
        with torch.no_grad():
            H,W,D = vessel_fix[i].shape[-3:]
            h1,w1,d1 = (22*16-H)//2-48,(16*16-W)//2-32,(22*16-D)//2-48
            h2,w2,d2 = 22*16-H-h1-96, 16*16-W-w1-48, 22*16-D-d1-96
            vessel_fix[i] = F.pad(vessel_fix[i],(d1,d2,w1,w2,h1,h2))
            vessel_mov[i] = F.pad(vessel_mov[i],(d1,d2,w1,w2,h1,h2))
    vessel_fix = torch.stack(vessel_fix)
    vessel_mov = torch.stack(vessel_mov)
    ##setting up registration training routine
    sym = True #always true
    #initialise UNets and optimisers
    unet1_ = []; unets = [];
    channels = (8,16,32,64,64,64)
    strides = (2,2,1,2,1)
    lambda_ = 0.5
    states = torch.load(chkpt_reg)
    for i in range(2):
        unet = UNet(spatial_dims=3,in_channels=2,out_channels=3,channels=channels,strides=strides).cuda()
        unet1_.append(unet)
        unets.append(torch.compile(unet1_[i]))
    tre0 = []; tre1 = []; tre2 = []
    print(tuple((torch.tensor(vessel_fix[0].shape[-3:])//2).tolist()))
    H,W,D = tuple((torch.tensor(vessel_fix[0].shape[-3:])//2).tolist())
    grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H,W,D),align_corners=False)

    for j in range(len(vessel_fix)):
        kt_fix = (keypts_f[j].div(2).div(torch.tensor([H/2,W/2,D/2])).flip(-1)-1)#.cuda()
        kt_mov = (keypts_m[j].div(2).div(torch.tensor([H/2,W/2,D/2])).flip(-1)-1)#.cuda()
        
        tre0_ = 2*((kt_fix-kt_mov)*torch.tensor([D/2,W/2,H/2]).cpu()).square().sum(-1).sqrt()
        tre0.append(tre0_)
        
        for i in range(len(unets)):
            unet1_[i].load_state_dict(states[i])

        optimizers = []
        for i in range(len(unets)):
            optimizers.append(torch.optim.Adam(unets[i].parameters(), lr=0.005))
        scaler = torch.cuda.amp.GradScaler()


        for subiter in trange(iter_tta):
            splat_fix = F.avg_pool3d(vessel_fix[j:j+1].cuda().unsqueeze(1).float(),2)
            splat_mov = F.avg_pool3d(vessel_mov[j:j+1].cuda().unsqueeze(1).float(),2)

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
                #midstep warp
                splat_fix = F.grid_sample(splat_fix.data.clone(),disp_square(field_fwd/2).permute(0,2,3,4,1)+grid0,align_corners=False).data
                splat_mov = F.grid_sample(splat_mov.data.clone(),disp_square(field_bwd/2).permute(0,2,3,4,1)+grid0,align_corners=False).data

                scaler.scale((loss+regular*lambda_)).backward()
                scaler.step(optimizers[i])
                scaler.update()
                #optimizers[i].step()
                if(i==0):
                    field1_fwd = field_fwd.clone().detach().data
                    field1_bwd = field_bwd.clone().detach().data
            if((subiter==0)|(subiter==iter_tta-1)):
                with torch.no_grad():
                    field = compose(compose(disp_square(field1_fwd/2),disp_square(field_fwd.data)),disp_square(field1_fwd/2))
                    field_bw = compose(compose(disp_square(field1_bwd/2),disp_square(field_bwd.data)),disp_square(field1_bwd/2))
                    disp_kpts = F.grid_sample(field.data.float().cpu(),kt_mov.view(1,-1,1,1,3),align_corners=False).squeeze().t()
        
                kt_mov_warp = kt_mov+disp_kpts
                if(subiter==0):
                    tre1_ = 2*((kt_fix-kt_mov_warp)*torch.tensor([D/2,W/2,H/2]).cpu()).square().sum(-1).sqrt()
                    tre1.append(tre1_)
                else:
                    tre2_ = 2*((kt_fix-kt_mov_warp)*torch.tensor([D/2,W/2,H/2]).cpu()).square().sum(-1).sqrt()
                    tre2.append(tre2_)
                
        print(tre0_.mean(),tre1_.mean(),tre2_.mean())
    np.savetxt(chkpt_reg[:-4]+'.csv',torch.stack((torch.cat(tre0),torch.cat(tre1),torch.cat(tre2)),-1),delimiter=',')
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'evaluate_registration args')
    parser.add_argument('iter_tta', help='our default 50')
    parser.add_argument('chkpt_reg', help='e.g. regnets_state_dilated_vessels.pth')
    args = parser.parse_args()
    main(int(args.iter_tta),args.chkpt_reg)
