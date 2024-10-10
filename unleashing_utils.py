import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm.auto import tqdm,trange
import torch.distributions as distributions

#straightforward transformer model (here used in 3D for token shape 16x13x16)
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(1025, 512)
        self.position_embedding = nn.Embedding(16*13*16, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.token_embedding.weight.data.normal_(.0,.02)
        self.position_embedding.weight.data.normal_(.0,.02)
        self.layer_norm.bias.data.zero_()
        self.layer_norm.weight.data.fill_(1.0)
        self.head = nn.Linear(512, 1024, bias=False)
        self.head.weight.data.normal_(.0,.02)
        
        self.transformer = nn.Sequential(*[nn.TransformerEncoderLayer(512, 16, dim_feedforward=1024,batch_first=True, activation=nn.GELU()) for _ in range(12)])

    def forward(self, q):
        x = self.token_embedding(q)
        x += self.position_embedding.weight.unsqueeze(0)
        x = self.transformer(x)
        return self.head(self.layer_norm(x))

#adapted from https://github.com/samb-t/unleashing-transformers/blob/master/models/absorbing_diffusion.py

def synthesise_fn(denoise_fn,temp=.9):
    num_timesteps = 256
    mask_id = 1024
    #create tensor with only masked entries
    x_t = torch.ones(2,16*13*16).cuda().long()*mask_id
    #create an empty tensor to store masked/unmasked elements
    unmasked = torch.zeros_like(x_t).bool()
    for t_iter in trange(num_timesteps):
        # define random change (gradually increase the number of unmasked elements)
        t1 = torch.ones(2,16*13*16).cuda() / float(num_timesteps-t_iter)
        # create and apply random mask
        change_mask = torch.rand_like(x_t.float()) < t1
        # don't unmask somewhere already unmasked
        change_mask = torch.bitwise_xor(change_mask, torch.bitwise_and(change_mask, unmasked))
        # predict logits with denoiser (and scale by tempature)
        x_0_logits = denoise_fn(x_t)/temp
        # instead of taking a "hard" argmax the probabilities of the logits should be used
        # to softly sample the best elements for the required change positions
        x_0_dist = distributions.Categorical(logits=x_0_logits)
        x_0_hat = x_0_dist.sample().long()
        # finally insert predicted tokens where change was required and update mask
        x_t[change_mask] = x_0_hat[change_mask]
        unmasked = torch.bitwise_or(unmasked, change_mask)

    return x_t


def weighted_elbo_loss(x_logits,target,t):
    cross_entropy_loss = F.cross_entropy(x_logits, target, ignore_index=-1, reduction='none').sum(1)
    weight = (1 - t) #lower weight for earlier (more difficult) time points
    loss = weight * cross_entropy_loss
    loss = loss / (float(torch.log(torch.tensor([2]))) * x_logits.shape[1:].numel())
    return loss.mean()

#thanks to https://github.com/samb-t/unleashing-transformers/blob/master/models/vqgan.py
#only extended to 3D by us
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=1024, emb_dim=256, beta=.25):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta  # commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        # find closest encodings
        d_argmin = torch.argmin(d, dim=1)
        # get quantized latent vectors
        z_q = self.embedding(d_argmin).view(z.shape)
        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()
        
        return z_q, loss, d_argmin
    
def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice
def optim_warmup(lr_base, step, optim, warmup_iters):
    lr = lr_base * float(step) / warmup_iters
    for param_group in optim.param_groups:
        param_group['lr'] = lr

## REGISTRATION

device = 'cuda'

def disp_square(field):
    field2 = field/2**12
    B,_,H,W,D = field.shape
    grid1 = F.affine_grid(torch.eye(3,4).unsqueeze(0).to(device).repeat(B,1,1),(B,1,H,W,D),align_corners=False)
    for i in range(12):
        field2 = F.grid_sample(field2,field2.permute(0,2,3,4,1)+grid1,align_corners=False,padding_mode='border')+field2 #compose
    return field2

def compose(field,field1):
    B,_,H,W,D = field.shape
    grid1 = F.affine_grid(torch.eye(3,4).unsqueeze(0).to(device).repeat(B,1,1),(B,1,H,W,D),align_corners=False)
    field2 = F.grid_sample(field1.float(),field.permute(0,2,3,4,1)+grid1,align_corners=False,padding_mode='border')+field #compose
    return field2
def warp_sym_step(splat_fix,splat_mov,unet):
    batch,_,H,W,D = splat_fix.shape
    grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda().repeat(batch,1,1),(batch,1,H,W,D),align_corners=False)
    kernel = 7; half_width = (kernel-1)//2
    avg5_ = nn.AvgPool3d(kernel,stride=2,padding=half_width)
    avg5 = nn.AvgPool3d(kernel,stride=1,padding=half_width)

    with torch.cuda.amp.autocast(dtype=torch.float16):

        output_fwd = torch.tanh(unet(torch.cat((splat_fix.data,splat_mov.data),1)))*.35#25
        output_bwd = torch.tanh(unet(torch.cat((splat_mov.data,splat_fix.data),1)))*.35#25

    field_fwd = F.interpolate(avg5(avg5(avg5_(avg5_(output_fwd-output_bwd)))),size=(H,W,D),mode='trilinear').float()
    hr_fwd = disp_square(field_fwd)

    warped_fix = F.grid_sample(splat_fix,hr_fwd.permute(0,2,3,4,1)+grid0,align_corners=False)
    field_bwd = F.interpolate(avg5(avg5(avg5_(avg5_(output_bwd-output_fwd)))),size=(H,W,D),mode='trilinear').float()
    hr_bwd = disp_square(field_bwd)
    warped_mov = F.grid_sample(splat_mov,hr_bwd.permute(0,2,3,4,1)+grid0,align_corners=False)
    return warped_fix,warped_mov,field_fwd,field_bwd,hr_fwd,hr_bwd

