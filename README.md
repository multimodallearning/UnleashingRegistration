# Unleashing Registration
Diffusion Models for Paired Synthetic 3D Data Generation

This respository provides pre-trained models and the implementation to replicate the 2D OASIS and 3D Lung250M-4B synthetic paired data generation and registration experiments.
Parts of the implementation are inspired by or adapted from Sam Bond Taylor's "Unleashing Transformers" ECCV 2022 https://github.com/samb-t/unleashing-transformers. 
The 3D Lung250M-4B dataset can be found publicly at https://github.com/multimodallearning/Lung250M-4B/, here we focus on the automatic vessel segmentations of the 248 3D lung CTs and dilated the volumes by one voxel (kernel 3x3x3). 

## Training of 3D VQ-AE
We randomly draw 128x128x128 patches from the 2x 97 training scans of Lung250M-4B to train a vector-quantised 3D VQ-AE using a MONAI model:
```
net = AutoEncoder(spatial_dims=3,in_channels=2,out_channels=4,channels=(48,48,64,64,128,192,256,384),strides=(1,2,1,2,1,2,1,2)).cuda()
```
This model has approx. 600 GFlops (MulAdd counted as 2) in each forward path. It requires 4.5 GByte VRAM for forward/backward and can hence be trained on any GPU. 



## Training of 3D absorbing diffusion transformer
While the 3D VQ-AE was trained on 128x128x128 patches, we can use the whole volumes for the diffusion transformer - which eases the training/inference (batching) and enables a common positional encoding. We pad/crop a scan to 256x208x256 as follows:
```
H,W,D = vessel_orig.shape[-3:]
h1,w1,d1 = (352-H)//2-48,(256-W)//2-32,(352-D)//2-48
h2,w2,d2 = 352-H-h1-96, 256-W-w1-48, 352-D-d1-96
vessel_crop = F.pad(vessel_orig,(d1,d2,w1,w2,h1,h2))
```
Next we use the trained VQ-AE to project this binary volumetric input into latent space of size 16x13x16, where the 384D vectors are quantised to a codebook of 1024 entries. This integer tensors serve as input and target for the absorbing diffusion model. The model implementation (omitting initialisation for brevity) is as simple as:
```
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(1025, 512)
        self.position_embedding = nn.Embedding(16*13*16, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.transformer = nn.Sequential(*[nn.TransformerEncoderLayer(512, 16, dim_feedforward=1024,\
                                            batch_first=True, activation=nn.GELU()) for _ in range(12)])

    def forward(self, q):
        x = self.token_embedding(q)
        x += self.position_embedding.weight.unsqueeze(0)
        x = self.transformer(x)
        return self.head(self.layer_norm(x))
```



## Synthetic paired 3D data generation (sampling)

## Training and evaluation of two-step inverse consistent registration
This registration network is inspired by Hasting Greer's "Inverse Consistency by Construction" MICCAI 2023 https://doi.org/10.1007/978-3-031-43999-5_65 (own implementation)

## Provided models

