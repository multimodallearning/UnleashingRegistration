# Unleashing Registration
Diffusion Models for Paired Synthetic 3D Data Generation

This respository provides pre-trained models and the implementation to replicate the 2D OASIS and 3D Lung250M-4B synthetic paired data generation and registration experiments.
Parts of the implementation are inspired by or adapted from Sam Bond Taylor's "Unleashing Transformers" ECCV 2022 https://github.com/samb-t/unleashing-transformers. 
The 3D Lung250M-4B dataset can be found publicly at https://github.com/multimodallearning/Lung250M-4B/, here we focus on the automatic vessel segmentations of the 248 3D lung CTs and dilated the volumes. 

## Training of 3D VQ-VAE


## Training of 3D absorbing diffusion transformer


## Synthetic paired 3D data generation (sampling)


## Training and evaluation of two-step inverse consistent registration
This registration network is inspired by Hasting Greer's "Inverse Consistency by Construction" MICCAI 2023 https://doi.org/10.1007/978-3-031-43999-5_65 (own implementation)

## Provided models

