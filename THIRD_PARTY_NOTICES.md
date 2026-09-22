# Third-party notices

This repository builds on the following open-source projects.

## VQ-Diffusion

<https://github.com/microsoft/VQ-Diffusion> — MIT License, Copyright (c)
Microsoft Corporation.

The discrete diffusion process, the denoising transformer and the training
solver in `laa_ldm/diffusion/` are derived from VQ-Diffusion ("Vector Quantized
Diffusion Model for Text-to-Image Synthesis", Gu et al., CVPR 2022, and
"Improved Vector Quantized Diffusion Models", Tang et al., 2022). The text and
class conditioning paths were replaced by conditioning on continuous anatomical
shape descriptors, and the content stream was moved from 2D images to 3D
occupancy volumes.

## MONAI and MONAI GenerativeModels

<https://github.com/Project-MONAI/MONAI> and
<https://github.com/Project-MONAI/GenerativeModels> — Apache License 2.0,
Copyright (c) MONAI Consortium.

The encoder/decoder blocks in `laa_ldm/codec/` are derived from the MONAI
GenerativeModels `AutoencoderKL`, extended with FiLM conditioning on the shape
descriptors. The patch discriminator and the perceptual loss are used directly
from MONAI GenerativeModels.

## Taming Transformers

<https://github.com/CompVis/taming-transformers> — MIT License.

The vector-quantiser and the VQ-GAN loss formulation follow "Taming
Transformers for High-Resolution Image Synthesis" (Esser et al., CVPR 2021).
