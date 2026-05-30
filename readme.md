# PtychoX: differentiable ptychography powered by CUDA!

  ![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?logo=apache)
  ![MATLAB](https://img.shields.io/badge/MATLAB-2024b-red.svg?logo=mathworks)
  ![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
  ![NVIDIA GPU](https://img.shields.io/badge/gpu-nvidia-green?logo=nvidia)
  ![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg?logo=nvidia)
  
PtychoX is an open-source library that provides CUDA-accelerated, differentiable phase-retrieval solvers for the ptychographic imaging family, with native MATLAB bindings. The project is inspired by the Chinese paper “CUDA-Accelerated Fourier Ptychographic Microscopy (Invited)” [[1](https://researching.cn/ArticlePdf/m00002/2025/62/15/1511001.pdf)]. Building upon this foundation, PtychoX extends the original framework to **four distinct ptychographic applications**, each implemented with end-to-end differentiable formulations.

<div align = 'center'>
<img src = "https://github.com/THUHoloLab/PtychoX/blob/main/resources/layouts.png" width = "600" alt="" align = center />
<br>
<em>Typical ptychographic imaging layouts</em>
</div>
<br>

## Requirements
- CUDA 12.8
- MATLAB R2023b or later
- MATLAB Deep Learning Toolbox
- A CUDA-capable NVIDIA GPU

## Testing
The following experiments were conducted on an OMEN Transcend 14 gaming laptop equipped with an Intel® Core™ Ultra 9 185H processor (2.50 GHz), 32 GB RAM, and an NVIDIA® GeForce RTX™ 4060 Laptop GPU. Raw image is of 256 by 256 pixels, and the reconstruction image is of 1024 by 1024 pixels.
<div align = 'center'>
<img src = "https://github.com/THUHoloLab/PtychoX/blob/main/resources/compared.gif" width = "600" alt="" align = center />
<br>
<em>Comparison between PtychoX and conventional gpuArray based recovery on classical ptychography.</em>
</div>
<br>

The following experiments were conducted on an desktop equipped with an Intel® Core™ i9-12900L 3.2GHz, 128 GB RAM, and an NVIDIA® GeForce RTX™ 3090. Raw image is of 512 by 512 pixels, a total of 900 raw images. The reconstruction image is of 2048 by 2048 pixels.
The scripts are available in `Benchmarks/ptychography-testing-RTX3090/`
<div align = 'center'>
<img src = "https://github.com/THUHoloLab/PtychoX/blob/main/resources/compared2.gif" width = "800" alt="" align = center />
<br>
<em>Comparison between PtychoX and conventional gpuArray based recovery on classical ptychography.</em>
</div>
<br>

## Example
### Conventional ptychography
The script `Examples/Ptychography/test_recon.m` provides a default example for training a ptychographic model based on the conventional Fourier-transform-based ptychography framework.  
This script follows the same reconstruction logic as the classical ptychography implementation.
