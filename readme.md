# PtychoX: differentiable ptychography powered by CUDA!

  ![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?logo=apache)
  ![NVIDIA GPU](https://img.shields.io/badge/gpu-nvidia-green?logo=nvidia)
  ![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)
  
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

