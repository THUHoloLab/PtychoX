# PtychoX: differentiable ptychography powered by CUDA!
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
