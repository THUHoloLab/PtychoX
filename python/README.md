# PyTorch PtychoX example

This directory contains a PyTorch port of the classical ptychography example in
`matlab/Examples/Ptychography/test_recon.m`.

The forward model is:

1. crop a probe-sized patch from the large complex object,
2. multiply the patch by the complex probe,
3. apply `fft2` and `fftshift`,
4. compare the predicted Fourier amplitude with measured amplitude.

PyTorch's native complex autograd is used for the backward pass, so no custom
CUDA extension is required.

## Quick run

```powershell
python python\ptychography\train_ptychography.py --epochs 20
```

Open a MATLAB-like live window while training:

```powershell
python python\ptychography\train_ptychography.py --epochs 100 --live
```

Or use a browser UI through `viser`:

```powershell
python python\ptychography\train_ptychography.py --epochs 100 --viewer viser --viser-port 8080
```

Then open `http://localhost:8080`. The viser panel exposes pause/stop controls,
an LR scale slider, scene zoom, and a live loss curve. Add `--keep-viewer-open`
if you want the viser page to remain available after training finishes.

The live viewers and image/loss plotting helpers live in `ptychoX.viewers` and
are shared by the ptychography and Fourier ptychography training scripts.

By default the script uses a small problem (`pix=128`, `mag=4`, `samples=48`) so
it can be tested quickly. To move closer to the MATLAB example:

```powershell
python python\ptychography\train_ptychography.py --pix 1024 --mag 4 --samples 300 --epochs 600 --batch-size 16 --lr 0.1
```

Outputs are written to `python/ptychography/runs/` unless `--out-dir` is set.

## Optional CUDA extension

The default operator is a custom `torch.autograd.Function` implemented with
PyTorch FFTs. It is fully differentiable and does not need compilation.

To prebuild the optional CUDA crop/scatter kernels, run:

```powershell
cd python\ptychography
python setup.py build_ext --inplace
python test_ops.py --use-cuda-kernels
```

The compiled extensions are placed inside their operator package directories,
for example:

- `python/ptychography/ptychoX/base/_C.cp313-win_amd64.pyd`
- `python/ptychography/ptychoX/fpm/_C.cp313-win_amd64.pyd`
- `python/ptychography/ptychoX/coded/_C.cp313-win_amd64.pyd`

Then enable the kernels during training:

```powershell
python python\ptychography\train_ptychography.py --use-cuda-kernels
```

If the compiled module is not found, `ops.py` falls back to lazy JIT compilation
inside `python/ptychography/.torch_extensions/` and prints compiler output.

Each CUDA extension caches cuFFT plans by `(device, batch, height, width)`, so a
fixed batch/probe size does not recreate `cufftHandle`s every iteration. Call
`ptychoX.clear_cuda_plan_cache()` if you need to explicitly destroy all cached
plans, or `ptychoX.clear_base_cuda_plan_cache()` /
`ptychoX.clear_fpm_cuda_plan_cache()` /
`ptychoX.clear_coded_cuda_plan_cache()` to clear one operator family.

## Fourier ptychography operator

`ptychoX.ptychoX_fourier()` ports the Fourier ptychography forward/backward
operator. The PyTorch convention is row-major and zero-based:

- object: `[high_h, high_w]` complex tensor
- pupil: `[low_h, low_w]` complex tensor
- LED indices: `[batch, 2]` zero-based `(x, y)` offsets
- output: `[batch, low_h, low_w]` predicted amplitudes

The optional CUDA path uses cuFFT for the high-resolution object FFT and the
batched low-resolution FFTs, with the same cuFFT plan cache used by the
classical ptychography operator.

Run the synthetic Fourier ptychography training demo:

```powershell
python python\ptychography\train_fpm.py --epochs 100 --viewer viser --use-cuda-kernels
```

The synthetic pupil includes low-order Zernike aberrations by default. Adjust or
disable them with:

```powershell
python python\ptychography\train_fpm.py --aberration-scale 1.2
python python\ptychography\train_fpm.py --no-aberration
```

For a quick smoke test without saving:

```powershell
python python\ptychography\train_fpm.py --low-res 16 --pratio 2 --samples 4 --epochs 2 --batch-size 2 --save-every 0 --no-save
```

## Coded ptychography

`ptychoX.ptychoX_coded()` ports the coded ptychography forward model from
the MATLAB examples. It has a torch FFT/autograd fallback and an optional
cached-cuFFT CUDA extension. It lives in its own package folder:

- `python/ptychography/ptychoX/coded/ops.py`
- `python/ptychography/train_coded.py`

Run the synthetic coded ptychography demo:

```powershell
python python\ptychography\train_coded.py --epochs 100 --viewer viser
```

Use the compiled CUDA path:

```powershell
python python\ptychography\train_coded.py --epochs 100 --viewer viser --use-cuda-kernels
```

Quick smoke test:

```powershell
python python\ptychography\train_coded.py --pix 16 --mag 2 --samples 4 --epochs 2 --batch-size 2 --save-every 0 --no-save
```
