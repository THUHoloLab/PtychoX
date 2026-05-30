from __future__ import annotations

import argparse

import torch

from ptychoX import ptychoX_base, ptychoX_coded, ptychoX_fourier


def native_forward(sample: torch.Tensor, probe: torch.Tensor, positions_xy: torch.Tensor) -> torch.Tensor:
    probe_h, probe_w = probe.shape
    patches = []
    for x, y in positions_xy.tolist():
        patches.append(sample[y : y + probe_h, x : x + probe_w])
    patch_stack = torch.stack(patches, dim=0)
    far_field = torch.fft.fftshift(torch.fft.fft2(patch_stack * probe.unsqueeze(0)), dim=(-2, -1))
    return far_field.abs()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda-kernels", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    sample0 = torch.randn(9, 10, device=device, dtype=torch.complex64)
    probe0 = torch.randn(4, 5, device=device, dtype=torch.complex64)
    positions = torch.tensor([[0, 0], [2, 3], [5, 4]], device=device, dtype=torch.long)
    grad = torch.randn(3, 4, 5, device=device)

    sample = sample0.clone().requires_grad_(True)
    probe = probe0.clone().requires_grad_(True)
    out = ptychoX_base(sample, probe, positions, use_cuda_kernels=args.use_cuda_kernels)
    (out * grad).sum().backward()

    sample_ref = sample0.clone().requires_grad_(True)
    probe_ref = probe0.clone().requires_grad_(True)
    out_ref = native_forward(sample_ref, probe_ref, positions)
    (out_ref * grad).sum().backward()

    print(f"device: {device}")
    print(f"use_cuda_kernels: {args.use_cuda_kernels}")
    print(f"forward max error: {(out - out_ref).abs().max().item():.3e}")
    print(f"sample grad max error: {(sample.grad - sample_ref.grad).abs().max().item():.3e}")
    print(f"probe grad max error: {(probe.grad - probe_ref.grad).abs().max().item():.3e}")

    object0 = torch.randn(16, 16, device=device, dtype=torch.complex64)
    pupil0 = torch.randn(8, 8, device=device, dtype=torch.complex64)
    leds = torch.tensor([[0, 0], [2, 1], [6, 5]], device=device, dtype=torch.long)
    fpm_grad = torch.randn(3, 8, 8, device=device)

    obj = object0.clone().requires_grad_(True)
    pup = pupil0.clone().requires_grad_(True)
    fpm_out = ptychoX_fourier(obj, pup, leds, use_cuda_kernels=args.use_cuda_kernels)
    (fpm_out * fpm_grad).sum().backward()

    obj_ref = object0.clone().requires_grad_(True)
    pup_ref = pupil0.clone().requires_grad_(True)
    obj_fft = torch.fft.fft2(obj_ref)
    patches = []
    for x0, y0 in leds.tolist():
        rows = (torch.arange(8, device=device) + y0 + 8) % 16
        cols = (torch.arange(8, device=device) + x0 + 8) % 16
        patches.append(obj_fft[rows[:, None], cols[None, :]])
    sub = torch.stack(patches, dim=0)
    fpm_ref = torch.fft.ifft2(torch.fft.fftshift(sub * pup_ref.unsqueeze(0), dim=(-2, -1))).abs()
    (fpm_ref * fpm_grad).sum().backward()

    print(f"fpm forward max error: {(fpm_out - fpm_ref).abs().max().item():.3e}")
    print(f"fpm object grad max error: {(obj.grad - obj_ref.grad).abs().max().item():.3e}")
    print(f"fpm pupil grad max error: {(pup.grad - pup_ref.grad).abs().max().item():.3e}")

    coded_object0 = torch.randn(16, 16, device=device, dtype=torch.complex64)
    coded_mask0 = torch.randn(16, 16, device=device, dtype=torch.complex64)
    transfer1 = torch.randn(16, 16, device=device, dtype=torch.complex64)
    transfer2 = torch.randn(16, 16, device=device, dtype=torch.complex64)
    shifts = torch.tensor([[0.01, -0.02], [0.03, 0.01]], device=device, dtype=torch.float32)
    coded_grad = torch.randn(2, 8, 8, device=device)

    coded_object = coded_object0.clone().requires_grad_(True)
    coded_mask = coded_mask0.clone().requires_grad_(True)
    coded_out = ptychoX_coded(
        coded_object,
        coded_mask,
        transfer1,
        transfer2,
        shifts,
        2,
        use_cuda_kernels=args.use_cuda_kernels,
    )
    (coded_out * coded_grad).sum().backward()

    coded_object_ref = coded_object0.clone().requires_grad_(True)
    coded_mask_ref = coded_mask0.clone().requires_grad_(True)
    coded_ref = ptychoX_coded(coded_object_ref, coded_mask_ref, transfer1, transfer2, shifts, 2)
    (coded_ref * coded_grad).sum().backward()

    print(f"coded forward max error: {(coded_out - coded_ref).abs().max().item():.3e}")
    print(f"coded object grad max error: {(coded_object.grad - coded_object_ref.grad).abs().max().item():.3e}")
    print(f"coded mask grad max error: {(coded_mask.grad - coded_mask_ref.grad).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
