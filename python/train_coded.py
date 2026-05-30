from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ptychoX import ptychoX_coded
from ptychoX.viewers import (
    LiveReconstructionViewer,
    ViserReconstructionViewer,
    save_complex_reconstruction,
    terminal_link,
)


def read_gray(path: Path, size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.BICUBIC)
    data = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).to(device=device, dtype=torch.float32)
    return data / 255.0


def mat2gray(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)


def make_object(size: int, source_dir: Path, device: torch.device) -> torch.Tensor:
    amp = read_gray(source_dir / "cameraman512.png", size, device)
    phase = read_gray(source_dir / "I01.bmp", size, device)
    return mat2gray(amp + 0.2).to(torch.complex64) * torch.exp(1j * math.pi * phase)


def make_coded_surface(
    size: int,
    device: torch.device,
    generator: torch.Generator,
    phase_scale: float,
) -> torch.Tensor:
    coarse = torch.rand((1, 1, 32, 32), device=device, generator=generator)
    amp = F.interpolate(coarse, size=(size, size), mode="area")[0, 0]
    phase = phase_scale * math.pi * (amp - amp.mean())
    return amp.clamp_min(0.05).to(torch.complex64) * torch.exp(1j * phase)


def make_transfer(
    size: int,
    pixel_size: float,
    wavelength: float,
    distance: float,
    device: torch.device,
) -> torch.Tensor:
    freq = torch.fft.fftfreq(size, d=pixel_size, device=device)
    fy, fx = torch.meshgrid(freq, freq, indexing="ij")
    inv_lambda = 1.0 / wavelength
    kz_sq = inv_lambda**2 - fx.square() - fy.square()
    kz_real = torch.sqrt(kz_sq.clamp_min(0.0))
    evanescent = torch.sqrt((-kz_sq).clamp_min(0.0))
    phase = torch.exp(1j * 2.0 * math.pi * distance * kz_real)
    decay = torch.exp(-2.0 * math.pi * abs(distance) * evanescent)
    return (phase * decay * (kz_sq >= 0)).to(torch.complex64)


def make_shifts(samples: int, shift_range: float, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    shifts_px = shift_range * (2.0 * torch.rand((samples, 2), device=device, generator=generator) - 1.0)
    return shifts_px


@torch.no_grad()
def synthesize_dataset(
    object_true: torch.Tensor,
    coded_true: torch.Tensor,
    transfer_1: torch.Tensor,
    transfer_2: torch.Tensor,
    shifts_norm: torch.Tensor,
    downsample: int,
    batch_size: int,
    use_cuda_kernels: bool,
) -> torch.Tensor:
    chunks = []
    for start in range(0, shifts_norm.shape[0], batch_size):
        chunks.append(
            ptychoX_coded(
                object_true,
                coded_true,
                transfer_1,
                transfer_2,
                shifts_norm[start : start + batch_size],
                downsample,
                use_cuda_kernels=use_cuda_kernels,
            )
        )
    observed = torch.cat(chunks, dim=0)
    return observed


def fd_loss(pred: torch.Tensor, target: torch.Tensor, isotropic: bool = True) -> torch.Tensor:
    diff = pred - target
    dx = torch.roll(diff, shifts=-1, dims=-1) - diff
    dy = torch.roll(diff, shifts=-1, dims=-2) - diff
    if isotropic:
        return torch.sqrt(dx.square() + dy.square() + 1e-5).sum()
    return (dx.abs() + dy.abs()).sum()


def cosine_lrs(base_lrs: list[float], min_lr: float, epochs: int, epoch: int, scale: float = 1.0) -> list[float]:
    phase = 1.0 if epochs <= 1 else (1.0 + math.cos(math.pi * (epoch - 1) / epochs)) * 0.5
    return [(min_lr + (base_lr - min_lr) * phase) * scale for base_lr in base_lrs]


def set_optimizer_lrs(optimizer: torch.optim.Optimizer, lrs: list[float]) -> None:
    for group, lr in zip(optimizer.param_groups, lrs):
        group["lr"] = lr


def build_viewer(args: argparse.Namespace):
    if args.live:
        args.viewer = "matplotlib"
    if args.viewer == "matplotlib":
        return LiveReconstructionViewer("PyTorch Coded Ptychography")
    if args.viewer == "viser":
        viewer = ViserReconstructionViewer(
            args.viser_host,
            args.viser_port,
            args.viser_panel_size,
            title="Coded Ptychography",
            amp_label="object amp",
            phase_label="object phase",
            probe_label="coded surface",
            probe_gt_label="coded_GT",
        )
        viewer_url = f"http://localhost:{args.viser_port}"
        print("")
        print("Viewer running:")
        print(f"  {terminal_link(viewer_url)}")
        print("")
        return viewer
    return None


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    high_res = args.pix * args.mag
    source_dir = Path(args.source_dir)
    object_true = make_object(high_res, source_dir, device)
    coded_true = make_coded_surface(high_res, device, generator, args.coded_phase_scale)

    pixel_size = args.pixel_size / args.mag
    transfer_1 = make_transfer(high_res, pixel_size, args.wavelength, args.distance_1, device)
    transfer_2 = make_transfer(high_res, pixel_size, args.wavelength, args.distance_2, device)
    shifts_px = make_shifts(args.samples, args.shift_range, device, generator)
    shifts_norm = shifts_px / args.pix

    observed_amp = synthesize_dataset(
        object_true,
        coded_true,
        transfer_1,
        transfer_2,
        shifts_norm,
        args.mag,
        args.batch_size,
        use_cuda_kernels=args.use_cuda_kernels,
    )

    object_param = torch.nn.Parameter(torch.ones_like(object_true))
    if args.fix_coded_surface:
        coded_param = coded_true.detach()
        param_groups = [{"params": [object_param], "lr": args.lr}]
    else:
        coded_param = torch.nn.Parameter(torch.ones_like(coded_true))
        param_groups = [
            {"params": [object_param], "lr": args.lr},
            {"params": [coded_param], "lr": args.coded_lr if args.coded_lr is not None else args.lr},
        ]
    optimizer = torch.optim.Adam(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-16,
    )
    initial_lrs = [group["lr"] for group in optimizer.param_groups]

    out_dir = Path(args.out_dir)
    print(f"device={device}, pix={args.pix}, high_res={high_res}, samples={args.samples}, cuda_kernels={args.use_cuda_kernels}")
    viewer = build_viewer(args)
    loss_history: list[float] = []
    if viewer is not None:
        viewer.update(object_param, coded_param, [1.0], 0, coded_true, initial_lrs)
        if isinstance(viewer, ViserReconstructionViewer):
            viewer.status.value = "epoch=0, initialized"
            viewer.server.flush()

    stopped_early = False
    final_epoch = 0
    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        order = torch.randperm(args.samples, generator=generator, device=device)
        epoch_loss = 0.0
        progress = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        for start in range(0, args.samples, args.batch_size):
            if isinstance(viewer, ViserReconstructionViewer):
                while viewer.paused and not viewer.stop_requested:
                    time.sleep(0.05)
                if viewer.stop_requested:
                    stopped_early = True
                    break

            batch_idx = order[start : start + args.batch_size]
            progress += batch_idx.numel()
            scale = float(viewer.lr_scale.value) if isinstance(viewer, ViserReconstructionViewer) else 1.0
            current_lrs = cosine_lrs(initial_lrs, args.min_lr, args.epochs, epoch, scale)
            set_optimizer_lrs(optimizer, current_lrs)

            pred_amp = ptychoX_coded(
                object_param,
                coded_param,
                transfer_1,
                transfer_2,
                shifts_norm[batch_idx],
                args.mag,
                use_cuda_kernels=args.use_cuda_kernels,
            )
            if args.loss == "mse":
                loss = F.mse_loss(pred_amp, observed_amp[batch_idx])
            else:
                loss = fd_loss(pred_amp, observed_amp[batch_idx], isotropic=args.loss == "fd-isotropic")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if args.real_coded_surface and not args.fix_coded_surface:
                with torch.no_grad():
                    coded_param.copy_(coded_param.real.clamp_min(0.0).to(torch.complex64))
            epoch_loss += loss.detach().item() * batch_idx.numel()

            if isinstance(viewer, ViserReconstructionViewer):
                viewer.status.value = (
                    f"epoch={epoch}, batch={start // args.batch_size + 1}/"
                    f"{math.ceil(args.samples / args.batch_size)}, loss={loss.detach().item():.6e}, running"
                )
                viewer.server.flush()

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event) / 1000.0

        if progress == 0:
            break

        epoch_loss /= progress
        loss_history.append(epoch_loss)
        scale = float(viewer.lr_scale.value) if isinstance(viewer, ViserReconstructionViewer) else 1.0
        current_lrs = cosine_lrs(initial_lrs, args.min_lr, args.epochs, epoch, scale)
        if epoch == 1 or epoch % args.log_every == 0:
            lr_text = ", ".join(f"{lr:.3e}" for lr in current_lrs)
            print(f"epoch {epoch:04d}  loss={epoch_loss:.6e}  lr=[{lr_text}]  time={elapsed_time_ms:.2f} s")

        if viewer is not None and (epoch == 1 or epoch % args.display_every == 0):
            viewer_alive = viewer.update(object_param, coded_param, loss_history, epoch, coded_true, current_lrs)
            if not viewer_alive:
                viewer = None

        if args.save_every > 0 and (epoch == args.epochs or epoch % args.save_every == 0):
            save_complex_reconstruction(object_param, coded_param, out_dir, epoch, prefix="coded")

        if stopped_early:
            break

       
            
    if not args.no_save:
        save_complex_reconstruction(object_param, coded_param, out_dir, final_epoch, prefix="coded")
        torch.save(
            {
                "object": object_param.detach().cpu(),
                "coded_surface": coded_param.detach().cpu(),
                "shifts_xy_pixels": shifts_px.detach().cpu(),
                "observed_amplitude": observed_amp.detach().cpu(),
                "args": vars(args),
            },
            out_dir / "coded_reconstruction.pt",
        )

    if args.keep_viewer_open and isinstance(viewer, ViserReconstructionViewer):
        print("Training finished. Press Ctrl+C to stop the viser server.")
        try:
            viewer.server.sleep_forever()
        except KeyboardInterrupt:
            viewer.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch coded ptychography reconstruction demo.")
    parser.add_argument("--pix", type=int, default=64)
    parser.add_argument("--mag", type=int, default=4)
    parser.add_argument("--samples", type=int, default=225)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--coded-lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("fd-isotropic", "fd-anisotropic", "mse"), default="mse")
    parser.add_argument("--coded-phase-scale", type=float, default=0.0, help="Synthetic phase scale for the coded mask.")
    parser.add_argument("--fix-coded-surface", action="store_true", help="Use the ground-truth coded surface and optimize only the object.")
    parser.add_argument(
        "--real-coded-surface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Constrain reconstructed coded surface to real nonnegative values.",
    )
    parser.add_argument("--shift-range", type=float, default=5.0)
    parser.add_argument("--wavelength", type=float, default=0.405e-6)
    parser.add_argument("--pixel-size", type=float, default=1.85e-6)
    parser.add_argument("--distance-1", type=float, default=396.95e-6)
    parser.add_argument("--distance-2", type=float, default=838e-6)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-cuda-kernels", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--viewer", choices=("none", "matplotlib", "viser"), default="none")
    parser.add_argument("--display-every", type=int, default=1)
    parser.add_argument("--viser-host", type=str, default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--viser-panel-size", type=int, default=512)
    parser.add_argument("--keep-viewer-open", action="store_true")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="matlab/Examples/Coded Ptychography/CP_datasets",
        help="Directory containing cameraman512.png and I01.bmp.",
    )
    parser.add_argument("--out-dir", type=str, default="python/ptychography/runs_coded")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
