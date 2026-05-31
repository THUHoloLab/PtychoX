from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ptychoX import ptychoX_fourier
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


def make_pupil(
    low_res: int,
    device: torch.device,
    aberration_scale: float,
    no_aberration: bool,
) -> torch.Tensor:
    coord = torch.linspace(-1.0, 1.0, low_res, device=device)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    rho = torch.sqrt(xx.square() + yy.square()).clamp_max(1.0)
    theta = torch.atan2(yy, xx)
    aperture = (rho <= 0.75).float()
    phase = torch.zeros_like(rho)
    if not no_aberration and aberration_scale != 0.0:
        phase = aberration_scale * default_zernike_aberration(rho / 0.75, theta) * aperture
    return aperture.to(torch.complex64) * torch.exp(1j * phase)


def default_zernike_aberration(rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """A modest synthetic wavefront aberration in radians.

    The modes are OSA/ANSI-like low-order Zernike terms on a unit pupil:
    defocus, astigmatism, coma, trefoil, and primary spherical aberration.
    Coefficients are intentionally small so the default demo remains trainable.
    """
    modes = {
        (2, 0): 0.55,   # defocus
        (2, 2): 0.25,   # oblique astigmatism
        (2, -2): -0.20, # vertical astigmatism
        (3, 1): 0.18,   # vertical coma
        (3, -1): -0.12, # horizontal coma
        (3, 3): 0.08,   # oblique trefoil
        (4, 0): 0.14,   # primary spherical
    }
    phase = torch.zeros_like(rho)
    for (n, m), coeff in modes.items():
        phase = phase + coeff * zernike(n, m, rho, theta)
    return phase


def zernike(n: int, m: int, rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    abs_m = abs(m)
    radial = zernike_radial(n, abs_m, rho)
    if m > 0:
        return radial * torch.cos(abs_m * theta)
    if m < 0:
        return radial * torch.sin(abs_m * theta)
    return radial


def zernike_radial(n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
    if (n - m) % 2 != 0:
        return torch.zeros_like(rho)
    radial = torch.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coeff = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
        )
        radial = radial + coeff * rho.pow(n - 2 * k)
    return radial


def make_led_indices(
    high_res: int,
    low_res: int,
    samples: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    max_offset = max(1, high_res - low_res)
    grid_n = int(math.ceil(math.sqrt(samples)))
    vals = torch.linspace(0, max_offset, grid_n, device=device).round().long()
    yy, xx = torch.meshgrid(vals, vals, indexing="ij")
    led = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)[:samples]
    if led.shape[0] < samples:
        extra = torch.randint(0, max_offset + 1, (samples - led.shape[0], 2), device=device, generator=generator)
        led = torch.cat([led, extra], dim=0)
    return led


@torch.no_grad()
def synthesize_dataset(
    object_true: torch.Tensor,
    pupil_true: torch.Tensor,
    led_indices: torch.Tensor,
    batch_size: int,
    use_cuda_kernels: bool,
) -> torch.Tensor:
    chunks = []
    for start in range(0, led_indices.shape[0], batch_size):
        chunks.append(
            ptychoX_fourier(
                object_true,
                pupil_true,
                led_indices[start : start + batch_size],
                use_cuda_kernels=use_cuda_kernels,
            )
        )
    return torch.cat(chunks, dim=0)


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
        return LiveReconstructionViewer("PyTorch Fourier Ptychography")
    if args.viewer == "viser":
        viewer = ViserReconstructionViewer(
            args.viser_host,
            args.viser_port,
            args.viser_panel_size,
            title="Fourier Ptychography",
            amp_label="object amp",
            phase_label="object phase",
            probe_label="pupil",
            probe_gt_label="pupil_GT",
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

    high_res = args.low_res * args.pratio
    source_dir = Path(args.source_dir)
    object_true = make_object(high_res, source_dir, device)
    pupil_true = make_pupil(args.low_res, device, args.aberration_scale, args.no_aberration)
    led_indices = make_led_indices(high_res, args.low_res, args.samples, device, generator)
    observed_amp = synthesize_dataset(
        object_true,
        pupil_true,
        led_indices,
        args.batch_size,
        use_cuda_kernels=args.use_cuda_kernels,
    )

    init_amp = F.interpolate(
        observed_amp.mean(dim=0, keepdim=True).unsqueeze(0),
        size=(high_res, high_res),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    object_param = torch.nn.Parameter(init_amp.to(torch.complex64))
    pupil_param = torch.nn.Parameter((pupil_true.abs() > 0).float().to(torch.complex64))

    optimizer = torch.optim.Adam(
        [
            {"params": [object_param], "lr": args.lr},
            {"params": [pupil_param], "lr": args.pupil_lr if args.pupil_lr is not None else args.lr},
        ],
        betas=(0.9, 0.999),
        eps=1e-16,
    )
    initial_lrs = [group["lr"] for group in optimizer.param_groups]

    out_dir = Path(args.out_dir)
    print(
        f"device={device}, low_res={args.low_res}, high_res={high_res}, "
        f"samples={args.samples}, cuda_kernels={args.use_cuda_kernels}"
    )
    viewer = build_viewer(args)
    loss_history: list[float] = []
    if viewer is not None:
        viewer.update(object_param, pupil_param, [1.0], 0, pupil_true, initial_lrs)
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

            pred_amp = ptychoX_fourier(
                object_param,
                pupil_param,
                led_indices[batch_idx],
                use_cuda_kernels=args.use_cuda_kernels,
            )
            loss = F.mse_loss(pred_amp, observed_amp[batch_idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
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
            print(f"epoch {epoch:04d}  loss={epoch_loss:.6e}  lr=[{lr_text}]  time={elapsed_time_ms:.2f}s")

        if viewer is not None and (epoch == 1 or epoch % args.display_every == 0):
            viewer_alive = viewer.update(object_param, pupil_param, loss_history, epoch, pupil_true, current_lrs)
            if not viewer_alive:
                viewer = None

        if args.save_every > 0 and (epoch == args.epochs or epoch % args.save_every == 0):
            save_complex_reconstruction(object_param, pupil_param, out_dir, epoch, prefix="fpm")

        if stopped_early:
            break

    if not args.no_save:
        save_complex_reconstruction(object_param, pupil_param, out_dir, final_epoch, prefix="fpm")
        torch.save(
            {
                "object": object_param.detach().cpu(),
                "pupil": pupil_param.detach().cpu(),
                "led_indices_xy": led_indices.detach().cpu(),
                "observed_amplitude": observed_amp.detach().cpu(),
                "args": vars(args),
            },
            out_dir / "fpm_reconstruction.pt",
        )

    if args.keep_viewer_open and isinstance(viewer, ViserReconstructionViewer):
        print("Training finished. Press Ctrl+C to stop the viser server.")
        try:
            viewer.server.sleep_forever()
        except KeyboardInterrupt:
            viewer.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch Fourier ptychography reconstruction demo.")
    parser.add_argument("--low-res", type=int, default=64)
    parser.add_argument("--pratio", type=int, default=4)
    parser.add_argument("--samples", type=int, default=225)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--pupil-lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--aberration-scale", type=float, default=0.8, help="Scale of synthetic Zernike pupil phase.")
    parser.add_argument("--no-aberration", action="store_true", help="Disable synthetic Zernike pupil aberration.")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-cuda-kernels", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10000)
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
        default="matlab/Examples/Fourier ptychography/dataset",
        help="Directory containing cameraman512.png and I01.bmp.",
    )
    parser.add_argument("--out-dir", type=str, default="python/ptychography/runs_fpm")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
