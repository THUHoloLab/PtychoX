"""PyTorch port of the classical PtychoX ptychography demo.

The MATLAB/CUDA implementation in this repository implements a custom
autodiff operator for the same forward map. This version expresses the model in
plain PyTorch complex tensors, so autograd supplies the backward pass.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ptychoX import ptychoX_base


class LiveReconstructionViewer:
    def __init__(self) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.fig, axes = plt.subplots(2, 2, num="PyTorch Ptychography Reconstruction", figsize=(10, 8))
        self.ax_amp, self.ax_phase, self.ax_probe, self.ax_loss = axes.ravel()
        self.im_amp = self.ax_amp.imshow(np.zeros((2, 2), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
        self.im_phase = self.ax_phase.imshow(np.zeros((2, 2), dtype=np.float32), cmap="twilight", vmin=-math.pi, vmax=math.pi)
        self.im_probe = self.ax_probe.imshow(np.zeros((2, 2), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
        (self.loss_line,) = self.ax_loss.plot([], [], color="tab:blue", linewidth=1.5)

        self.ax_amp.set_title("Object amplitude")
        self.ax_phase.set_title("Object phase")
        self.ax_probe.set_title("Probe amplitude")
        self.ax_loss.set_title("Training loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_yscale("log")
        for ax in (self.ax_amp, self.ax_phase, self.ax_probe):
            ax.set_xticks([])
            ax.set_yticks([])
        self.fig.tight_layout()
        self.fig.show()

    @torch.no_grad()
    def update(self, sample: torch.Tensor, probe: torch.Tensor, losses: list[float], epoch: int) -> bool:
        if not self.plt.fignum_exists(self.fig.number):
            return False

        amp = sample.abs().detach().float().cpu()
        amp = (amp / amp.max().clamp_min(1e-12)).clamp(0, 1).numpy()
        phase = torch.angle(sample.detach()).float().cpu().numpy()
        probe_amp = probe.abs().detach().float().cpu()
        probe_amp = (probe_amp / probe_amp.max().clamp_min(1e-12)).clamp(0, 1).numpy()

        self.im_amp.set_data(amp)
        self.im_phase.set_data(phase)
        self.im_probe.set_data(probe_amp)

        epochs = np.arange(1, len(losses) + 1)
        self.loss_line.set_data(epochs, np.asarray(losses, dtype=np.float64))
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        self.fig.suptitle(f"Epoch {epoch}")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)
        return True


class ViserReconstructionViewer:
    def __init__(self, host: str, port: int, panel_size: int) -> None:
        import viser

        self.panel_size = panel_size
        self.latest_images: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.paused = False
        self.stop_requested = False
        self.server = viser.ViserServer(host=host, port=port, label="Ptychography", verbose=True)
        self.server.initial_camera.position = (0.0, 0.0, 5.0)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)
        self.server.initial_camera.up = (0.0, 1.0, 0.0)
        self.server.initial_camera.fov = math.radians(45.0)

        self.status = self.server.gui.add_text("Status", "Waiting for first update", disabled=True, order=0)
        self.pause_toggle = self.server.gui.add_checkbox("Pause training", False, order=0.15)
        self.stop_button = self.server.gui.add_button("Stop after current batch", color="red", order=0.2)
        self.lr_scale = self.server.gui.add_slider(
            "LR scale",
            min=0.0,
            max=2.0,
            step=0.05,
            initial_value=1.0,
            order=0.3,
        )
        self.lr_status = self.server.gui.add_text("Effective LR", "", disabled=True, order=0.4)
        self.zoom = self.server.gui.add_slider(
            "Scene zoom",
            min=0.35,
            max=2.0,
            step=0.05,
            initial_value=1.0,
            order=0.5,
        )
        self.overview = self.server.scene.add_image(
            "/reconstruction_overview",
            np.zeros((panel_size * 2, panel_size * 2, 3), dtype=np.uint8),
            render_width=4.0,
            render_height=4.0,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            cast_shadow=False,
            receive_shadow=False,
        )
        self.zoom.on_update(lambda _: self.render_latest())
        self.pause_toggle.on_update(lambda event: self.set_paused(bool(event.target.value)))
        self.stop_button.on_click(lambda _: self.request_stop())

    @torch.no_grad()
    def update(
        self,
        sample: torch.Tensor,
        probe: torch.Tensor,
        losses: list[float],
        epoch: int,
        probe_true: torch.Tensor | None = None,
        current_lrs: list[float] | None = None,
    ) -> bool:
        gt_source = probe_true.abs() if probe_true is not None else torch.zeros_like(probe.abs())
        self.latest_images = (
            gray_rgb_uint8(sample.abs()),
            phase_rgb_uint8(sample),
            gray_rgb_uint8(probe.abs()),
            gray_rgb_uint8(gt_source),
        )
        self.render_latest()
        state = "paused" if self.paused else "running"
        self.status.value = f"epoch={epoch}, loss={losses[-1]:.6e}, {state}"
        if current_lrs is not None:
            self.lr_status.value = ", ".join(f"{lr:.3e}" for lr in current_lrs)
        self.update_loss_plot(losses)
        self.server.flush()
        return True

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.status.value = "paused" if paused else "running"
        self.server.flush()

    def request_stop(self) -> None:
        self.stop_requested = True
        self.status.value = "stop requested"
        self.server.flush()

    def render_latest(self) -> None:
        if self.latest_images is None:
            return
        panel_size = max(64, int(round(self.panel_size * float(self.zoom.value))))
        overview = make_viser_overview_from_images(*self.latest_images, panel_size=panel_size)
        self.overview.image = np.flipud(overview)
        self.overview.scale = float(self.zoom.value)
        self.server.flush()

    def update_loss_plot(self, losses: list[float]) -> None:
        if not hasattr(self, "loss_plot"):
            self.loss_plot = self.server.gui.add_image(
                np.zeros((180, 360, 3), dtype=np.uint8),
                label="Loss curve",
                order=2,
            )
        self.loss_plot.image = loss_plot_rgb_uint8(losses)

    def stop(self) -> None:
        self.server.stop()


def gray_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    x_cpu = x.detach().float().cpu()
    x_cpu = x_cpu - x_cpu.min()
    x_cpu = x_cpu / x_cpu.max().clamp_min(1e-12)
    img = (255.0 * x_cpu.clamp(0, 1)).byte().numpy()
    return np.repeat(img[..., None], 3, axis=-1)


def phase_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    phase = torch.angle(x.detach()).float().cpu().numpy()
    hue = (phase + math.pi) / (2.0 * math.pi)
    return hsv_to_rgb_uint8(hue, np.ones_like(hue), np.ones_like(hue))


def loss_plot_rgb_uint8(losses: list[float], width: int = 1400, height: int = 700) -> np.ndarray:
    canvas = Image.new("RGB", (width, height), "white")
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        tick_font = pil_font(66)
        margin_l, margin_r, margin_t, margin_b = 360, 90, 120, 170
        x0, y0 = margin_l, height - margin_b
        x1, y1 = width - margin_r, margin_t
        draw.rectangle((x0, y1, x1, y0), outline=(200, 200, 200), width=2)

        if len(losses) >= 1:
            values = np.log10(np.maximum(np.asarray(losses, dtype=np.float64), 1e-30))
            lo = float(values.min())
            hi = float(values.max())
            if abs(hi - lo) < 1e-12:
                hi = lo + 1.0
            xs = np.linspace(x0, x1, len(values))
            ys = y0 - (values - lo) / (hi - lo) * (y0 - y1)
            points = [(float(x), float(y)) for x, y in zip(xs, ys)]
            if len(points) == 1:
                x, y = points[0]
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(30, 110, 220))
            else:
                draw.line(points, fill=(30, 110, 220), width=4)
            for tick in np.linspace(lo, hi, 5):
                y_tick = y0 - (tick - lo) / (hi - lo) * (y0 - y1)
                draw.line((x0 - 8, y_tick, x0, y_tick), fill=(120, 120, 120), width=2)
                draw.text((30, y_tick - 40), compact_scientific(10**tick), fill=(70, 70, 70), font=tick_font)

            x_ticks = np.unique(np.rint(np.linspace(1, len(losses), min(5, len(losses)))).astype(np.int32))
            for tick_epoch in x_ticks:
                if len(losses) <= 1:
                    x_tick = x0
                else:
                    x_tick = x0 + (tick_epoch - 1) / (len(losses) - 1) * (x1 - x0)
                draw.line((x_tick, y0, x_tick, y0 + 8), fill=(120, 120, 120), width=2)
                label = str(int(tick_epoch))
                draw.text((x_tick - 27 * len(label), y0 + 44), label, fill=(70, 70, 70), font=tick_font)
    except Exception:
        pass
    return np.asarray(canvas, dtype=np.uint8)


def compact_scientific(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def make_viser_overview(
    sample: torch.Tensor,
    probe: torch.Tensor,
    probe_true: torch.Tensor | None,
    panel_size: int,
) -> np.ndarray:
    gt_source = probe_true.abs() if probe_true is not None else torch.zeros_like(probe.abs())
    return make_viser_overview_from_images(
        gray_rgb_uint8(sample.abs()),
        phase_rgb_uint8(sample),
        gray_rgb_uint8(probe.abs()),
        gray_rgb_uint8(gt_source),
        panel_size,
    )


def make_viser_overview_from_images(
    amp_image: np.ndarray,
    phase_image: np.ndarray,
    probe_image: np.ndarray,
    probe_gt_image: np.ndarray,
    panel_size: int,
) -> np.ndarray:
    title_h = max(64, int(round(panel_size * 0.12)))
    font_size = max(42, int(round(panel_size * 0.075)))
    gap = max(10, int(round(panel_size * 0.02)))

    amp_panel = labeled_panel(amp_image, "amplitude", panel_size, panel_size, title_h, font_size)
    phase_panel = labeled_panel(phase_image, "phase", panel_size, panel_size, title_h, font_size)
    probe_panel = labeled_panel(probe_image, "probe", panel_size, panel_size, title_h, font_size)
    gt_panel = labeled_panel(probe_gt_image, "probe_GT", panel_size, panel_size, title_h, font_size)

    top_row = np.concatenate([amp_panel, phase_panel], axis=1)
    bottom_row = np.concatenate([probe_panel, gt_panel], axis=1)
    spacer = np.full((gap, panel_size * 2, 3), 255, dtype=np.uint8)
    return np.concatenate([top_row, spacer, bottom_row], axis=0)


def labeled_panel(
    image: np.ndarray,
    title: str,
    width: int,
    height: int,
    title_h: int,
    font_size: int,
) -> np.ndarray:
    body_h = max(1, height - title_h)
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    image = resize_rgb_fit(image, width, body_h)
    y0 = title_h + (body_h - image.shape[0]) // 2
    x0 = (width - image.shape[1]) // 2
    panel[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1], :] = image

    canvas = Image.fromarray(panel)
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, width, title_h - 1), fill=(245, 245, 245))
        draw.text((16, max(4, (title_h - font_size) // 2)), title, fill=(20, 20, 20), font=pil_font(font_size))
    except Exception:
        pass
    return np.asarray(canvas)


def resize_rgb_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / max(1, src_w), height / max(1, src_h))
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))
    pil = Image.fromarray(image)
    pil = pil.resize((out_w, out_h), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def pil_font(size: int) -> Image.Image:
    from PIL import ImageFont

    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def hsv_to_rgb_uint8(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = np.choose(i % 6, [v, q, p, p, t, v])
    g = np.choose(i % 6, [t, v, v, q, p, p])
    b = np.choose(i % 6, [p, p, t, v, v, q])
    return (255.0 * np.stack([r, g, b], axis=-1).clip(0, 1)).astype(np.uint8)


def terminal_link(url: str, label: str | None = None) -> str:
    """Return a clickable terminal hyperlink when the terminal supports OSC 8."""
    label = label or url
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return f"{label} ({url})"
    return f"\033]8;;{url}\033\\{label}\033]8;;\033\\"


def complex_abs_to_uint8(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu()
    x = x - x.min()
    x = x / x.max().clamp_min(1e-12)
    return Image.fromarray((255.0 * x).byte().numpy())


def phase_to_uint8(x: torch.Tensor) -> Image.Image:
    x = torch.angle(x.detach().cpu())
    x = (x + math.pi) / (2.0 * math.pi)
    return Image.fromarray((255.0 * x.clamp(0, 1)).byte().numpy())


def read_gray(path: Path, size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.BICUBIC)
    data = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).to(device=device, dtype=torch.float32)
    return data / 255.0


def mat2gray(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)


def make_scan_positions(
    pix: int,
    mag: int,
    samples: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    object_size = pix * mag
    low = round(pix / 7)
    high = object_size - pix - low
    if high <= low:
        raise ValueError("The object is too small for the requested probe size.")

    positions = torch.randint(
        low=low,
        high=high + 1,
        size=(samples, 2),
        generator=generator,
        device=device,
    )
    return positions.to(torch.long)


def make_probe(pix: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    length = 400.0
    fx_1d = torch.arange(-pix // 2, pix // 2, device=device, dtype=torch.float32) / length
    fy, fx = torch.meshgrid(fx_1d, fx_1d, indexing="ij")

    wavelength = 0.532
    k = 2.0 * math.pi / wavelength
    fz_arg = 1.0 - wavelength**2 * (fx.square() + fy.square())
    fz = torch.sqrt(fz_arg.clamp_min(0.0))

    aperture = ((fx.square() + fy.square()) < (40.0 / length) ** 2).float()
    p0 = torch.rand((1, 1, 32, 32), generator=generator, device=device)
    p0 = F.interpolate(p0, size=(pix, pix), mode="area")[0, 0]

    field = aperture * p0
    prop = torch.fft.ifftshift(
        torch.fft.fftshift(torch.fft.fft2(field)) * torch.exp(1j * k * 860.0 * fz)
    )
    prop = torch.fft.ifft2(prop)
    return prop.to(torch.complex64) * (prop.abs() > 0.1)


def ptychography_forward(
    sample: torch.Tensor,
    probe: torch.Tensor,
    positions: torch.Tensor,
    use_cuda_kernels: bool = False,
) -> torch.Tensor:
    """Return shifted Fourier amplitudes for zero-based `(x, y)` positions."""
    return ptychoX_base(sample, probe, positions, use_cuda_kernels=use_cuda_kernels)


@torch.no_grad()
def synthesize_dataset(
    object_true: torch.Tensor,
    probe_true: torch.Tensor,
    positions: torch.Tensor,
    batch_size: int,
    use_cuda_kernels: bool = False,
) -> torch.Tensor:
    chunks = []
    for start in range(0, positions.shape[0], batch_size):
        chunks.append(
            ptychography_forward(
                object_true,
                probe_true,
                positions[start : start + batch_size],
                use_cuda_kernels=use_cuda_kernels,
            )
        )
    return torch.cat(chunks, dim=0)


def save_reconstruction(sample: torch.Tensor, probe: torch.Tensor, out_dir: Path, epoch: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    complex_abs_to_uint8(sample.abs().clamp(max=1.0)).save(out_dir / f"object_amp_epoch_{epoch:04d}.png")
    phase_to_uint8(sample).save(out_dir / f"object_phase_epoch_{epoch:04d}.png")
    complex_abs_to_uint8(probe.abs()).save(out_dir / f"probe_amp_epoch_{epoch:04d}.png")


def cosine_lrs(base_lrs: list[float], min_lr: float, epochs: int, epoch: int, scale: float = 1.0) -> list[float]:
    if epochs <= 1:
        phase = 1.0
    else:
        phase = (1.0 + math.cos(math.pi * (epoch - 1) / epochs)) * 0.5
    return [(min_lr + (base_lr - min_lr) * phase) * scale for base_lr in base_lrs]


def set_optimizer_lrs(optimizer: torch.optim.Optimizer, lrs: list[float]) -> None:
    for group, lr in zip(optimizer.param_groups, lrs):
        group["lr"] = lr


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    source_dir = Path(args.source_dir)
    object_size = args.pix * args.mag

    img_amp = read_gray(source_dir / "cameraman512.png", object_size, device)
    img_phase = read_gray(source_dir / "I01.bmp", object_size, device)
    object_true = mat2gray(img_amp + 0.2).to(torch.complex64) * torch.exp(1j * math.pi * img_phase)
    probe_true = make_probe(args.pix, device, generator)

    positions = make_scan_positions(args.pix, args.mag, args.samples, device, generator)
    observed_amp = synthesize_dataset(
        object_true,
        probe_true,
        positions,
        args.batch_size,
        use_cuda_kernels=args.use_cuda_kernels,
    )

    sample_param = torch.nn.Parameter(torch.rand((object_size, object_size), device=device, generator=generator).to(torch.complex64))
    probe_param = torch.nn.Parameter(
        (((probe_true.abs() > 0).float()).to(torch.complex64)).clone()
    )

    optimizer = torch.optim.Adam(
        [
            {"params": [sample_param], "lr": args.lr},
            {"params": [probe_param], "lr": args.probe_lr if args.probe_lr is not None else args.lr},
        ],
        betas=(0.9, 0.999),
        eps=1e-16,
    )
    initial_lrs = [group["lr"] for group in optimizer.param_groups]

    out_dir = Path(args.out_dir)
    print(f"device={device}, pix={args.pix}, object={object_size}x{object_size}, samples={args.samples}")
    viewer = None
    if args.live:
        args.viewer = "matplotlib"
    if args.viewer == "matplotlib":
        viewer = LiveReconstructionViewer()
    elif args.viewer == "viser":
        viewer = ViserReconstructionViewer(args.viser_host, args.viser_port, args.viser_panel_size)
        viewer_url = f"http://localhost:{args.viser_port}"
        print("")
        print("Viewer running:")
        print(f"  {terminal_link(viewer_url)}")
        print("")
    loss_history: list[float] = []

    stopped_early = False
    final_epoch = 0

    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        order = torch.randperm(args.samples, generator=generator, device=device)
        epoch_loss = 0.0
        progress = 0

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

            pred_amp = ptychography_forward(
                sample_param,
                probe_param,
                positions[batch_idx],
                use_cuda_kernels=args.use_cuda_kernels,
            )
            loss = F.mse_loss(pred_amp, observed_amp[batch_idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item() * batch_idx.numel()

        if progress == 0:
            break

        epoch_loss /= progress
        loss_history.append(epoch_loss)
        scale = float(viewer.lr_scale.value) if isinstance(viewer, ViserReconstructionViewer) else 1.0
        current_lrs = cosine_lrs(initial_lrs, args.min_lr, args.epochs, epoch, scale)
        if epoch == 1 or epoch % args.log_every == 0:
            lr_text = ", ".join(f"{lr:.3e}" for lr in current_lrs)
            print(f"epoch {epoch:04d}  loss={epoch_loss:.6e}  lr=[{lr_text}]")

        if viewer is not None and (epoch == 1 or epoch % args.display_every == 0):
            if isinstance(viewer, ViserReconstructionViewer):
                viewer_alive = viewer.update(sample_param, probe_param, loss_history, epoch, probe_true, current_lrs)
            else:
                viewer_alive = viewer.update(sample_param, probe_param, loss_history, epoch)
            if not viewer_alive:
                viewer = None

        if args.save_every > 0 and (epoch == args.epochs or epoch % args.save_every == 0):
            save_reconstruction(sample_param, probe_param, out_dir, epoch)

        if stopped_early:
            break

    save_reconstruction(sample_param, probe_param, out_dir, final_epoch)
    torch.save(
        {
            "sample": sample_param.detach().cpu(),
            "probe": probe_param.detach().cpu(),
            "positions_xy": positions.detach().cpu(),
            "observed_amplitude": observed_amp.detach().cpu(),
            "args": vars(args),
        },
        out_dir / "reconstruction.pt",
    )
    if args.keep_viewer_open and isinstance(viewer, ViserReconstructionViewer):
        print("Training finished. Press Ctrl+C to stop the viser server.")
        try:
            viewer.server.sleep_forever()
        except KeyboardInterrupt:
            viewer.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch ptychography reconstruction demo.")
    parser.add_argument("--pix", type=int, default=128, help="Probe/diffraction frame width.")
    parser.add_argument("--mag", type=int, default=4, help="Object size multiplier.")
    parser.add_argument("--samples", type=int, default=225, help="Number of scan positions.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--probe-lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=1e-4, help="Minimum learning rate for cosine annealing.")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", type=str, default=None, help="Example: cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--use-cuda-kernels",
        action="store_true",
        help="Use the optional PyTorch CUDA extension for crop/scatter kernels.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--live", action="store_true", help="Open a MATLAB-like live reconstruction window.")
    parser.add_argument(
        "--viewer",
        choices=("none", "matplotlib", "viser"),
        default="none",
        help="Live viewer backend. Use --live as a shortcut for matplotlib.",
    )
    parser.add_argument("--display-every", type=int, default=1, help="Refresh the live window every N epochs.")
    parser.add_argument("--viser-host", type=str, default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--viser-panel-size", type=int, default=512, help="Pixel size of each viser overview panel.")
    parser.add_argument("--keep-viewer-open", action="store_true", help="Keep the viser server alive after training.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="matlab/Examples/Ptychography/source",
        help="Directory containing cameraman512.png and I01.bmp.",
    )
    parser.add_argument("--out-dir", type=str, default="python/ptychography/runs")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
