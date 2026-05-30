from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class LiveReconstructionViewer:
    def __init__(self, title: str = "PyTorch PtychoX Reconstruction") -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.fig, axes = plt.subplots(2, 2, num=title, figsize=(10, 8))
        self.ax_amp, self.ax_phase, self.ax_probe, self.ax_loss = axes.ravel()
        self.im_amp = self.ax_amp.imshow(np.zeros((2, 2), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
        self.im_phase = self.ax_phase.imshow(
            np.zeros((2, 2), dtype=np.float32), cmap="twilight", vmin=-math.pi, vmax=math.pi
        )
        self.im_probe = self.ax_probe.imshow(np.zeros((2, 2), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
        (self.loss_line,) = self.ax_loss.plot([], [], color="tab:blue", linewidth=1.5)

        self.ax_amp.set_title("Object amplitude")
        self.ax_phase.set_title("Object phase")
        self.ax_probe.set_title("Probe/Pupil amplitude")
        self.ax_loss.set_title("Training loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_yscale("log")
        for ax in (self.ax_amp, self.ax_phase, self.ax_probe):
            ax.set_xticks([])
            ax.set_yticks([])
        self.fig.tight_layout()
        self.fig.show()

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
        if not self.plt.fignum_exists(self.fig.number):
            return False

        amp = normalized_float(sample.abs())
        phase = torch.angle(sample.detach()).float().cpu().numpy()
        probe_amp = normalized_float(probe.abs())

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
    def __init__(
        self,
        host: str,
        port: int,
        panel_size: int,
        title: str = "PtychoX",
        amp_label: str = "amplitude",
        phase_label: str = "phase",
        probe_label: str = "probe",
        probe_gt_label: str = "probe_GT",
    ) -> None:
        import viser

        self.panel_size = panel_size
        self.amp_label = amp_label
        self.phase_label = phase_label
        self.probe_label = probe_label
        self.probe_gt_label = probe_gt_label
        self.latest_images: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.paused = False
        self.stop_requested = False
        self.server = viser.ViserServer(host=host, port=port, label=title, verbose=True)
        self.server.initial_camera.position = (0.0, 0.0, 5.0)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)
        self.server.initial_camera.up = (0.0, 1.0, 0.0)
        self.server.initial_camera.fov = math.radians(45.0)

        self.status = self.server.gui.add_text("Status", "Waiting for first update", disabled=True, order=0)
        self.pause_toggle = self.server.gui.add_checkbox("Pause training", False, order=0.15)
        self.stop_button = self.server.gui.add_button("Stop after current batch", color="red", order=0.2)
        self.lr_scale = self.server.gui.add_slider(
            "LR scale", min=0.0, max=2.0, step=0.05, initial_value=1.0, order=0.3
        )
        self.lr_status = self.server.gui.add_text("Effective LR", "", disabled=True, order=0.4)
        self.zoom = self.server.gui.add_slider(
            "Scene zoom", min=0.35, max=2.0, step=0.05, initial_value=1.0, order=0.5
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
        overview = make_overview_from_images(
            *self.latest_images,
            panel_size=panel_size,
            labels=(self.amp_label, self.phase_label, self.probe_label, self.probe_gt_label),
        )
        self.overview.image = np.flipud(overview)
        self.overview.scale = float(self.zoom.value)
        self.server.flush()

    def update_loss_plot(self, losses: list[float]) -> None:
        if not hasattr(self, "loss_plot"):
            self.loss_plot = self.server.gui.add_image(
                np.zeros((180, 360, 3), dtype=np.uint8), label="Loss curve", order=2
            )
        self.loss_plot.image = loss_plot_rgb_uint8(losses)

    def stop(self) -> None:
        self.server.stop()


def terminal_link(url: str, label: str | None = None) -> str:
    import os

    label = label or url
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return f"{label} ({url})"
    return f"\033]8;;{url}\033\\{label}\033]8;;\033\\"


def normalized_float(x: torch.Tensor) -> np.ndarray:
    x_cpu = x.detach().float().cpu()
    x_cpu = x_cpu - x_cpu.min()
    x_cpu = x_cpu / x_cpu.max().clamp_min(1e-12)
    return x_cpu.clamp(0, 1).numpy()


def gray_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    img = (255.0 * normalized_float(x)).astype(np.uint8)
    return np.repeat(img[..., None], 3, axis=-1)


def phase_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    phase = torch.angle(x.detach()).float().cpu().numpy()
    hue = (phase + math.pi) / (2.0 * math.pi)
    return hsv_to_rgb_uint8(hue, np.ones_like(hue), np.ones_like(hue))


def complex_abs_to_uint8(x: torch.Tensor) -> Image.Image:
    return Image.fromarray((255.0 * normalized_float(x)).astype(np.uint8))


def phase_to_uint8(x: torch.Tensor) -> Image.Image:
    phase = torch.angle(x.detach().cpu())
    phase = (phase + math.pi) / (2.0 * math.pi)
    return Image.fromarray((255.0 * phase.clamp(0, 1)).byte().numpy())


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
                x_tick = x0 if len(losses) <= 1 else x0 + (tick_epoch - 1) / (len(losses) - 1) * (x1 - x0)
                draw.line((x_tick, y0, x_tick, y0 + 8), fill=(120, 120, 120), width=2)
                label = str(int(tick_epoch))
                draw.text((x_tick - 27 * len(label), y0 + 44), label, fill=(70, 70, 70), font=tick_font)
    except Exception:
        pass
    return np.asarray(canvas, dtype=np.uint8)


def compact_scientific(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def make_overview_from_images(
    amp_image: np.ndarray,
    phase_image: np.ndarray,
    probe_image: np.ndarray,
    probe_gt_image: np.ndarray,
    panel_size: int,
    labels: tuple[str, str, str, str] = ("amplitude", "phase", "probe", "probe_GT"),
) -> np.ndarray:
    title_h = max(64, int(round(panel_size * 0.12)))
    font_size = max(42, int(round(panel_size * 0.075)))
    gap = max(10, int(round(panel_size * 0.02)))

    amp_panel = labeled_panel(amp_image, labels[0], panel_size, panel_size, title_h, font_size)
    phase_panel = labeled_panel(phase_image, labels[1], panel_size, panel_size, title_h, font_size)
    probe_panel = labeled_panel(probe_image, labels[2], panel_size, panel_size, title_h, font_size)
    gt_panel = labeled_panel(probe_gt_image, labels[3], panel_size, panel_size, title_h, font_size)

    top_row = np.concatenate([amp_panel, phase_panel], axis=1)
    bottom_row = np.concatenate([probe_panel, gt_panel], axis=1)
    spacer = np.full((gap, panel_size * 2, 3), 255, dtype=np.uint8)
    return np.concatenate([top_row, spacer, bottom_row], axis=0)


def labeled_panel(image: np.ndarray, title: str, width: int, height: int, title_h: int, font_size: int) -> np.ndarray:
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


def pil_font(size: int):
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


def save_complex_reconstruction(sample: torch.Tensor, probe: torch.Tensor, out_dir: Path, epoch: int, prefix: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    complex_abs_to_uint8(sample.abs().clamp(max=1.0)).save(out_dir / f"{stem}object_amp_epoch_{epoch:04d}.png")
    phase_to_uint8(sample).save(out_dir / f"{stem}object_phase_epoch_{epoch:04d}.png")
    complex_abs_to_uint8(probe.abs()).save(out_dir / f"{stem}probe_amp_epoch_{epoch:04d}.png")

