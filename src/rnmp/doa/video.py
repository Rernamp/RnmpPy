"""
Video of the DOA tracking: the MUSIC spectrum on the sphere per window + azimuth/elevation
tracks with a time cursor, with the recording's stereo sound, encoded by ffmpeg into an mp4 (H.264 + AAC).

Frame k shows the window centered at times[k] and lasts one window step, so the video runs in real time
and the sound lines up with the estimates (an estimate averages its window, so it lags by ~half a window).
Frames are drawn with matplotlib (Agg, no GUI) and piped to ffmpeg as raw RGB, without temporary images.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.io import wavfile

from rnmp.doa.music import MIC_LAYOUT, direction_vectors


def stereo_mix(samples, channels):
    """(n, 2): left = mean of the selected mics left of the array center (x < 0), right = the ones right of it.
    A side without mics gets the mean of all of them. Normalized to a 0.9 peak."""
    left = [ch for ch in channels if MIC_LAYOUT[ch][0] < 0]
    right = [ch for ch in channels if MIC_LAYOUT[ch][0] > 0]
    mono = samples[:, channels].mean(axis=1)
    stereo = np.stack([
        samples[:, left].mean(axis=1) if left else mono,
        samples[:, right].mean(axis=1) if right else mono,
    ], axis=1)
    peak = np.abs(stereo).max()
    return stereo * (0.9 / peak) if peak > 0 else stereo


class _FrameRenderer:
    """One matplotlib figure, updated in place frame by frame: only colors and markers change."""

    def __init__(self, azimuths_deg, elevations_deg, times, peak_az, peak_el, valid, vmin, size_px, dpi):
        self.vmin = vmin
        self.cmap = matplotlib.colormaps["viridis"]
        self.fig = Figure(figsize=(size_px[0] / dpi, size_px[1] / dpi), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        grid = self.fig.add_gridspec(2, 2, width_ratios=[1.15, 1], left=0.01, right=0.97, top=0.9, bottom=0.09,
                                     wspace=0.12, hspace=0.3)

        # sphere: upper hemisphere, seam at +-180 closed by repeating the first azimuth
        # computed_zorder=False: the peak marker is drawn over the surface, not hidden in it
        ax = self.fig.add_subplot(grid[:, 0], projection="3d", computed_zorder=False)
        u = direction_vectors(np.append(azimuths_deg, azimuths_deg[0] + 360), elevations_deg)
        self.surface = ax.plot_surface(u[..., 0], u[..., 1], u[..., 2], rstride=1, cstride=1, shade=False,
                                       antialiased=False, linewidth=0, color="gray", zorder=1)
        (self.ray,) = ax.plot([0, 0], [0, 0], [0, 0], color="red", linewidth=4, zorder=3)
        (self.dot,) = ax.plot([0], [0], [0], "o", color="red", markeredgecolor="white", markeredgewidth=2,
                              markersize=12, zorder=4)
        ring = np.deg2rad(np.arange(0, 361, 5))
        ax.plot(np.sin(ring), -np.cos(ring), 0 * ring, color="gray", linewidth=0.8, zorder=2)  # horizon
        for x, y, label in [(0, -1.3, "0° user"), (1.3, 0, "90°"), (0, 1.3, "180°"), (-1.3, 0, "-90°")]:
            ax.text(x, y, 0, label, color="dimgray", ha="center", zorder=5)
        ax.text(0, 0, 1.25, "up", color="dimgray", ha="center", zorder=5)
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(0, 1.1))
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 0.5), zoom=1.35)
        view_elev, view_azim = np.deg2rad(30), np.deg2rad(-60)
        ax.view_init(elev=np.rad2deg(view_elev), azim=np.rad2deg(view_azim))
        # toward the camera; a peak on the far side is drawn faded, it is seen through the sphere
        self.camera = np.array([np.cos(view_elev) * np.cos(view_azim), np.cos(view_elev) * np.sin(view_azim),
                                np.sin(view_elev)])
        self.fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(vmin, 0), self.cmap),
                          ax=ax, shrink=0.5, pad=0.0, label="dB")

        # tracks of the first source, with a time cursor
        self.cursors = []
        for row, (values, label, limits) in enumerate([
            (peak_az[:, 0], "azimuth, deg", (-180, 180)), (peak_el[:, 0], "elevation, deg", (0, 90)),
        ]):
            track = self.fig.add_subplot(grid[row, 1])
            track.scatter(times[valid], values[valid], s=6, color="tab:blue")
            track.scatter(times[~valid], values[~valid], s=4, color="lightgray")
            track.set(xlim=(times[0], times[-1]), ylim=limits, ylabel=label)
            track.grid(alpha=0.3)
            self.cursors.append(track.axvline(times[0], color="red"))
        track.set_xlabel("time, s")
        self.title = self.fig.suptitle("")

    def render(self, spectrum_dB, peaks, time, is_valid):
        """RGB bytes of the frame."""
        closed = np.concatenate([spectrum_dB, spectrum_dB[:, :1]], axis=1)
        # plot_surface makes one face per grid cell in row-major order, colored by its first corner
        face_dB = np.clip(closed[:-1, :-1], self.vmin, 0).ravel()
        self.surface.set_facecolor(self.cmap((face_dB - self.vmin) / -self.vmin))

        if peaks:
            p_az, p_el = peaks[0]
            u = direction_vectors(np.array([p_az]), np.array([p_el]))[0, 0]
            self.ray.set_data_3d([u[0], 1.35 * u[0]], [u[1], 1.35 * u[1]], [u[2], 1.35 * u[2]])
            self.dot.set_data_3d([u[0]], [u[1]], [u[2]])
            text = f"azimuth {p_az:.0f}°, elevation {p_el:.0f}°"
            alpha = 1.0 if u @ self.camera > -0.1 else 0.35
        else:
            text, alpha = "no peak", 0.0
        color = "red" if is_valid else "lightgray"
        for artist in (self.ray, self.dot):
            artist.set_color(color)
            artist.set_alpha(alpha)
        for cursor in self.cursors:
            cursor.set_xdata([time, time])
        self.title.set_text(f"t = {time:6.2f} s    {text}" + ("" if is_valid else "    (filtered out)"))

        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].tobytes()


def render_video(path, spectra_dB, azimuths_deg, elevations_deg, peak_az, peak_el, times, valid, step,
                 audio, sample_rate, audio_start=0.0, vmin=-30.0, display_step_deg=2, size_px=(1280, 640), dpi=100,
                 progress=lambda frames: frames):
    """Write the mp4.

    spectra_dB (W, n_el, n_az), peak_az / peak_el (W, K), times (W,) - window centers, s, valid (W,) bool,
    step - window step, s (= frame duration); audio (n, 2) float in [-1, 1], its first sample at audio_start, s.
    display_step_deg - the sphere is drawn on a grid not finer than this. progress wraps the frame iterator
    (e.g. mo.status.progress_bar).
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not found in PATH")
    stride = max(1, round(display_step_deg / (azimuths_deg[1] - azimuths_deg[0])))
    renderer = _FrameRenderer(azimuths_deg[::stride], elevations_deg[::stride], times, peak_az, peak_el, valid,
                              vmin, size_px, dpi)

    # video time 0 = start of the first frame = half a step before the first window center
    first = round((times[0] - step / 2 - audio_start) * sample_rate)
    count = round(len(times) * step * sample_rate)
    clip = audio[max(first, 0) : max(first, 0) + count]

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "audio.wav"
        wavfile.write(audio_path, sample_rate, (np.clip(clip, -1, 1) * 32767).astype(np.int16))
        log_path = Path(tmp) / "ffmpeg.log"
        with open(log_path, "wb") as log:
            encoder = subprocess.Popen([
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size_px[0]}x{size_px[1]}",
                "-framerate", f"{1 / step:.6f}", "-i", "-",
                "-i", str(audio_path),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k", "-shortest",
                str(path),
            ], stdin=subprocess.PIPE, stderr=log)
            try:
                for k in progress(range(len(times))):
                    peaks = [(a, e) for a, e in zip(peak_az[k], peak_el[k]) if not np.isnan(a)]
                    encoder.stdin.write(renderer.render(spectra_dB[k, ::stride, ::stride], peaks, times[k], valid[k]))
            finally:
                encoder.stdin.close()
                code = encoder.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed ({code}): {log_path.read_text(errors='replace')}")
    return Path(path)
