"""H.264 encoding with ffmpeg (bundled by imageio-ffmpeg).

Clips are first written losslessly while they are captured (libx264rgb, crf 0), then encoded to the final MP4 from that
intermediate. That way the size/quality knob can be turned and the final files regenerated without recapturing anything.
"""
from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def fps_for(loop_ms: float, target_step_ms: float = 64.0, min_frames: int = 12) -> tuple[int, Fraction]:
    """Number of frames and exact frame rate so that `frames / fps == loop_ms`: the clip loops with no phase error."""
    n = max(min_frames, round(loop_ms / target_step_ms))
    return n, Fraction(n * 1000, int(round(loop_ms))).limit_denominator(1000)


class LosslessWriter:
    def __init__(self, path: Path, width: int, height: int, fps: Fraction):
        self.path, self.w, self.h = path, width, height
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "-", "-c:v", "libx264rgb", "-crf", "0",
             "-preset", "ultrafast", str(path)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        assert frame.shape == (self.h, self.w, 3), (frame.shape, self.w, self.h)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
        if self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}")


def finalize(lossless: Path, out: Path, crf: int = 22) -> int:
    """Lossless intermediate -> PowerPoint-friendly MP4 (H.264 yuv420p, bt709, no audio, fast start). Returns bytes."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(lossless),
           "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p",
           "-c:v", "libx264", "-preset", "slow", "-tune", "animation", "-crf", str(crf), "-g", "60",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-an", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out.stat().st_size
