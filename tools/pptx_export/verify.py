"""Checks on a built deck. Nothing here can prove playback: that only a real Slide Show shows.

  inspect  - read the XML PowerPoint wrote: effect types per slide, loops, media, click-to-pause triggers, transition
  compare  - export every slide through PowerPoint and compare it with the browser's final build state
"""
from __future__ import annotations

import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import win32com.client
from PIL import Image


def inspect(pptx: Path) -> list[dict]:
    out = []
    with zipfile.ZipFile(pptx) as z:
        names = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                       key=lambda n: int(re.search(r"(\d+)", n.split("/")[-1]).group(1)))
        for n in names:
            x = z.read(n).decode("utf-8")
            t = re.search(r"<p:timing>.*</p:timing>", x, re.S)
            t = t.group(0) if t else ""
            out.append({
                "slide": n.split("/")[-1],
                "click": t.count('nodeType="clickEffect"'), "with": t.count('nodeType="withEffect"'), "after": t.count('nodeType="afterEffect"'),
                "fade": t.count('filter="fade"'), "exit": t.count('presetClass="exit"'),
                "media_calls": t.count('presetClass="mediacall"'), "loops": t.count('repeatCount="indefinite"'),
                "interactive": t.count("interactiveSeq"), "videos": x.count("<p:video>") + x.count("<a:videoFile"),
                "transition": "<p:transition" in x,
            })
        media = [i for i in z.infolist() if i.filename.startswith("ppt/media/")]
    total_media = sum(i.file_size for i in media)
    return out, len(media), total_media


def compare(pptx: Path, expected: dict[int, np.ndarray], size: tuple[int, int]) -> list[dict]:
    """expected: {position in this deck (1-based): RGB array of the browser's final state}. Exports each slide via PowerPoint."""
    app = win32com.client.Dispatch("PowerPoint.Application")
    pre = app.Presentations.Count
    pres = app.Presentations.Open(str(pptx), ReadOnly=True, Untitled=False, WithWindow=False)
    results = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for pos, want in expected.items():
                p = Path(tmp) / f"s{pos}.png"
                pres.Slides(pos).Export(str(p), "PNG", size[0], size[1])
                got = np.asarray(Image.open(p).convert("RGB"))
                if got.shape != want.shape:
                    results.append({"slide": pos, "error": f"size {got.shape} vs {want.shape}"})
                    continue
                d = np.abs(got.astype(np.int16) - want.astype(np.int16)).max(axis=2)
                big = d > 24
                ys, xs = np.nonzero(big)
                results.append({"slide": pos, "mean_abs": round(float(d.mean()), 3), "pixels_over_24": int(big.sum()),
                                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if big.any() else None})
    finally:
        pres.Close()
        if pre == 0:
            app.Quit()
    return results
