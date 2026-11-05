"""Capture one whole slide: every build state as a static picture plus the looping clips of its animated regions.

Per state:  analyse -> static frame, moving regions, loop length of each region
            record  -> one loop of every region (same page for SMIL/CSS slides, whose time can be rewound; a replayed
                       fresh page for JavaScript-driven slides, whose clock cannot)
Across states the pictures are turned into patches (only the pixels that changed since the previous state), and clips are
classified as new / kept (identical to a clip already playing) / obsolete (to be stopped and hidden).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import regions as R
from . import video
from .slides import Slide

PATCH_THRESHOLD = 6
FULL_PATCH_FRACTION = 0.55

# Title slides whose text sits on top of a full-bleed animated background. The background is captured without the text
# (so the video carries no baked-in lettering) and each text element becomes a crisp transparent PNG on top of it.
# (css selector, entrance): entrance is None (already there) or (delay_s, duration_s) of a PowerPoint Fade that
# reproduces the page's own `t-in` entrance.
TEXT_LAYERS = {
    "02.SNN/01.Teo/01-title.html": [(".title", None), (".subtitle", None)],
    "03.GNN/01-title-gnn.html": [(".title", (0.0, 1.0)), (".subtitle", (0.45, 1.0))],
}

# JavaScript loops that cannot repeat seamlessly as written, retimed in the captured copy only (listed in the report).
# Message passing: the aggregation phase loops every 5.2 s but its ring is dashed ("8 7" = 15 px) and marches at 16 px/s, which is
# 5.55 dash cycles per loop. At 90/83.2 times that speed (+8 %) it is exactly 6 cycles, so the loop closes without a jump.
JS_RETIME = {
    "03.GNN/06-rrr-neural-messrrrr-age-passing.html": (
        "(() => { const set = Element.prototype.setAttribute;"
        " Element.prototype.setAttribute = function (n, v) {"
        "   if (n === 'stroke-dashoffset') v = String(parseFloat(v) * (90 / 83.2));"
        "   return set.call(this, n, v); }; })();"),
}


@dataclass
class Clip:
    id: str
    box: tuple[int, int, int, int]      # device pixels (x, y, w, h), even-aligned
    loop_ms: int
    frames: int
    quality: str
    lossless: Path | None = None
    mp4: Path | None = None
    frame0: np.ndarray | None = None
    mp4_bytes: int = 0


@dataclass
class State:
    b: int
    static: np.ndarray
    clips: list[Clip] = field(default_factory=list)
    retime: list | None = None            # [{'box': css box, 'map': {old_s: new_s}}] applied before recording
    notes: list[str] = field(default_factory=list)
    overlays: list[dict] = field(default_factory=list)   # transparent text PNGs: {png, box, alt, entrance}


def uses_js_loops(slide: Slide) -> bool:
    return slide.runtime == "tia" or "requestAnimationFrame" in slide.path.read_text(encoding="utf-8")


def _record_into(s, state: State, work: Path, scale: float):
    """Write one loop of every clip of `state` (lossless intermediates) from an already settled session."""
    writers = {}
    schedule: dict[int, list[tuple[Clip, int]]] = {}
    try:
        for c in state.clips:
            n, fps = video.fps_for(c.loop_ms)
            c.frames = n
            c.lossless = work / f"{c.id}_lossless.mp4"
            writers[c.id] = video.LosslessWriter(c.lossless, c.box[2], c.box[3], fps)
            for j in range(n):
                schedule.setdefault(round(j * c.loop_ms / n), []).append((c, j))
        for t in sorted(schedule):
            s.advance_to(t)
            jobs = schedule[t]
            full = s.shot() if len(jobs) > 1 else None
            for c, j in jobs:
                x, y, w, h = c.box
                fr = full[y:y + h, x:x + w] if full is not None else s.shot(clip=(x / scale, y / scale, w / scale, h / scale))
                if j == 0:
                    c.frame0 = fr.copy()
                writers[c.id].write(fr)
    finally:
        for w in writers.values():
            w.close()


def smil_periods(slide: Slide) -> list[int]:
    import re
    return sorted({round(float(x) * 1000) for x in re.findall(r'dur="([0-9.]+)s"', slide.path.read_text(encoding='utf-8'))})


def analyse(browser, slide: Slide, b: int, scale: float, work: Path, crf: int, slide_plan=None) -> State:
    js = uses_js_loops(slide)
    layers = TEXT_LAYERS.get(slide.rel)
    hide = ", ".join(sel for sel, _ in layers) if layers else None
    s = browser.session(slide.url, js=js, init_js=JS_RETIME.get(slide.rel))
    try:
        s.press(b)
        s.settle()
        if hide:
            s.hide(hide)
        static = s.shot()
        fh, fw = static.shape[:2]
        state = State(b, static)
        declared = s.declared()
        if not js and not any(d.get("visible", True) for d in declared):
            return state                                  # nothing visible can move: a purely static state
        step, n_probe = (200.0, 90) if js else (250.0, 32)
        frames = [s.shot(css_scale=True)]
        for i in range(1, n_probe + 1):
            s.advance_to(i * step)
            frames.append(s.shot(css_scale=True))
        for box in R.find_regions(R.moving_mask(frames), s.svgs()):
            ds = R.declared_for(box, [d for d in declared if d.get("visible", True)])
            emp, err = (R.empirical_loop([f[int(box.y):int(box.y1), int(box.x):int(box.x1)] for f in frames], step)
                        if js else (None, 1.0))
            if ds:
                loop, quality, rmap = R.choose_loop(ds, slide_plan)
                if rmap:
                    state.retime = (state.retime or []) + [{"box": [box.x, box.y, box.w, box.h], "map": rmap}]
                    state.notes.append(f"region {int(box.x)},{int(box.y)}: SMIL periods {sorted(set(ds))} ms nudged to {loop} ms")
                if emp and all(abs(round(emp / d) * d - emp) / emp < 0.03 for d in ds):
                    loop, quality = emp, "empirical+declared"
                elif emp and err < 0.35:
                    loop, quality = emp, "empirical"
            elif emp:
                loop, quality = emp, "empirical"
            else:
                loop, quality = R.MAX_LOOP_MS, "capped-seam"
            state.clips.append(Clip(f"s{b}c{len(state.clips) + 1}", box.device(scale, fw, fh), int(loop), 0, quality))
        if s.errors:
            state.notes.append("page errors: " + "; ".join(s.errors[:3]))
        if state.clips and not js:                        # SMIL/CSS time can be rewound: record in this very page
            if state.retime:
                s.retime_regions(state.retime)
            s.advance_to(0)
            _record_into(s, state, work, scale)
            return state
    finally:
        s.close()
    if state.clips:                                       # JavaScript-driven: replay on a fresh page, same key presses
        s = browser.session(slide.url, js=True, init_js=JS_RETIME.get(slide.rel))
        try:
            s.press(b)
            s.settle()
            if hide:
                s.hide(hide)
            if state.retime:
                s.retime_regions(state.retime)
            _record_into(s, state, work, scale)
        finally:
            s.close()
    return state


def capture_text_layers(browser, slide: Slide, state: State, scale: float, work: Path):
    """One transparent PNG per text element listed in TEXT_LAYERS, cropped to the pixels it actually paints (halo included)."""
    s = browser.session(slide.url, js=uses_js_loops(slide))
    try:
        s.press(state.b)
        s.settle()
        for k, (sel, entrance) in enumerate(TEXT_LAYERS[slide.rel], 1):
            texts = s.isolate(sel)
            if not texts:
                continue
            rgba = s.shot_rgba()
            ys, xs = np.where(rgba[:, :, 3] > 2)
            if len(xs) == 0:
                continue
            x0, y0 = max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2)
            x1, y1 = min(rgba.shape[1], int(xs.max()) + 3), min(rgba.shape[0], int(ys.max()) + 3)
            png = work / f"s{state.b}_text{k}.png"
            Image.fromarray(rgba[y0:y1, x0:x1], "RGBA").save(png, optimize=True)
            state.overlays.append({"png": png.name, "box": [x0, y0, x1 - x0, y1 - y0], "alt": " ".join(texts), "entrance": entrance})
        s.isolate(None)
    finally:
        s.close()


def finalize_clips(state: State, work: Path, crf: int):
    for c in state.clips:
        c.mp4 = work / f"{c.id}.mp4"
        c.mp4_bytes = video.finalize(c.lossless, c.mp4, crf)


def _same_clip(a: Clip, b: Clip) -> bool:
    if a.frame0 is None or b.frame0 is None or a.frame0.shape != b.frame0.shape:
        return False
    if any(abs(p - q) > 4 for p, q in zip(a.box, b.box)) or abs(a.loop_ms - b.loop_ms) > 0.02 * a.loop_ms:
        return False
    return float(np.abs(a.frame0.astype(np.int16) - b.frame0.astype(np.int16)).mean()) < 1.5


def _boxes_mask(shape, boxes, pad=0):
    m = np.zeros(shape[:2], np.uint8)
    for x, y, w, h in boxes:
        m[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] = 1
    return m


def plan_states(states: list[State]) -> list[dict]:
    """Turn captured states into patches and kept/new/obsolete clips, ready for the PowerPoint builder."""
    out, playing = [], []
    for st in states:
        h, w = st.static.shape[:2]
        entry = {"b": st.b, "patches": [], "base": st.b == 0}
        new, keep = [], []
        for c in st.clips:
            match = next((p for p in playing if _same_clip(p, c)), None)
            if match:
                keep.append(match)
            else:
                new.append(c)
        stop = [p for p in playing if p not in keep]
        if st.b > 0:
            prev = states[st.b - 1].static
            diff = (np.abs(st.static.astype(np.int16) - prev.astype(np.int16)).max(axis=2) > PATCH_THRESHOLD).astype(np.uint8)
            diff |= _boxes_mask(st.static.shape, [p.box for p in stop])     # a hidden clip exposes older pixels: re-cover them
            mask = diff & (1 - _boxes_mask(st.static.shape, [c.box for c in new + keep]))
            mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            boxes = R._merge([R.Box(*map(float, stats[i][:4])).padded(2, w, h) for i in range(1, n) if stats[i][4] >= 16])
            if sum(b.area for b in boxes) > FULL_PATCH_FRACTION * w * h:
                boxes = [R.Box(0, 0, w, h)]
            entry["patches"] = [[int(b.x), int(b.y), int(b.w), int(b.h)] for b in boxes]
        entry.update(new=new, keep=[c.id for c in keep], stop=[c.id for c in stop])
        playing = keep + new
        out.append(entry)
    return out


def cache_key(slide: Slide, scale: float, crf: int) -> str:
    """Changes when the slide, its runtime files or any capture code changes: then the slide is captured again."""
    import hashlib
    h = hashlib.sha1()
    here = Path(__file__).parent
    for f in [slide.path, slide.path.parent / 'slideui.js', slide.path.parent / 'slideui.css',
              *(here / n for n in ('capture.py', 'regions.py', 'video.py', 'slide_capture.py', 'slides.py'))]:
        if f.exists():
            h.update(f.read_bytes())
    h.update(f'{scale}:{crf}'.encode())
    return h.hexdigest()[:16]


def capture_slide(browser, slide: Slide, out_dir: Path, scale: float = 1.25, crf: int = 18) -> dict:
    work = out_dir / f"slide_{slide.index:02d}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    slide_plan = R.retime_plan(smil_periods(slide))
    states = [analyse(browser, slide, b, scale, work, crf, slide_plan) for b in range(slide.states)]
    if slide.rel in TEXT_LAYERS:
        for st in states:
            capture_text_layers(browser, slide, st, scale, work)
    plan = plan_states(states)
    used = {c.id for e in plan for c in e["new"]}
    for st in states:                                     # encode only the clips that are actually referenced
        st.clips = [c for c in st.clips if c.id in used]
        finalize_clips(st, work, crf)
    for f in work.glob("*_lossless.mp4"):
        f.unlink()
    manifest = {"key": cache_key(slide, scale, crf), "index": slide.index, "rel": slide.rel, "title": slide.title, "runtime": slide.runtime,
                "clicks": slide.clicks, "scale": scale, "size": [states[0].static.shape[1], states[0].static.shape[0]], "states": []}
    for st, e in zip(states, plan):
        pics = []
        if e["base"]:
            p = work / f"s{st.b}_base.png"
            Image.fromarray(st.static).save(p, optimize=True)
            pics.append({"png": p.name, "box": [0, 0, st.static.shape[1], st.static.shape[0]]})
        for k, (x, y, w, h) in enumerate(e["patches"], 1):
            p = work / f"s{st.b}_p{k}.png"
            Image.fromarray(st.static[y:y + h, x:x + w]).save(p, optimize=True)
            pics.append({"png": p.name, "box": [x, y, w, h]})
        clips = [{"id": c.id, "mp4": c.mp4.name, "box": list(c.box), "loop_ms": c.loop_ms, "frames": c.frames,
                  "quality": c.quality, "bytes": c.mp4_bytes} for c in e["new"] if c.mp4]
        manifest["states"].append({"b": st.b, "pictures": pics, "clips": clips, "overlays": st.overlays, "keep": e["keep"], "stop": e["stop"], "notes": st.notes})
    (work / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest
