"""Check a built deck and write dist/report.md and dist/contact_sheet.png.

    python -m tools.pptx_export.report            # compares dist/deck.pptx with the browser, every slide

What is checked here: the saved XML (clicks, loops, no click-to-pause triggers), the file size, and the last build state of each
slide as PowerPoint draws it against the same state in the browser. What cannot be checked here: real playback in Slide Show.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import slides as deck
from .export import DIST, WORK

SIZE_CAP_MB = 60.0
THUMB_W, THUMB_H, COLS = 384, 216, 6
MEAN_LIMIT, OVER60_LIMIT = 3.0, 1500


def _final_frame(index: int, scale: float) -> tuple[int, np.ndarray]:
    from .capture import Browser_
    from .slide_capture import uses_js_loops

    slide = deck.enumerate_slides()[index - 1]
    br = Browser_(scale)
    try:
        s = br.session(slide.url, js=uses_js_loops(slide))
        s.press(slide.clicks)
        s.settle()
        frame = s.shot()
        s.close()
    finally:
        br.close()
    return index, frame


def _export_and_compare(pptx: Path, expected: dict[int, np.ndarray], out_dir: Path, size: tuple[int, int]) -> list[dict]:
    import win32com.client

    out_dir.mkdir(parents=True, exist_ok=True)
    app = win32com.client.Dispatch("PowerPoint.Application")
    pre = app.Presentations.Count
    pres = app.Presentations.Open(str(pptx), ReadOnly=True, Untitled=False, WithWindow=False)
    rows = []
    try:
        for pos, want in expected.items():
            p = out_dir / f"slide_{pos:02d}.png"
            pres.Slides(pos).Export(str(p), "PNG", size[0], size[1])
            got = np.asarray(Image.open(p).convert("RGB"))
            d = np.abs(got.astype(np.int16) - want.astype(np.int16)).max(axis=2)
            big = d > 24
            ys, xs = np.nonzero(big)
            rows.append({"index": pos, "mean": float(d.mean()), "over24": int(big.sum()), "over60": int((d > 60).sum()),
                         "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if big.any() else None})
    finally:
        pres.Close()
        if pre == 0:
            app.Quit()
    return rows


def _contact_sheet(index_dir: Path, indices: list[int], out: Path):
    rows = (len(indices) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * THUMB_W, rows * THUMB_H), "white")
    draw = ImageDraw.Draw(sheet)
    for k, i in enumerate(indices):
        im = Image.open(index_dir / f"slide_{i:02d}.png").convert("RGB").resize((THUMB_W - 6, THUMB_H - 6), Image.LANCZOS)
        x, y = (k % COLS) * THUMB_W, (k // COLS) * THUMB_H
        sheet.paste(im, (x + 3, y + 3))
        draw.rectangle([x + 3, y + 3, x + 30, y + 17], fill="black")
        draw.text((x + 6, y + 4), str(i), fill="white")
        draw.rectangle([x + 3, y + 3, x + THUMB_W - 4, y + THUMB_H - 4], outline=(200, 200, 200))
    sheet.save(out, optimize=True)


def main(argv=None):
    from .verify import inspect

    pptx = DIST / "deck.pptx"
    scale = 1.25
    if not pptx.exists():
        raise SystemExit(f"{pptx} does not exist: run `export all` first")
    manifests = {m["index"]: m for m in (json.loads(f.read_text(encoding="utf-8")) for f in sorted(WORK.glob("slide_*/manifest.json")))}
    slide_list = deck.enumerate_slides()
    indices = list(range(1, len(slide_list) + 1))

    xml_rows, n_media, media_bytes = inspect(pptx)
    size_mb = pptx.stat().st_size / 1e6

    with ProcessPoolExecutor(max_workers=4) as ex:
        want = dict(ex.map(_final_frame, indices, [scale] * len(indices)))
    img_dir = DIST / "_export"
    cmp = {r["index"]: r for r in _export_and_compare(pptx, want, img_dir, (want[1].shape[1], want[1].shape[0]))}
    _contact_sheet(img_dir, indices, DIST / "contact_sheet.png")

    L = []
    L.append("# PowerPoint export report\n")
    L.append(f"* file: `{pptx.relative_to(DIST.parent)}`, **{size_mb:.1f} MB** (cap {SIZE_CAP_MB:.0f} MB){'' if size_mb <= SIZE_CAP_MB else '  **OVER THE CAP**'}")
    L.append(f"* slides: {len(xml_rows)} in the file, {len(slide_list)} in the HTML deck")
    L.append(f"* media in the package: {n_media} files, {media_bytes / 1e6:.1f} MB")
    L.append(f"* build states: {sum(m['clicks'] + 1 for m in manifests.values())}, "
             f"clip files: {sum(len(s['clips']) for m in manifests.values() for s in m['states'])}\n")
    L.append("| # | slide | runtime | clicks (html/pptx) | pictures | clips | clip KB | PNG KB | mean diff | px > 24 | px > 60 | notes |")
    L.append("|--:|---|---|---|--:|--:|--:|--:|--:|--:|--:|---|")
    warnings = []
    for i, sl in zip(indices, slide_list):
        m, x, c = manifests.get(i), xml_rows[i - 1], cmp[i]
        if m is None:
            L.append(f"| {i} | {sl.rel} | {sl.runtime} | – | – | – | – | – | – | – | – | **not captured** |")
            warnings.append(f"slide {i} ({sl.rel}) has no capture")
            continue
        folder = WORK / f"slide_{i:02d}"
        clips = [cl for s in m["states"] for cl in s["clips"]]
        pics = [p for s in m["states"] for p in s["pictures"]] + [o for s in m["states"] for o in s.get("overlays", [])]
        png_kb = sum((folder / p["png"]).stat().st_size for p in pics if (folder / p["png"]).exists()) // 1024
        clip_kb = sum(cl["bytes"] for cl in clips) // 1024
        notes = [n for s in m["states"] for n in s["notes"]]
        weak = sorted({cl["quality"] for cl in clips} - {"exact", "empirical", "empirical+declared", "retimed"})
        flag = []
        if x["click"] != m["clicks"]:
            flag.append("CLICKS DIFFER")
        if x["loops"] != len(clips):
            flag.append("LOOP COUNT DIFFERS")
        if x["interactive"]:
            flag.append("click-to-pause trigger present")
        if weak:
            flag.append("loop quality: " + ", ".join(weak))
        if c["mean"] > MEAN_LIMIT or c["over60"] > OVER60_LIMIT:      # small edge differences from 4:2:0 video are normal
            flag.append("final state differs from the browser")
        warnings += [f"slide {i} ({sl.rel}): {f}" for f in flag] + [f"slide {i} ({sl.rel}): {n}" for n in notes]
        L.append(f"| {i} | {sl.rel} | {sl.runtime} | {m['clicks']}/{x['click']} | {len(pics)} | {len(clips)} | {clip_kb} | {png_kb} | "
                 f"{c['mean']:.2f} | {c['over24']} | {c['over60']} | {'; '.join(flag + [n[:80] for n in notes])} |")
    L.append("\n## Warnings and adjustments\n")
    L += [f"* {w}" for w in warnings] or ["* none"]
    L.append("\n## What this report does not prove\n")
    L.append("Playback. Whether the loops start by themselves, the clicks advance the builds and the fades look right can only be "
             "confirmed by running the deck in PowerPoint (Slide Show, from the start). `contact_sheet.png` shows each slide's last build state "
             "as PowerPoint draws it.")
    (DIST / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    bad = [w for w in warnings if "DIFFER" in w or "differs" in w or "trigger" in w]
    print(f"wrote {DIST / 'report.md'} and {DIST / 'contact_sheet.png'}  ({size_mb:.1f} MB)")
    print(f"{len(warnings)} warning line(s), {len(bad)} serious")
    if size_mb > SIZE_CAP_MB:
        raise SystemExit("deck is over the size cap")


if __name__ == "__main__":
    main()
