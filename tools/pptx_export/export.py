"""HTML deck -> PowerPoint.

    python -m tools.pptx_export.export capture --slides 27,9,3      # capture (cached under dist/_work)
    python -m tools.pptx_export.export build   --slides 27,9,3      # assemble dist/deck.pptx through PowerPoint
    python -m tools.pptx_export.export all                          # everything

Run it from the repository root. Needs: playwright (+ Microsoft Edge), imageio-ffmpeg, pywin32, numpy, opencv-python, pillow,
and desktop PowerPoint on Windows for `build`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import slides as deck

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
WORK = DIST / "_work"


def parse_selection(text: str | None, total: int) -> list[int]:
    if not text:
        return list(range(1, total + 1))
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return sorted(set(out))


def _capture_worker(index: int, scale: float, crf: int, force: bool = False) -> dict:
    from .capture import Browser_
    from .slide_capture import cache_key, capture_slide

    slide = deck.enumerate_slides()[index - 1]
    cached = WORK / f"slide_{index:02d}" / "manifest.json"
    if cached.exists() and not force:
        m = json.loads(cached.read_text(encoding="utf-8"))
        if m.get("key") == cache_key(slide, scale, crf):
            m["seconds"], m["cached"] = 0.0, True
            return m
    t0 = time.time()
    for attempt in (1, 2):
        browser = Browser_(scale)
        try:
            manifest = capture_slide(browser, slide, WORK, scale, crf)
            break
        except Exception as exc:                      # one retry with a fresh browser, then report the failure
            if attempt == 2:
                return {"index": index, "rel": slide.rel, "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.time() - t0, 1)}
        finally:
            browser.close()
    manifest["seconds"] = round(time.time() - t0, 1)
    return manifest


def _report_line(i: int, m: dict, all_slides) -> None:
    if "error" in m:
        print(f"slide {i:>2} FAILED after {m['seconds']}s: {m['error']}  ({all_slides[i - 1].rel})", flush=True)
        return
    clips = sum(len(s['clips']) for s in m['states'])
    kb = sum(c['bytes'] for s in m['states'] for c in s['clips']) // 1024
    tag = 'cached' if m.get('cached') else f"{m['seconds']}s"
    print(f"slide {i:>2} {tag:>8}  states={len(m['states']):<2} clips={clips:<2} clip KB={kb:<5} ({all_slides[i - 1].rel})", flush=True)


def cmd_capture(args) -> list[dict]:
    all_slides = deck.enumerate_slides()
    picked = parse_selection(args.slides, len(all_slides))
    WORK.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.workers <= 1 or len(picked) == 1:
        for i in picked:
            m = _capture_worker(i, args.scale, args.crf, args.force)
            results[i] = m
            _report_line(i, m, all_slides)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_capture_worker, i, args.scale, args.crf, args.force): i for i in picked}
            for f in as_completed(futs):
                i = futs[f]
                m = f.result()
                results[i] = m
                _report_line(i, m, all_slides)
    return [results[i] for i in sorted(results)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["capture", "build", "all", "list"])
    ap.add_argument("--slides", help="e.g. 3,9,27 or 20-28 (deck positions, 1-based); default all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.25, help="device scale factor: 1.25 -> 1600x900 pictures")
    ap.add_argument("--crf", type=int, default=18, help="H.264 quality for clips (lower = bigger and sharper)")
    ap.add_argument("--force", action="store_true", help="capture again even when the cache is current")
    args = ap.parse_args(argv)
    if args.command == "list":
        for s in deck.enumerate_slides():
            print(f"{s.index:>2} {s.runtime:<6} states={s.states:<2} {s.rel}")
        return
    if args.command in ("capture", "all"):
        cmd_capture(args)
    if args.command in ("build", "all"):
        from .build_pptx import build
        build(args)


if __name__ == "__main__":
    main()
