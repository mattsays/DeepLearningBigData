"""Enumerate the HTML deck in presentation order and describe the click structure of every slide.

The order replicates `compareSlides` in pmaker.html: folder path components are compared with a numeric,
case-insensitive collator (a shallower path sorts first), then file names with the same collator.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLIDES_DIR = ROOT / "slides"


def natural_key(text: str):
    """Numeric, case-insensitive sort key (like Intl.Collator with numeric: true, sensitivity: 'base')."""
    return [(0, int(tok)) if tok.isdigit() else (1, tok.lower()) for tok in re.split(r"(\d+)", text) if tok != ""]


def sort_key(path: Path):
    rel = path.relative_to(SLIDES_DIR)
    folders = tuple(tuple(natural_key(part)) for part in rel.parts[:-1])
    return (folders, tuple(natural_key(rel.name)))


@dataclass(frozen=True)
class Slide:
    index: int            # 1-based position in the deck
    path: Path
    title: str
    runtime: str          # "shared": data-step elements are hidden until a click; "tia": data-steps builds, build 1 shown on entry
    states: int           # number of distinct screens (state 0 is what is on screen when the slide opens)

    @property
    def clicks(self) -> int:
        return self.states - 1

    @property
    def rel(self) -> str:
        return self.path.relative_to(SLIDES_DIR).as_posix()

    @property
    def url(self) -> str:
        return self.path.resolve().as_uri()


def describe(path: Path, index: int) -> Slide:
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"<title>(.*?)</title>", text, re.S)
    title = html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip()) if title_match else path.stem
    builds = re.search(r'data-steps="(\d+)"', text)
    if builds:
        # Embedded runtime: build 1 is already on screen when the slide opens, so K builds need K-1 clicks.
        return Slide(index, path, title, "tia", max(1, int(builds.group(1))))
    steps = {m for m in re.findall(r'<[^>]*\sdata-step="([\d.]+)"', text)}
    # Shared runtime: nothing with data-step is visible at first, every distinct value needs one click.
    return Slide(index, path, title, "shared", len(steps) + 1)


def enumerate_slides() -> list[Slide]:
    files = sorted(SLIDES_DIR.rglob("*.htm*"), key=sort_key)
    return [describe(p, i) for i, p in enumerate(files, 1)]


if __name__ == "__main__":
    total_states = 0
    for s in enumerate_slides():
        total_states += s.states
        print(f"{s.index:>2} {s.runtime:<6} states={s.states:<2} {s.rel}  | {s.title}")
    print("total states:", total_states)
