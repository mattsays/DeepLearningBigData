"""Find the animated regions of a settled slide and the loop length of each.

Regions come from the temporal range of every pixel over a probe sequence (max - min over time): static pixels never move,
animated ones do. Loop lengths come first from the periods the page declares (SMIL `dur`, CSS animation duration); when a
diagram has no short common period they are searched for, or, if allowed, nudged to one shared period at capture time.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import cv2
import numpy as np

PROBE_THRESHOLD = 10          # gray levels of temporal range that count as "moving"
MAX_REGIONS = 6
MERGE_GAP = 10                # css px: boxes closer than this are merged
GOOD_LOOP_MS = 9000           # a natural common period up to this length is kept as is
MAX_EXACT_LOOP_MS = 32000
MAX_LOOP_MS = 16000
RETIME_TOLERANCE = 0.20       # snapping may change a loop's speed by at most 20 %


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self):
        return self.x + self.w

    @property
    def y1(self):
        return self.y + self.h

    @property
    def area(self):
        return self.w * self.h

    def intersects(self, o: "Box", pad: float = 0) -> bool:
        return not (self.x1 + pad < o.x or o.x1 + pad < self.x or self.y1 + pad < o.y or o.y1 + pad < self.y)

    def union(self, o: "Box") -> "Box":
        x, y = min(self.x, o.x), min(self.y, o.y)
        return Box(x, y, max(self.x1, o.x1) - x, max(self.y1, o.y1) - y)

    def padded(self, p: float, w: float, h: float) -> "Box":
        x, y = max(0, self.x - p), max(0, self.y - p)
        return Box(x, y, min(w, self.x1 + p) - x, min(h, self.y1 + p) - y)

    def device(self, scale: float, fw: int, fh: int) -> tuple[int, int, int, int]:
        """Even-aligned device-pixel rectangle (x, y, w, h) that contains this css box (H.264 needs even sizes)."""
        x0 = int(np.floor(self.x * scale / 2)) * 2
        y0 = int(np.floor(self.y * scale / 2)) * 2
        x1 = min(fw, int(np.ceil(self.x1 * scale / 2)) * 2)
        y1 = min(fh, int(np.ceil(self.y1 * scale / 2)) * 2)
        return x0, y0, max(2, x1 - x0), max(2, y1 - y0)


def moving_mask(frames: list[np.ndarray], thr: int = PROBE_THRESHOLD) -> np.ndarray:
    stack = np.stack(frames).astype(np.int16)
    return (stack.max(axis=0) - stack.min(axis=0)).max(axis=2) > thr


def _merge(boxes: list[Box]) -> list[Box]:
    boxes = boxes[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                u = a.union(b)
                if a.intersects(b, MERGE_GAP) or u.area <= 1.25 * (a.area + b.area):
                    boxes[i] = u
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def raw_regions(mask: np.ndarray) -> list[Box]:
    """Connected components of the moving pixels (css px), padded, not merged."""
    h, w = mask.shape
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.dilate(m, np.ones((13, 13), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return [Box(*map(float, stats[i][:4])).padded(4, w, h) for i in range(1, n) if stats[i][4] >= 80]


def _cap(boxes: list[Box]) -> list[Box]:
    boxes = boxes[:]
    while len(boxes) > MAX_REGIONS:           # merge the pair whose union wastes the least area
        best = min(((i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))),
                   key=lambda p: boxes[p[0]].union(boxes[p[1]]).area - boxes[p[0]].area - boxes[p[1]].area)
        boxes[best[0]] = boxes[best[0]].union(boxes[best[1]])
        del boxes[best[1]]
    return boxes


def find_regions(mask: np.ndarray, svgs: list[dict] | None = None, full_frac: float = 0.7) -> list[Box]:
    """Animated regions of a slide (css px). Moving pixels that lie inside an <svg> diagram become that whole diagram
    (one region, one loop, clean edges). The remaining ones (text, arrows, CSS effects) are merged when close together."""
    h, w = mask.shape
    raw = raw_regions(mask)
    snapped: list[Box] = []
    loose: list[Box] = []
    for b in raw:
        hit = None
        for sv in svgs or []:
            sb = Box(*sv["box"])
            ix = max(0.0, min(b.x1, sb.x1) - max(b.x, sb.x))
            iy = max(0.0, min(b.y1, sb.y1) - max(b.y, sb.y))
            if b.area > 0 and ix * iy >= 0.5 * b.area and sb.w > 8 and sb.h > 8:
                hit = sb if hit is None else hit.union(sb)
        if hit is None:
            loose.append(b)
            continue
        for i, m in enumerate(snapped):           # boxes that now coincide become one region
            ix = max(0.0, min(hit.x1, m.x1) - max(hit.x, m.x))
            iy = max(0.0, min(hit.y1, m.y1) - max(hit.y, m.y))
            if ix * iy >= 0.5 * min(hit.area, m.area):
                snapped[i] = m.union(hit)
                break
        else:
            snapped.append(hit)
    boxes = snapped + _merge(loose)
    boxes = [bx for bx in boxes if not any(o is not bx and o.x <= bx.x and o.y <= bx.y and o.x1 >= bx.x1 and o.y1 >= bx.y1 for o in boxes)]
    boxes = _cap(boxes)
    if sum(bx.area for bx in boxes) > full_frac * w * h:
        return [Box(0, 0, w, h)]
    return sorted(boxes, key=lambda bx: (bx.y, bx.x))


# ---------------------------------------------------------------------------------------------- loop lengths
def _lcm(values: list[int]) -> int:
    out = 1
    for v in values:
        out = out * v // gcd(out, v)
    return out


def declared_for(box: Box, declared: list[dict]) -> list[int]:
    ds = []
    for d in declared:
        db = Box(*d["box"])
        if db.w > 0 and db.h > 0 and db.intersects(box):
            ds.append(int(d["ms"]))
    return sorted(set(ds))


def retime_plan(periods_ms: list[int]) -> tuple[int, dict[float, float]] | None:
    """Find one period P so that every loop can be nudged to P / k (k integer) within RETIME_TOLERANCE.
    Returns (P in ms, {old_seconds: new_seconds}) or None when impossible."""
    ds = sorted(set(periods_ms))
    if len(ds) < 2:
        return None
    best = None
    for p in np.arange(min(ds) * 0.8, max(ds) * 1.2 + 1, 20.0):
        worst = max(abs((p / max(1, round(p / d))) / d - 1) for d in ds)
        if worst <= RETIME_TOLERANCE and (best is None or worst < best[0] - 1e-9 or (abs(worst - best[0]) < 1e-9 and p < best[1])):
            best = (worst, float(p))
    if best is None:
        return None
    p = best[1]
    return int(round(p)), {d / 1000: round(p / max(1, round(p / d)) / 1000, 4) for d in ds}


def choose_loop(periods_ms: list[int], slide_plan: tuple[int, dict[float, float]] | None = None) -> tuple[int, str, dict[float, float] | None]:
    """Loop length for a region given the periods declared inside it: (loop_ms, quality, retime map or None).
    A retime plan computed once for the whole slide is preferred, so a loop keeps the same speed in every build state."""
    ds = sorted(set(periods_ms))
    if len(ds) == 1:
        return ds[0], "exact", None
    lcm = _lcm(ds)
    if lcm <= GOOD_LOOP_MS:
        return lcm, "exact", None
    if slide_plan and all(round(d / 1000, 3) in {round(k, 3) for k in slide_plan[1]} for d in ds):
        return slide_plan[0], "retimed", {k: v for k, v in slide_plan[1].items() if round(k * 1000) in ds}
    plan = retime_plan(ds)
    if plan:
        return plan[0], "retimed", plan[1]
    if lcm <= MAX_EXACT_LOOP_MS:
        return lcm, "exact-long", None
    loop, quality = loop_from_declared(ds)
    return loop, quality, None


def loop_from_declared(periods_ms: list[int]) -> tuple[int, str]:
    """Loop length (ms) for a region whose declared periods are known; second value says how good it is."""
    ds = sorted(set(periods_ms))
    if len(ds) == 1:
        return ds[0], "exact"
    lcm = _lcm(ds)
    if lcm <= MAX_EXACT_LOOP_MS:
        return lcm, "exact"
    best = (1e9, MAX_LOOP_MS)
    for L in range(2000, MAX_LOOP_MS + 1, 10):      # least phase mismatch over all declared periods
        cost = max(min(L % d, d - (L % d)) / d for d in ds)
        if cost < best[0] - 1e-9:
            best = (cost, L)
    return best[1], f"seam~{best[0] * 100:.0f}%"


def empirical_loop(crops: list[np.ndarray], step_ms: float, min_ms: int = 1000) -> tuple[int | None, float]:
    """Smallest lag (ms) at which the sequence repeats itself, from low-resolution crops sampled every step_ms."""
    g = [cv2.resize(c, (c.shape[1] // 2 or 1, c.shape[0] // 2 or 1), interpolation=cv2.INTER_AREA).astype(np.int16) for c in crops]
    n = len(g)
    base = np.mean([np.abs(g[i] - g[i + 1]).mean() for i in range(min(n - 1, 20))]) or 1.0
    min_lag = max(1, int(min_ms / step_ms))
    best = (None, 1e9)
    k0 = min(5, n // 4)                              # ignore the first frames: a build's one-time intro is not part of the loop
    for lag in range(min_lag, n // 2 + 1):
        err = float(np.mean([np.abs(g[k] - g[k + lag]).mean() for k in range(k0, n - lag, max(1, (n - lag) // 12))]))
        if err < best[1]:
            best = (lag, err)
        if err < 0.35 * base:
            return int(round(lag * step_ms)), err / base
    if best[0] is not None and best[1] < 0.6 * base:
        return int(round(best[0] * step_ms)), best[1] / base
    return None, 1.0
