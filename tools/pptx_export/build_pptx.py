"""Assemble the captured slides into a .pptx by driving desktop PowerPoint through COM.

PowerPoint writes the animation and media XML itself (nothing is hand-written), using the calls verified in the smoke test:
  * Fade entrance (effect 10) for pictures and clips, first effect of a click "on click", the rest "with previous";
  * MediaPlay (83) with previous for every new clip; the clip's loop flag = PlaySettings.LoopUntilStopped;
  * the clips of the previous state are stopped (MediaStop, 85) and hidden (Appear + Exit) *after* the fade has finished;
  * the automatic "click the video to pause" triggers PowerPoint adds to every video are deleted, otherwise a click that lands
    on an animated diagram would pause it instead of advancing the slide.
Only presentations created here are ever closed, and PowerPoint is only quit if it had no windows open when we started.
"""
from __future__ import annotations

import json
from pathlib import Path

import win32com.client

from .export import DIST, WORK

# MsoAnimEffect / MsoAnimTriggerType / PpEntryEffect / PpSlideLayout values (checked against the type library in the smoke test)
FADE, APPEAR, MEDIA_PLAY, MEDIA_STOP = 10, 1, 83, 85
ON_CLICK, WITH_PREV, AFTER_PREV = 1, 2, 3
ENTRY_FADE_SMOOTHLY = 3849
LAYOUT_TITLE_ONLY = 11
SAVE_AS_PPTX = 24
SLIDE_W_PT, SLIDE_H_PT = 960.0, 540.0
FADE_S = 0.3


class Deck:
    def __init__(self, scale: float):
        self.app = win32com.client.Dispatch("PowerPoint.Application")
        self.pre_existing = self.app.Presentations.Count
        self.app.DisplayAlerts = 1                    # ppAlertsNone
        self.pres = self.app.Presentations.Add(WithWindow=True)
        self.pres.PageSetup.SlideWidth, self.pres.PageSetup.SlideHeight = SLIDE_W_PT, SLIDE_H_PT
        self.pt = SLIDE_W_PT / (1280 * scale)         # points per captured pixel

    def box(self, b):
        return b[0] * self.pt, b[1] * self.pt, b[2] * self.pt, b[3] * self.pt

    def finish(self, out: Path):
        if out.exists():
            out.unlink()
        self.pres.SaveAs(str(out), SAVE_AS_PPTX)
        self.pres.Close()
        if self.pre_existing == 0:
            self.app.Quit()


def _effect(seq, shape, kind, trigger, duration=None, exit_=False):
    e = seq.AddEffect(shape, kind, 0, trigger)
    if exit_:
        e.Exit = True
    if duration is not None:
        e.Timing.Duration = duration
    return e


def add_slide(deck: Deck, m: dict, index_in_deck: int, work: Path) -> dict:
    slide = deck.pres.Slides.Add(index_in_deck, LAYOUT_TITLE_ONLY)
    title = slide.Shapes.Title
    title.TextFrame.TextRange.Text = m["title"]
    title.Left, title.Top, title.Width, title.Height = 0, -200, SLIDE_W_PT, 60       # parked off-slide: outline/accessibility only
    folder = work / f"slide_{m['index']:02d}"
    shapes_by_state: list[dict] = []
    clip_shape: dict[str, object] = {}
    # 1) shapes first (z-order = creation order: older states below newer ones)
    for st in m["states"]:
        pics, clips = [], []
        for k, pic in enumerate(st["pictures"], 1):
            l, t, w, h = deck.box(pic["box"])
            shp = slide.Shapes.AddPicture(str(folder / pic["png"]), False, True, l, t, w, h)
            shp.Name = f"S{m['index']:02d}_b{st['b']}_pic{k}"
            shp.AlternativeText = m["title"] if k == 1 and st["b"] == 0 else f"{m['title']} (build {st['b']})"
            pics.append(shp)
        for c in st["clips"]:
            l, t, w, h = deck.box(c["box"])
            shp = slide.Shapes.AddMediaObject2(str(folder / c["mp4"]), False, True, l, t, w, h)
            shp.LockAspectRatio = False
            shp.Left, shp.Top, shp.Width, shp.Height = l, t, w, h
            shp.Name = f"S{m['index']:02d}_{c['id']}"
            shp.AlternativeText = f"Animation: {m['title']}"
            # Only the loop flag is set. Touching HideWhileNotPlaying makes PowerPoint silently drop the loop (found by bisecting).
            shp.AnimationSettings.PlaySettings.LoopUntilStopped = True
            clip_shape[c["id"]] = shp
            clips.append(shp)
        texts = []
        for k, ov in enumerate(st.get("overlays", []), 1):          # text above the clips, kept as a crisp transparent PNG
            l, t, w, h = deck.box(ov["box"])
            shp = slide.Shapes.AddPicture(str(folder / ov["png"]), False, True, l, t, w, h)
            shp.Name = f"S{m['index']:02d}_b{st['b']}_text{k}"
            shp.AlternativeText = ov["alt"]
            texts.append((shp, ov["entrance"]))
        shapes_by_state.append({"pics": pics, "clips": clips, "texts": texts})
    # 2) then the effects, in click order
    seq = slide.TimeLine.MainSequence
    while seq.Count:                                  # AddMediaObject2 may have created effects of its own: start clean
        seq.Item(seq.Count).Delete()
    for st, sh in zip(m["states"], shapes_by_state):
        if st["b"] == 0:                              # on entry: clips start by themselves, no click involved
            for n, shp in enumerate(sh["clips"]):
                _effect(seq, shp, MEDIA_PLAY, AFTER_PREV if n == 0 else WITH_PREV)
            for shp, entrance in sh["texts"]:
                if entrance:                              # the page's own entrance: fades in while the clip starts playing
                    delay, dur = entrance
                    e = _effect(seq, shp, FADE, WITH_PREV, dur)
                    e.Timing.TriggerDelayTime = delay
            continue
        first = True                                  # the first effect of every later state is the click itself

        def nxt():
            nonlocal first
            t, first = (ON_CLICK if first else WITH_PREV), False
            return t

        for shp in sh["pics"]:
            _effect(seq, shp, FADE, nxt(), FADE_S)
        for shp in sh["clips"]:
            _effect(seq, shp, FADE, nxt(), FADE_S)
            _effect(seq, shp, MEDIA_PLAY, WITH_PREV)
        for shp, _ in sh["texts"]:
            _effect(seq, shp, FADE, nxt(), FADE_S)
        for n, cid in enumerate(st["stop"]):          # old clips go away once the new picture is fully opaque
            old = clip_shape[cid]
            _effect(seq, old, MEDIA_STOP, ON_CLICK if first else (AFTER_PREV if n == 0 else WITH_PREV))
            first = False
            _effect(seq, old, APPEAR, WITH_PREV, exit_=True)
        if first:                                     # nothing visible changes, but the click must still exist
            ph = slide.Shapes.AddShape(1, 0, 0, 1, 1)
            ph.Fill.Visible = False
            ph.Line.Visible = False
            ph.Name = f"S{m['index']:02d}_b{st['b']}_placeholder"
            _effect(seq, ph, FADE, ON_CLICK, 0.01)
    # 3) no "click the video to pause" triggers
    its = slide.TimeLine.InteractiveSequences
    for i in range(its.Count, 0, -1):
        iseq = its.Item(i)
        for j in range(iseq.Count, 0, -1):
            iseq.Item(j).Delete()
    slide.SlideShowTransition.EntryEffect = ENTRY_FADE_SMOOTHLY
    slide.SlideShowTransition.Duration = 0.2
    clicks = sum(1 for i in range(1, seq.Count + 1) if seq.Item(i).Timing.TriggerType == ON_CLICK)
    return {"index": m["index"], "title": m["title"], "expected_clicks": m["clicks"], "clicks": clicks,
            "effects": seq.Count, "pictures": sum(len(s["pics"]) for s in shapes_by_state),
            "clips": sum(len(s["clips"]) for s in shapes_by_state),
            "interactive_left": its.Count}


def build(args):
    manifests = []
    for f in sorted(WORK.glob("slide_*/manifest.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        manifests.append(m)
    if args.slides:
        from .export import parse_selection
        from .slides import enumerate_slides
        wanted = set(parse_selection(args.slides, len(enumerate_slides())))
        manifests = [m for m in manifests if m["index"] in wanted]
    manifests.sort(key=lambda m: m["index"])
    if not manifests:
        raise SystemExit("nothing captured yet: run the capture step first")
    deck = Deck(manifests[0]["scale"])
    out = DIST / ("deck.pptx" if not args.slides else "deck_slice.pptx")
    report = []
    try:
        for i, m in enumerate(manifests, 1):
            r = add_slide(deck, m, i, WORK)
            report.append(r)
            flag = "" if r["clicks"] == r["expected_clicks"] and r["interactive_left"] == 0 else "   <-- CHECK"
            print(f"slide {m['index']:>2}: clicks {r['clicks']}/{r['expected_clicks']}  effects {r['effects']:>3}  pictures {r['pictures']:>3}  clips {r['clips']:>2}{flag}", flush=True)
    finally:
        deck.finish(out)
    fixed = fix_posters(out, manifests)
    if fixed:
        print(f"repaired {fixed} video poster frame(s) that PowerPoint saved blank")
    size = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size:.1f} MB, {len(manifests)} slides)")
    (DIST / "build_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    problems = post_check(out, report)
    problems += poster_problems(out, manifests)
    if problems:
        raise SystemExit("POST-BUILD CHECK FAILED:\n  " + "\n  ".join(problems))
    print("post-build check passed: every clip loops, valid poster frames, no click-to-pause triggers, click counts match the HTML")


def _posters(pptx: Path, manifests: list[dict]):
    """Yield (slide position, shape name, relationship id, poster part, clip entry, slide manifest) for every video shape of the
    saved file. The poster is the picture PowerPoint shows for a video that is not playing (editor, export, print)."""
    import re
    import zipfile
    with zipfile.ZipFile(pptx) as z:
        for pos, m in enumerate(manifests, 1):
            x = z.read(f"ppt/slides/slide{pos}.xml").decode("utf-8")
            rels = z.read(f"ppt/slides/_rels/slide{pos}.xml.rels").decode("utf-8")
            target = {}
            for rel in re.findall(r"<Relationship\b[^>]*>", rels):
                rid, tg = re.search(r'Id="([^"]+)"', rel), re.search(r'Target="([^"]+)"', rel)
                if rid and tg:
                    target[rid.group(1)] = tg.group(1)
            clips = {f"S{m['index']:02d}_{c['id']}": c for s in m["states"] for c in s["clips"]}
            for pic in re.findall(r"<p:pic>.*?</p:pic>", x, re.S):
                if "<a:videoFile" not in pic:
                    continue
                name = re.search(r'<p:cNvPr[^>]*\bname="([^"]*)"', pic).group(1)
                if name not in clips:
                    raise RuntimeError(f"slide {pos}: video shape {name!r} is not a clip of slide {m['index']}")
                rid = re.search(r'<a:blip r:embed="([^"]+)"', pic).group(1)
                yield pos, name, rid, "ppt/" + target[rid].replace("../", ""), clips[name], m


def _first_frame(mp4: Path):
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode {mp4}")
    return frame[:, :, ::-1]


def _poster_ok(raw: bytes, first) -> bool:
    import io
    import numpy as np
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    if im.size != (first.shape[1], first.shape[0]):
        return False
    return float(np.abs(np.asarray(im).astype(np.int16) - first.astype(np.int16)).mean()) < 12


def fix_posters(pptx: Path, manifests: list[dict]) -> int:
    """PowerPoint makes poster frames asynchronously and sometimes saves a blank 1x1 one (identical blanks are then stored once
    and shared by several clips). Give every such clip its own poster picture, its first frame: a new image part and a new
    relationship in its slide. Nothing else in the package is touched."""
    import io
    import re
    import shutil
    import zipfile
    from PIL import Image

    bad = []                                              # (slide position, shape name, old rid, first frame)
    with zipfile.ZipFile(pptx) as z:
        for pos, name, rid, poster, clip, m in _posters(pptx, manifests):
            first = _first_frame(WORK / f"slide_{m['index']:02d}" / clip["mp4"])
            if not _poster_ok(z.read(poster), first):
                bad.append((pos, name, rid, first))
    if not bad:
        return 0
    by_slide: dict[int, list] = {}
    for b in bad:
        by_slide.setdefault(b[0], []).append(b)
    changed, new_parts = {}, {}
    with zipfile.ZipFile(pptx) as z:
        for pos, items in by_slide.items():
            x = z.read(f"ppt/slides/slide{pos}.xml").decode("utf-8")
            rels = z.read(f"ppt/slides/_rels/slide{pos}.xml.rels").decode("utf-8")
            for k, (_, name, old_rid, first) in enumerate(items, 1):
                new_rid, part = f"rIdPoster{k}", f"ppt/media/poster_s{pos}_{k}.png"
                buf = io.BytesIO()
                Image.fromarray(first).save(buf, "PNG")
                new_parts[part] = buf.getvalue()
                rels = rels.replace("</Relationships>",
                                    f'<Relationship Id="{new_rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                                    f'Target="../media/{part.split("/")[-1]}"/></Relationships>')
                for pic in re.findall(r"<p:pic>.*?</p:pic>", x, re.S):
                    if re.search(rf'<p:cNvPr[^>]*\bname="{re.escape(name)}"', pic) and "<a:videoFile" in pic:
                        x = x.replace(pic, pic.replace(f'<a:blip r:embed="{old_rid}"', f'<a:blip r:embed="{new_rid}"', 1), 1)
                        break
            changed[f"ppt/slides/slide{pos}.xml"] = x.encode("utf-8")
            changed[f"ppt/slides/_rels/slide{pos}.xml.rels"] = rels.encode("utf-8")
        # an old poster picture that no slide refers to any more is left out of the package
        still_used = set()
        for n in z.namelist():
            if re.fullmatch(r"ppt/slides/_rels/slide\d+\.xml\.rels", n):
                data = changed[n].decode("utf-8") if n in changed else z.read(n).decode("utf-8")
                still_used |= {"ppt/" + t.replace("../", "") for t in re.findall(r'Target="(\.\./media/[^"]+)"', data)}
        old_posters = {p for _, _, _, p, _, _ in _posters(pptx, manifests)}
        tmp = pptx.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in z.infolist():
                if item.filename in old_posters and item.filename not in still_used:
                    continue
                zout.writestr(item, changed.get(item.filename, z.read(item.filename)))
            for part, data in new_parts.items():
                zout.writestr(part, data)
    shutil.move(str(tmp), str(pptx))
    return len(bad)


def poster_problems(pptx: Path, manifests: list[dict]) -> list[str]:
    import zipfile
    out = []
    with zipfile.ZipFile(pptx) as z:
        for pos, name, rid, poster, clip, m in _posters(pptx, manifests):
            if not _poster_ok(z.read(poster), _first_frame(WORK / f"slide_{m['index']:02d}" / clip["mp4"])):
                out.append(f"slide {m['index']}: poster frame of {clip['id']} is wrong ({poster})")
    return out


def post_check(pptx: Path, report: list[dict]) -> list[str]:
    """Read the saved XML (not COM) and compare it with what was meant to be built."""
    from .verify import inspect
    rows, _, _ = inspect(pptx)
    problems = []
    for r, x in zip(report, rows):
        tag = f"slide {r['index']} ({x['slide']})"
        if x["loops"] != r["clips"]:
            problems.append(f"{tag}: {r['clips']} clips but {x['loops']} loop flags in the file")
        if x["interactive"]:
            problems.append(f"{tag}: {x['interactive']} click-to-pause trigger(s) left")
        if x["click"] != r["expected_clicks"]:
            problems.append(f"{tag}: {x['click']} click effects in the file, the HTML has {r['expected_clicks']}")
    if len(rows) != len(report):
        problems.append(f"{len(rows)} slides in the file, {len(report)} built")
    return problems
