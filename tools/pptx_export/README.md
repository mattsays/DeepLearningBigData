# HTML deck → PowerPoint

Turns the HTML slide deck (`slides/**`, the order `pmaker.html` plays) into `dist/deck.pptx`. The HTML stays the source of truth:
the export is a **pixel-faithful picture of every build state**, with the animated diagrams as **looping videos**, and the clicks
rebuilt as real PowerPoint animations. Text in the .pptx is therefore not editable; change the HTML and run the export again.

## Run it

From the repository root (Windows, desktop PowerPoint installed, Microsoft Edge installed):

```
pip install playwright imageio-ffmpeg pywin32 numpy opencv-python pillow
python -m tools.pptx_export.export all          # capture every slide, then build dist/deck.pptx
python -m tools.pptx_export.report              # check it, write dist/report.md and dist/contact_sheet.png
```

Smaller runs:

```
python -m tools.pptx_export.export list                         # slide numbers, runtimes, number of build states
python -m tools.pptx_export.export capture --slides 3,9,27      # capture only these (cached in dist/_work)
python -m tools.pptx_export.export build   --slides 3,9,27      # build dist/deck_slice.pptx from the captured slides
```

Options: `--workers 4` parallel browsers, `--scale 1.25` (1600×900 pictures), `--crf 18` video quality (lower = sharper and bigger),
`--force` captures again even when a slide's cache is current. A slide is captured again automatically when its HTML, its
`slideui.js/css` or the capture code changes. Nothing is written outside `dist/` and `tools/pptx_export/.cache/` (both ignored by git).
Close other PowerPoint presentations you care about before `build`; it never closes or quits a PowerPoint that was already running.

## What it does, per slide

1. **Capture** (`capture.py`, `slide_capture.py`): Playwright drives the real slide runtime with the same key presses. Time is made
   deterministic (fake clock for JavaScript loops, SMIL `setCurrentTime`, CSS animations paused and seeked), so a state is always the
   same picture at τ = 0.
2. **Regions** (`regions.py`): frames over time are differenced; pixels that move are grouped into boxes, snapped to whole `<svg>`
   diagrams. The loop length comes from the declared SMIL/CSS periods (exact common period when it is ≤ 9 s), from a measured lag for
   JavaScript loops, and otherwise from a **slide-level retiming** (a diagram's periods are nudged to a shared one, at most 20 %, only
   in the captured copy; the HTML is untouched). Every retimed loop is listed in `report.md`.
3. **Pictures**: build 0 is a full picture; every later build adds *patches* (only the pixels that changed since the previous build).
4. **Clips**: one loop of every moving region, written as H.264 (`video.py`; yuv420p, bt709, no audio), the frame rate chosen so that the
   loop is seamless.
5. **Title slides** (`TEXT_LAYERS` in `slide_capture.py`): the blurred animated background is a video without lettering, the text is a
   separate transparent PNG on top, and the GNN title keeps its own fade-in (title, then subtitle 0.45 s later).
6. **Build** (`build_pptx.py`): PowerPoint itself writes the file through COM. Every click is a Fade entrance (0.3 s) of the new pictures
   and clips; new clips start playing with the fade and loop until stopped; the clips of the previous build are stopped and hidden after the
   fade; the automatic "click the video to pause" triggers are deleted so a click never pauses a diagram; Fade slide transition 0.2 s;
   a title and alt text on every slide.
7. **Check** (`verify.py`, `report.py`): after the build the saved XML is read back (one loop flag per clip, no pause triggers, the same
   number of clicks as the HTML) and every slide's last build state, exported by PowerPoint, is compared with the browser.

## Known differences from the HTML

* Camera zoom/pan between builds (the SNN neuron slides and message passing) becomes a cross-fade.
* Loops are finite videos; a diagram whose periods have no short common period is retimed (≤ 20 %) or, failing that, looped over a
  longer common period. The seam, if any, is reported.
* Fonts and MathJax formulas are pictures, so they look exactly like the browser but cannot be edited or searched; the title placeholder
  and alt text keep the outline and screen-reader order.

## What only a person can confirm

The tool cannot watch a Slide Show. Open `dist/deck.pptx` in PowerPoint, start the show from the first slide, and check that the diagrams
loop by themselves, each click advances one build, and the fades look right.
