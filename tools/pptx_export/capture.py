"""Drive one slide in Edge (Playwright) through the real slide runtime and grab deterministic frames.

Time is made deterministic on three independent axes, all advanced together by `Session.advance_to(t_ms)`:
  * JavaScript (requestAnimationFrame / performance.now): Playwright's fake clock (`page.clock`);
  * CSS animations: paused and seeked with the Web Animations API (finite ones are finished, infinite ones are seeked);
  * SMIL (<animate>, <animateMotion>): `svg.pauseAnimations()` + `svg.setCurrentTime()`.
tau = 0 is the instant the slide has settled after its key presses. Replaying the same key presses on a fresh page
reproduces the same frames, which is what lets the exporter record a clip in a second pass.

Screenshots go straight through the DevTools protocol (`Page.captureScreenshot`, optimizeForSpeed), which is about 15x
faster than Playwright's own screenshot call on this machine.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import Browser, BrowserContext, Page, Route, sync_playwright

CACHE = Path(__file__).resolve().parent / ".cache"
VIEW_W, VIEW_H = 1280, 720

FREEZE_JS = r"""
() => {
  if (window.__fz) return;
  window.__fz = { T: 0 };
  window.__seek = (T) => {
    // Every looping CSS animation is put at exactly T ms from its own start (its delay included), never at the wall-clock
    // offset it happened to have, so tau = 0 is the same picture on every fresh page. Finite ones (entrances) are finished.
    window.__fz.T = T;
    for (const a of document.getAnimations()) {
      const eff = a.effect, timing = eff && eff.getComputedTiming ? eff.getComputedTiming() : null;
      if (timing && timing.iterations === Infinity) { a.pause(); a.currentTime = T; }
      else { try { a.finish(); } catch (e) {} }
    }
    for (const svg of document.querySelectorAll('svg')) {
      if (svg.ownerSVGElement) continue;
      try { svg.pauseAnimations(); svg.setCurrentTime(T / 1000); } catch (e) {}
    }
  };
  // Outermost <svg> diagrams: the natural unit of an animated region (css px).
  window.__svgs = () => [...document.querySelectorAll('svg')].filter(s => !s.ownerSVGElement).map(s => {
    const r = s.getBoundingClientRect();
    return { box: [r.left, r.top, r.width, r.height], animated: !!s.querySelector('animate, animateMotion, animateTransform') };
  });
  // Declared loop periods (SMIL and CSS) with the box they animate, in CSS pixels.
  window.__declared = () => {
    const out = [];
    const box = (el) => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; };
    const vis = (el) => (el.checkVisibility ? el.checkVisibility({ checkOpacity: false, checkVisibilityCSS: true }) : true);
    for (const el of document.querySelectorAll('animate, animateMotion, animateTransform')) {
      let d = NaN;
      try { d = el.getSimpleDuration(); } catch (e) {}
      const tgt = el.targetElement || el.parentElement;
      if (isFinite(d) && d > 0 && tgt) out.push({ kind: 'smil', ms: Math.round(d * 1000), box: box(tgt), visible: vis(tgt) });
    }
    for (const a of document.getAnimations()) {
      const t = a.effect && a.effect.getComputedTiming ? a.effect.getComputedTiming() : null;
      const tgt = a.effect && a.effect.target;
      if (t && t.iterations === Infinity && t.duration > 0 && tgt && tgt.getBoundingClientRect)
        out.push({ kind: 'css', ms: Math.round(t.duration), box: box(tgt), visible: vis(tgt) });
    }
    return out;
  };
}
"""


def _cdn_route(route: Route):
    """Record-and-replay for CDN requests (MathJax and its fonts), so a re-run is offline and reproducible."""
    url = route.request.url
    key = CACHE / "cdn" / hashlib.sha1(url.encode()).hexdigest()
    meta, body = key.with_suffix(".json"), key.with_suffix(".bin")
    if meta.exists() and body.exists():
        route.fulfill(status=200, headers=json.loads(meta.read_text())["headers"], body=body.read_bytes())
        return
    resp = route.fetch()
    data = resp.body()
    key.parent.mkdir(parents=True, exist_ok=True)
    body.write_bytes(data)
    meta.write_text(json.dumps({"headers": {k: v for k, v in resp.headers.items() if k.lower() in ("content-type", "access-control-allow-origin")}}))
    route.fulfill(response=resp, body=data)


class Browser_:
    """One Edge process and one browsing context per worker; every pass of every slide gets a fresh page in it."""

    def __init__(self, scale: float):
        self.scale = scale
        self.pw = sync_playwright().start()
        self.browser: Browser = self.pw.chromium.launch(channel="msedge", headless=True)
        self.ctx: BrowserContext = self.browser.new_context(
            viewport={"width": VIEW_W, "height": VIEW_H}, device_scale_factor=scale)
        self.ctx.route("https://cdn.jsdelivr.net/**", _cdn_route)

    def session(self, url: str, js: bool = True, init_js: str | None = None) -> "Session":
        return Session(self, url, js, init_js)

    def close(self):
        try:
            self.ctx.close()
            self.browser.close()
        finally:
            self.pw.stop()


class Session:
    """One fresh page for one slide, positioned on state `b` and settled at tau = 0."""

    def __init__(self, owner: Browser_, url: str, js: bool = True, init_js: str | None = None):
        self.js = js          # False: only SMIL/CSS animate, so time can be rewound by seeking (no replay needed)
        self.scale = owner.scale
        self.page: Page = owner.ctx.new_page()
        self.cdp = owner.ctx.new_cdp_session(self.page)
        self.errors: list[str] = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        if init_js:           # capture-time only retiming of a JavaScript loop (the HTML file itself is never changed)
            self.page.add_init_script(init_js)
        self.page.clock.install(time=0)
        self.page.goto(url, wait_until="load")
        self._ready()
        self.page.add_style_tag(content="*{transition:none!important;caret-color:transparent!important}")
        self.t_ms = 0.0

    def _pump(self, js_cond: str, timeout_s: float = 15.0):
        """Wait for a condition while the fake clock is advanced a little at a time (timers and animation frames
        are fake, so a plain wait_for_function would never see them fire)."""
        import time
        end = time.time() + timeout_s
        while time.time() < end:
            if self.page.evaluate(js_cond):
                return
            self.page.clock.run_for(100)
            time.sleep(0.02)
        raise TimeoutError(f"condition not met: {js_cond}")

    def _ready(self):
        p = self.page
        if p.evaluate("typeof window.MathJax !== 'undefined'"):      # this slide loads MathJax: wait until it has typeset
            self._pump("!!(window.MathJax && window.MathJax.startup && window.MathJax.startup.promise)")
            p.evaluate("() => { window.__mj = false; MathJax.startup.promise.then(() => MathJax.typesetPromise ? MathJax.typesetPromise() : null).then(() => { window.__mj = true; }); }")
            self._pump("window.__mj === true")
        p.evaluate("() => { window.__fonts = false; document.fonts.ready.then(() => { window.__fonts = true; }); }")
        self._pump("window.__fonts === true")
        self._pump("!document.documentElement.hasAttribute('data-slideui-init')")
        now = p.evaluate("Date.now()")
        p.clock.pause_at(now + 100)          # from here on only `run_for` moves JavaScript time
        p.evaluate(FREEZE_JS)

    # -- navigation ------------------------------------------------------------------------------
    def press(self, n: int):
        for _ in range(n):
            self.page.keyboard.press("ArrowRight")
            self.page.clock.run_for(60)

    def settle(self, settle_ms: int | None = None):
        """Let build tweens finish, then define tau = 0 (finite CSS animations finished, loops rewound to 0)."""
        self.page.clock.run_for(settle_ms if settle_ms is not None else (1800 if self.js else 400))
        self.page.evaluate("() => window.__seek(0)")
        self.t_ms = 0.0

    # -- time ------------------------------------------------------------------------------------
    def advance_to(self, t_ms: float):
        """Move all three time axes to the absolute time t_ms (never backwards)."""
        t_ms = round(t_ms)
        delta = t_ms - round(self.t_ms)
        if delta > 0 and self.js:
            self.page.clock.run_for(delta)
        self.t_ms = float(t_ms)
        self.page.evaluate("(T) => window.__seek(T)", self.t_ms)

    def svgs(self) -> list[dict]:
        return self.page.evaluate("() => window.__svgs()")

    def declared(self) -> list[dict]:
        return self.page.evaluate("() => window.__declared()")

    def retime_regions(self, items: list[dict]):
        """Capture-time only: nudge the SMIL `dur` of the animations inside given css boxes to a shared period.
        items: [{"box": [x, y, w, h], "map": {old_seconds: new_seconds}}]."""
        self.page.evaluate(
            """(items) => { for (const el of document.querySelectorAll('animate, animateMotion, animateTransform')) {
                 const tgt = el.targetElement || el.parentElement; const d = parseFloat(el.getAttribute('dur'));
                 if (!tgt || !isFinite(d)) continue;
                 const r = tgt.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
                 for (const it of items) {
                   const [x, y, w, h] = it.box;
                   if (cx >= x && cx <= x + w && cy >= y && cy <= y + h) {
                     const k = String(Math.round(d * 1000) / 1000);
                     if (it.map[k] !== undefined) el.setAttribute('dur', it.map[k] + 's');
                     break; } } } }""",
            [{"box": it["box"], "map": {str(k): v for k, v in it["map"].items()}} for it in items],
        )
        self.page.evaluate("() => window.__seek(0)")
        self.t_ms = 0.0

    # -- pictures --------------------------------------------------------------------------------
    def shot(self, clip=None, css_scale: bool = False) -> np.ndarray:
        """RGB frame. `clip` is (x, y, w, h) in css pixels; css_scale=True returns css-resolution pixels (cheap probes)."""
        x, y, w, h = clip if clip is not None else (0, 0, VIEW_W, VIEW_H)
        r = self.cdp.send("Page.captureScreenshot", {
            "format": "png", "optimizeForSpeed": True, "fromSurface": True,
            "clip": {"x": x, "y": y, "width": w, "height": h, "scale": 1.0 if css_scale else self.scale}})
        img = cv2.imdecode(np.frombuffer(base64.b64decode(r["data"]), np.uint8), cv2.IMREAD_COLOR)
        return img[:, :, ::-1]

    def shot_rgba(self) -> np.ndarray:
        """Full slide with a transparent page background (used for text layers over a video)."""
        self.cdp.send("Emulation.setDefaultBackgroundColorOverride", {"color": {"r": 0, "g": 0, "b": 0, "a": 0}})
        r = self.cdp.send("Page.captureScreenshot", {
            "format": "png", "optimizeForSpeed": True, "fromSurface": True,
            "clip": {"x": 0, "y": 0, "width": VIEW_W, "height": VIEW_H, "scale": self.scale}})
        self.cdp.send("Emulation.setDefaultBackgroundColorOverride")
        img = cv2.imdecode(np.frombuffer(base64.b64decode(r["data"]), np.uint8), cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    # -- text layers -----------------------------------------------------------------------------
    _ISO_JS = """(args) => {
        const [sel, mode] = args;
        let st = document.getElementById('__iso');
        if (!st) {
          st = document.createElement('style'); st.id = '__iso';
          st.textContent = '.__hide{visibility:hidden!important}' +
            'html.__isolating,html.__isolating body,html.__isolating .__bg{background:none!important;box-shadow:none!important}' +
            'html.__isolating body *{visibility:hidden!important}' +
            'html.__isolating .__keep,html.__isolating .__keep *{visibility:visible!important}';
          document.head.appendChild(st);
        }
        document.querySelectorAll('.__keep,.__bg,.__hide').forEach(e => e.classList.remove('__keep', '__bg', '__hide'));
        const root = document.documentElement;
        root.classList.remove('__isolating');
        if (!sel) return [];
        const hits = [...document.querySelectorAll(sel)];
        if (mode === 'hide') { hits.forEach(e => e.classList.add('__hide')); }
        else {
          hits.forEach(e => { e.classList.add('__keep'); for (let p = e.parentElement; p; p = p.parentElement) p.classList.add('__bg'); });
          root.classList.add('__isolating');
        }
        return hits.map(e => (e.innerText || '').trim());
    }"""

    def hide(self, selector):
        """Make the matching elements invisible (they keep their space). None undoes any hide/isolate."""
        return self.page.evaluate(self._ISO_JS, [selector, "hide"])

    def isolate(self, selector):
        """Show ONLY the matching elements, on a transparent page. Returns their texts. None undoes it."""
        return self.page.evaluate(self._ISO_JS, [selector, "show"])

    def close(self):
        try:
            self.page.close()
        except Exception:
            pass
