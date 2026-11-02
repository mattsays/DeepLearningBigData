/* SlideUI: scala la slide a 16:9, gestisce gli step e le formule MathJax */
(function slideRuntime() {
  var opts = window.__SLIDEUI || {};   // {showAll, atEnd} iniettato dall'editor / player
  var items = [], values = [], cur = -1;

  function fit() {
    var s = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
    document.documentElement.style.setProperty('--slide-scale', s);
  }
  function setStep(i) {
    cur = i;
    var limit = i >= 0 ? values[i] : -Infinity;
    items.forEach(function (it) {
      var show = it.n <= limit;
      it.el.classList.toggle('step-visible', show);
      it.el.classList.toggle('step-hidden', !show);
    });
  }
  function notify(action) {
    if (window.parent !== window) window.parent.postMessage({ slideui: action }, '*');
  }
  // F: schermo intero. Dentro un iframe la richiesta va inoltrata alla pagina che
  // lo contiene, che è l'unica autorizzata a entrare in fullscreen.
  function toggleFullscreen() {
    if (window.parent !== window) { notify('fullscreen'); return; }
    if (document.fullscreenElement) { document.exitFullscreen(); return; }
    if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen();
  }
  // Il player passa indice e totale: il contatore sta nella tela e scala con lei,
  // con lo stesso look del modulo 3.
  function renderCounter() {
    if (opts.index == null || !opts.total) return;
    var host = document.querySelector('.slide-layout') || document.body;
    var el = host.querySelector('.slideui-counter');
    if (!el) {
      el = document.createElement('div');
      el.className = 'slideui-counter';
      el.setAttribute('aria-hidden', 'true');
      el.innerHTML = '<span class="current-slide"></span><span class="total-slides"></span>';
      host.appendChild(el);
    }
    el.querySelector('.current-slide').textContent = String(opts.index + 1);
    el.querySelector('.total-slides').textContent = ' / ' + opts.total;
  }
  function handleKey(key) {
    if (key === 'f' || key === 'F') { toggleFullscreen(); return true; }
    if (opts.showAll) return false;
    if (key === 'ArrowRight' || key === ' ' || key === 'PageDown' || key === 'Enter') {
      if (cur < values.length - 1) setStep(cur + 1); else notify('next');
      return true;
    }
    if (key === 'ArrowLeft' || key === 'PageUp') {
      if (cur >= 0) setStep(cur - 1); else notify('prev');
      return true;
    }
    if (key === 'Escape') { notify('exit'); return true; }
    return false;
  }
  function loadMath() {
    if (window.MathJax || !/\\\(|\\\[|\$/.test(document.body.textContent)) return;
    window.MathJax = { tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    } };
    var s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
    s.async = true;
    document.head.appendChild(s);
  }
  function init() {
    fit();
    renderCounter();
    window.addEventListener('resize', fit);
    items = Array.prototype.map.call(document.querySelectorAll('[data-step]'), function (el) {
      return { el: el, n: parseFloat(el.getAttribute('data-step')) || 0 };
    });
    items.forEach(function (it) { if (values.indexOf(it.n) < 0) values.push(it.n); });
    values.sort(function (a, b) { return a - b; });
    if (!opts.showAll) setStep(opts.atEnd ? values.length - 1 : -1);
    document.addEventListener('keydown', function (e) { if (handleKey(e.key)) e.preventDefault(); });
    document.addEventListener('click', function (e) {
      if (!e.target.closest('a, button, input, textarea')) handleKey('ArrowRight');
    });
    window.addEventListener('message', function (e) {
      if (e.data && e.data.slideui === 'key') handleKey(e.data.key);
    });
    loadMath();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();