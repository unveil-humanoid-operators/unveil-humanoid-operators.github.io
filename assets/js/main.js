/* ── Navbar: transparent → white on scroll ─────────────────── */
(function () {
  const nav = document.getElementById('main-nav');
  if (!nav) return;
  window.addEventListener('scroll', () => {
    if (window.scrollY > 40) {
      nav.classList.add('scrolled');
    } else {
      nav.classList.remove('scrolled');
    }
  }, { passive: true });
})();

/* ── Left section nav: scrollspy highlight of the current section ── */
(function () {
  const nav = document.getElementById('side-nav');
  if (!nav) return;

  const links = Array.from(nav.querySelectorAll('a[href^="#"]'));
  const entries = links
    .map(link => ({ link, section: document.getElementById(link.getAttribute('href').slice(1)) }))
    .filter(e => e.section);
  if (!entries.length) return;

  function setActive(activeLink) {
    links.forEach(l => l.classList.toggle('active', l === activeLink));
  }

  let ticking = false;
  function update() {
    ticking = false;
    // Marker line sits ~30% down the viewport; the last section crossing it wins.
    const marker = window.scrollY + window.innerHeight * 0.30;
    let current = entries[0];
    for (const e of entries) {
      const top = e.section.getBoundingClientRect().top + window.scrollY;
      if (top <= marker) current = e;
    }
    // At the very bottom of the page, force-highlight the last section.
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
      current = entries[entries.length - 1];
    }
    setActive(current.link);
  }

  function onScroll() {
    if (!ticking) { requestAnimationFrame(update); ticking = true; }
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  // Immediate feedback on click (before the smooth-scroll settles).
  links.forEach(l => l.addEventListener('click', () => setActive(l)));
  update();
})();

/* ── Generic fade-in via IntersectionObserver ──────────────── */
(function () {
  const targets = document.querySelectorAll('.fade-in');
  if (!targets.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  targets.forEach(el => observer.observe(el));
})();

/* ── Teaser pipeline: staggered reveal ─────────────────────── */
(function () {
  const pipeline = document.querySelector('.teaser-pipeline');
  if (!pipeline) return;

  const nodes   = pipeline.querySelectorAll('.pipeline-node');
  const arrows  = pipeline.querySelectorAll('.pipeline-arrow');
  const allEls  = Array.from(pipeline.children);

  const observer = new IntersectionObserver((entries) => {
    if (!entries[0].isIntersecting) return;
    observer.disconnect();

    // Reveal elements in order with staggered delays
    allEls.forEach((el, i) => {
      setTimeout(() => {
        el.classList.add('visible');
      }, i * 260);
    });

    // After everything is visible, start counter animations
    const delay = allEls.length * 260 + 100;
    setTimeout(animateCounters, delay);
  }, { threshold: 0.25 });

  observer.observe(pipeline);
})();

/* ── Counter animation for numeric attr values ─────────────── */
function animateCounters() {
  document.querySelectorAll('[data-count-to]').forEach(el => {
    const target   = parseFloat(el.dataset.countTo);
    const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 0;
    const suffix   = el.dataset.suffix   || '';
    const duration = 900; // ms
    const start    = performance.now();

    function step(now) {
      const elapsed  = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic
      const eased    = 1 - Math.pow(1 - progress, 3);
      const current  = target * eased;
      el.textContent = current.toFixed(decimals) + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
  });
}

/* ── Per-block demo controller (selector + sync) ────────────
   Each .demo-block contains its own dropdown, 3 iframes, and sync bar.
   The blocks are independent: each has its own master clock, its own
   loaded activity, and its own attribute chips. Iframes are lazy-loaded
   when the block scrolls within ~300px of the viewport.
   ──────────────────────────────────────────────────────────── */
(function () {
  let demosConfig = null;

  /* Per-activity icon (shown to the left of the label in the selector). */
  const ACTIVITY_ICONS = {
    walking:      '🚶',
    jogging:      '🏃',
    gesture:      '👋',
    dancing:      '💃',
    jumping:      '🤸',
    turning:      '🔄',
    climbing_box: '🧗',
    kneeling:     '🧎',
    pulling:      '🪢',
    hiphop:       '🕺',
    throwing:     '🤾',
    guitar:       '🎸',
  };

  /* Build a signed numeric delta string: "−4 cm", "+5 kg", "+1 yr", or "✓" for zero.
     Uses Unicode minus (U+2212) so + and − columns line up in the badge. */
  function numericDelta(pred, gt, unit) {
    const d = pred - gt;
    if (d === 0) return '✓';
    const sign = d > 0 ? '+' : '−';
    const abs  = Math.abs(d);
    const u    = unit === 'yr' ? (abs === 1 ? 'yr' : 'yrs') : unit;
    return `${sign}${abs} ${u}`;
  }

  function updateBlockChips(block, demo) {
    const p = demo.predicted, g = demo.groundTruth;
    const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

    function setRow(attr, predText, gtText, deltaText, isMatch) {
      const row = block.querySelector(`.comparison-row[data-attr="${attr}"]`);
      if (!row) return;
      row.classList.toggle('match', isMatch);
      const predEl = row.querySelector('.comparison-pred');
      const gtEl   = row.querySelector('.comparison-gt');
      const txtEl  = row.querySelector('.delta-text');
      if (predEl) predEl.textContent = predText;
      if (gtEl)   gtEl.textContent   = gtText;
      if (txtEl)  txtEl.textContent  = deltaText;
    }

    setRow('height',
      p.height + ' cm', g.height + ' cm',
      numericDelta(p.height, g.height, 'cm'),
      p.height === g.height);

    setRow('weight',
      p.weight + ' kg', g.weight + ' kg',
      numericDelta(p.weight, g.weight, 'kg'),
      p.weight === g.weight);

    setRow('age',
      p.age + ' yrs', g.age + ' yrs',
      numericDelta(p.age, g.age, 'yr'),
      p.age === g.age);

    setRow('gender',
      cap(p.gender), cap(g.gender),
      p.gender === g.gender ? '✓' : `→ ${cap(g.gender)}`,
      p.gender === g.gender);
  }

  function blockIframes(block) {
    return Array.from(block.querySelectorAll('iframe'));
  }

  /* Per-activity playback speed. Keep the slow-motion 0.5×, but if a clip would
     run longer than 15s at 0.5× (i.e. its natural length exceeds 7.5s), play it
     at 1× so the long clips don't drag. */
  function autoSpeedFor(demo) {
    const naturalSec = demo.numFrames / demo.fps;
    return (naturalSec / 0.5) > 15 ? 1 : 0.5;
  }
  function applyAutoSpeed(block, demo) {
    const sel = block.querySelector('.sync-speed');
    if (!sel) return;
    sel.value = String(autoSpeedFor(demo));
    sel.dispatchEvent(new Event('change'));   // notify the per-block sync controller
  }

  function sendDemoToBlock(block, demo) {
    const iframes = blockIframes(block);
    iframes.forEach(ifr => ifr.closest('.demo-panel')?.classList.add('loading'));
    updateBlockChips(block, demo);
    iframes.forEach(ifr => {
      ifr.contentWindow?.postMessage({ type: 'LOAD_DEMO', demo }, '*');
    });
    block._seqDuration = demo.numFrames / demo.fps;
    applyAutoSpeed(block, demo);
    setTimeout(() => {
      iframes.forEach(ifr => ifr.closest('.demo-panel')?.classList.remove('loading'));
    }, 3500);
  }

  /* ── Lazy iframe loading: hydrate src= when the block enters the viewport ── */
  const lazyObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.querySelectorAll('iframe[data-src]').forEach(ifr => {
        ifr.src = ifr.dataset.src;
        ifr.removeAttribute('data-src');
      });
      lazyObserver.unobserve(entry.target);
    });
  }, { rootMargin: '300px 0px' });

  /* ── Per-block sync controller ─────────────────────────────── */
  function attachSync(block) {
    const btnPlay  = block.querySelector('.sync-play');
    const progEl   = block.querySelector('.sync-progress');
    const timeEl   = block.querySelector('.sync-time');
    const speedSel = block.querySelector('.sync-speed');
    if (!btnPlay) return;

    let masterTime = 0;
    let playing    = false;
    let lastTS     = null;
    let speed      = parseFloat(speedSel?.value) || 1;

    function iframes() {
      return blockIframes(block).map(el => el.contentWindow).filter(Boolean);
    }
    function broadcast() {
      const msg = { type: 'SYNC', time: masterTime, playing, speed };
      iframes().forEach(w => w.postMessage(msg, '*'));
    }
    function setPlaying(val) {
      playing = val;
      btnPlay.innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
      if (playing) lastTS = null;
      broadcast();
    }

    btnPlay.addEventListener('click', () => setPlaying(!playing));
    progEl.addEventListener('input', () => {
      const dur = block._seqDuration || (3303 / 120);
      masterTime = parseFloat(progEl.value) * dur;
      broadcast();
    });
    speedSel.addEventListener('change', () => {
      speed = parseFloat(speedSel.value);
      broadcast();
    });

    function tick(ts) {
      requestAnimationFrame(tick);
      const dur = block._seqDuration || (3303 / 120);
      if (playing) {
        if (lastTS === null) lastTS = ts;
        const dt = (ts - lastTS) * 0.001 * speed;
        lastTS = ts;
        masterTime += dt;
        if (masterTime >= dur) masterTime %= dur;
      } else {
        lastTS = null;
      }
      broadcast();
      progEl.value = masterTime / dur;
      timeEl.textContent = masterTime.toFixed(2) + ' / ' + dur.toFixed(2) + 's';
    }

    /* Auto-play this block once one of its iframes posts READY */
    window.addEventListener('message', e => {
      if (!e.data || e.data.type !== 'READY' || playing) return;
      const ifrs = blockIframes(block);
      if (ifrs.some(ifr => ifr.contentWindow === e.source)) {
        setPlaying(true);
      }
    });

    requestAnimationFrame(tick);
  }

  /* ── Bootstrap each block ──────────────────────────────────── */
  async function init() {
    const blocks = document.querySelectorAll('.demo-block');
    if (!blocks.length) return;

    let cfg;
    try {
      cfg = await fetch('./assets/demos/demos_config.json').then(r => r.json());
      demosConfig = cfg;
    } catch (e) {
      console.warn('demos_config not ready yet:', e.message);
      cfg = { demos: [], defaultDemo: null };
    }

    blocks.forEach(block => {
      const list       = block.querySelector('.activity-list');
      const defaultId  = block.dataset.defaultDemo || cfg.defaultDemo;

      /* Populate the activity list (one clickable rectangle per activity).
         The default (selected) activity is pulled to the top of the list. */
      if (list && cfg.demos.length) {
        list.innerHTML = '';
        const ordered = cfg.demos.filter(d => d.id === defaultId)
                        .concat(cfg.demos.filter(d => d.id !== defaultId));
        ordered.forEach(demo => {
          const item = document.createElement('button');
          item.type = 'button';
          item.className = 'activity-item';
          item.dataset.demoId = demo.id;
          item.setAttribute('role', 'option');

          const icon = document.createElement('span');
          icon.className = 'activity-icon';
          icon.setAttribute('aria-hidden', 'true');
          icon.textContent = ACTIVITY_ICONS[demo.id] || '•';

          const text = document.createElement('span');
          text.className = 'activity-text';
          text.textContent = demo.label;

          item.appendChild(icon);
          item.appendChild(text);

          const isDefault = demo.id === defaultId;
          item.classList.toggle('active', isDefault);
          item.setAttribute('aria-selected', isDefault ? 'true' : 'false');
          /* Click → load this activity into the block and mark it active. */
          item.addEventListener('click', () => {
            if (item.classList.contains('active')) return;
            list.querySelectorAll('.activity-item').forEach(el => {
              el.classList.remove('active');
              el.setAttribute('aria-selected', 'false');
            });
            item.classList.add('active');
            item.setAttribute('aria-selected', 'true');
            sendDemoToBlock(block, demo);
          });
          list.appendChild(item);
        });
      }

      /* Seed seqDuration + sync comparison rows from the block's default demo. */
      const defaultDemo = cfg.demos.find(d => d.id === defaultId);
      if (defaultDemo) {
        block._seqDuration = defaultDemo.numFrames / defaultDemo.fps;
        updateBlockChips(block, defaultDemo);
      }

      attachSync(block);
      if (defaultDemo) applyAutoSpeed(block, defaultDemo);
      lazyObserver.observe(block);
    });
  }

  init();
  window.getDemosConfig = () => demosConfig;
})();

/* ── Demo panel fade-in with staggered delay ───────────────── */
(function () {
  const panels = document.querySelectorAll('.demo-panel[data-delay]');
  if (!panels.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el    = entry.target;
      const delay = parseInt(el.dataset.delay || 0);
      setTimeout(() => el.classList.add('visible'), delay);
      observer.unobserve(el);
    });
  }, { threshold: 0.15 });

  panels.forEach(p => observer.observe(p));
})();

/* ── PDF.js: render PDF figures to canvas ──────────────────── */
(function () {
  function renderPDFs() {
    const canvases = document.querySelectorAll('canvas[data-pdf]');
    if (!canvases.length) return;

    const pdfjsLib = window['pdfjs-dist/build/pdf'];
    if (!pdfjsLib) { console.warn('PDF.js not loaded'); return; }

    pdfjsLib.GlobalWorkerOptions.workerSrc =
      'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

    canvases.forEach(async (canvas) => {
      try {
        const pdf      = await pdfjsLib.getDocument(canvas.dataset.pdf).promise;
        const page     = await pdf.getPage(1);
        const dpr      = window.devicePixelRatio || 1;
        const contW    = canvas.parentElement.offsetWidth || 900;
        const baseVp   = page.getViewport({ scale: 1 });
        const scale    = (contW / baseVp.width) * dpr;
        const viewport = page.getViewport({ scale });

        canvas.width  = viewport.width;
        canvas.height = viewport.height;

        await page.render({
          canvasContext: canvas.getContext('2d'),
          viewport,
        }).promise;

        canvas.classList.add('rendered');
        // Hide fallback link once canvas is drawn
        const fallback = canvas.nextElementSibling;
        if (fallback?.classList.contains('pdf-fallback')) fallback.style.display = 'none';
      } catch (err) {
        console.error('PDF render error:', canvas.dataset.pdf, err);
      }
    });
  }

  // Defer until after layout so offsetWidth is reliable
  if (document.readyState === 'complete') {
    renderPDFs();
  } else {
    window.addEventListener('load', renderPDFs);
  }
})();

/* ── Smooth close of mobile nav on link click ──────────────── */
(function () {
  const toggler    = document.querySelector('.navbar-toggler');
  const navCollapse = document.getElementById('navbarLinks');
  if (!navCollapse) return;

  document.querySelectorAll('#navbarLinks .nav-link').forEach(link => {
    link.addEventListener('click', () => {
      if (navCollapse.classList.contains('show')) {
        toggler && toggler.click();
      }
    });
  });
})();
