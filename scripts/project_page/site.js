// Synchronised rows: every .pair plays its videos together, only while on screen; a play/pause button and frame steps per
// row; space / , / . act on the row under the pointer. Filters hide rows by target category. No dependencies.
(function () {
  const FPS = 16;
  const rows = Array.from(document.querySelectorAll('.pair'));
  let hovered = null;

  function vids(row) { return Array.from(row.querySelectorAll('video')); }
  function playing(row) { return vids(row).some(v => !v.paused); }
  function play(row) { vids(row).forEach(v => { v.play().catch(() => {}); }); setBtn(row, true); }
  function pause(row) { vids(row).forEach(v => v.pause()); setBtn(row, false); }
  function sync(row) { const vs = vids(row); if (!vs.length) return; const t = vs[0].currentTime; vs.slice(1).forEach(v => { v.currentTime = t; }); }
  function step(row, d) {
    pause(row);
    const vs = vids(row); if (!vs.length) return;
    const dur = vs[0].duration || 0; let t = vs[0].currentTime + d / FPS;
    if (dur) t = (t + dur) % dur;
    vs.forEach(v => { v.currentTime = t; });
  }
  function setBtn(row, on) { const b = row.querySelector('.ctl .pp'); if (b) b.textContent = on ? 'pause' : 'play'; }

  rows.forEach(row => {
    const ctl = row.querySelector('.ctl');
    if (ctl && vids(row).length) {
      ctl.innerHTML = "<button class='st' title='previous frame (,)'>‹</button><button class='pp'>play</button><button class='st' title='next frame (.)'>›</button>";
      const [prev, pp, next] = ctl.querySelectorAll('button');
      pp.addEventListener('click', () => { playing(row) ? pause(row) : play(row); });
      prev.addEventListener('click', () => step(row, -1));
      next.addEventListener('click', () => step(row, +1));
    }
    row.addEventListener('pointerenter', () => { hovered = row; });
    row.addEventListener('pointerleave', () => { if (hovered === row) hovered = null; });
    // the first video of a row drives the others: on loop restart, realign
    const vs = vids(row);
    if (vs.length > 1) vs[0].addEventListener('seeked', () => sync(row));
  });

  // play rows while they are on screen, pause them when they leave
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      const row = e.target;
      if (e.isIntersecting) { if (!row.dataset.userPaused) { sync(row); play(row); } }
      else pause(row);
    });
  }, { threshold: 0.25 });
  rows.forEach(r => io.observe(r));
  // a user's pause sticks until they press play
  rows.forEach(row => {
    const pp = row.querySelector('.ctl .pp');
    if (pp) pp.addEventListener('click', () => { if (playing(row)) delete row.dataset.userPaused; else row.dataset.userPaused = '1'; });
  });

  document.addEventListener('keydown', ev => {
    if (!hovered || ev.target.closest('input, textarea, button')) return;
    if (ev.code === 'Space') { ev.preventDefault(); if (playing(hovered)) { pause(hovered); hovered.dataset.userPaused = '1'; } else { delete hovered.dataset.userPaused; play(hovered); } }
    else if (ev.key === ',') { ev.preventDefault(); hovered.dataset.userPaused = '1'; step(hovered, -1); }
    else if (ev.key === '.') { ev.preventDefault(); hovered.dataset.userPaused = '1'; step(hovered, +1); }
  });

  // filters
  document.querySelectorAll('.filters button').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('.filters button').forEach(x => x.classList.toggle('on', x === b));
    const f = b.dataset.f;
    document.querySelectorAll('#results .rows .pair').forEach(r => r.classList.toggle('hidden', f !== 'all' && r.dataset.cat !== f));
  }));
})();
