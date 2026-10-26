'use strict';
(() => {
  const themeKey = 'informatics-library-background';
  const motionKey = 'informatics-library-video-paused';
  const modes = ['video', 'black', 'white'];
  const root = document.documentElement;
  const read = key => { try { return localStorage.getItem(key); } catch { return null; } };
  const save = (key, value) => { try { localStorage.setItem(key, value); } catch {} };
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let mode = modes.includes(read(themeKey)) ? read(themeKey) : 'video';
  let paused = read(motionKey) === null ? reducedMotion.matches : read(motionKey) === 'true';
  let film;const shared=window.parent!==window&&!!parent.document.querySelector('#experience');
  root.dataset.readingBackground = mode;

  function syncControls() {
    document.querySelectorAll('button[data-reading-background]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.readingBackground === mode));
    });
    document.querySelectorAll('[data-reading-motion]').forEach(button => {
      button.hidden = shared || mode !== 'video';
      const stopped = paused || film?.paused;
      button.textContent = stopped ? 'Play video' : 'Pause video';
      button.setAttribute('aria-label', stopped ? 'Play background video' : 'Pause background video');
    });
  }

  function notifyParent() {
    if (window.parent !== window) parent.postMessage({type: 'informatics:library-background', mode}, location.origin);
  }

  function syncFilm() {
    if (!film) return;
    if(shared){syncControls();return;}
    if (mode !== 'video' || document.hidden) {
      film.pause();
    } else {
      film.muted = true;
      film.defaultMuted = true;
      film.volume = 0;
      if (!film.getAttribute('src')) film.src = film.dataset.src;
      if (paused) film.pause();
      else film.play().catch(() => syncControls());
    }
    syncControls();
  }

  function setMode(value, persist = true) {
    mode = modes.includes(value) ? value : 'video';
    root.dataset.readingBackground = mode;
    if (persist) save(themeKey, mode);
    syncFilm();
    syncControls();
    notifyParent();
  }

  function init() {
    if(shared)document.body.classList.add('embedded');
    document.querySelectorAll('[data-reading-controls]').forEach(host => {
      host.innerHTML = '<div class="reading-options" role="group" aria-label="Reading background">' +
        modes.map(value => `<button type="button" data-reading-background="${value}" aria-pressed="${value === mode}">${value[0].toUpperCase() + value.slice(1)}</button>`).join('') +
        '</div><button type="button" class="reading-motion" data-reading-motion>Pause video</button>';
    });
    film = shared?parent.document.getElementById('film'):document.getElementById('library-film');
    if(shared)document.querySelectorAll('[data-reading-motion]').forEach(b=>b.hidden=true);
    for (const event of ['play', 'pause', 'error']) film.addEventListener(event, syncControls);
    document.addEventListener('click', event => {
      const option = event.target.closest('button[data-reading-background]');
      if (option) setMode(option.dataset.readingBackground);
      if (event.target.closest('[data-reading-motion]')) {
        paused = !film.paused;
        save(motionKey, String(paused));
        syncFilm();
      }
    });
    document.addEventListener('visibilitychange', syncFilm);
    window.addEventListener('pagehide', () => {if(!shared)film.pause();});
    reducedMotion.addEventListener('change', () => {
      if (read(motionKey) === null) { paused = reducedMotion.matches; syncFilm(); }
    });
    setMode(mode, false);
  }

  window.addEventListener('storage', event => {
    if (event.key === themeKey || event.key === null) setMode(read(themeKey), false);
    if (event.key === motionKey || event.key === null) {
      paused = read(motionKey) === null ? reducedMotion.matches : read(motionKey) === 'true';
      syncFilm();
    }
  });
  document.addEventListener('DOMContentLoaded', init, {once: true});
})();
