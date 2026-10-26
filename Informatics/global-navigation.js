'use strict';
// Shared labels, order and destinations for every Universal Informatics menu.
(() => {
  const root = new URL('.', document.currentScript.src);
  const entries = Object.freeze([
    ['about', 'About'],
    ['geometry', 'Geometry Atlas'],
    ['transcription', 'Transcription Atlas'],
    ['molecular', 'Molecular Atlas'],
    ['quantum', 'Quantum Atlas'],
    ['sound', 'Sound Atlas'],
    ['vision', 'Vision Atlas'],
    ['language', 'Language Atlas'],
    ['research', 'Universal Research'],
    ['connect', 'Connect your AI'],
    ['partners', 'Partners']
  ].map(([id, label]) => Object.freeze({id, label})));
  window.InformaticsNavigation = Object.freeze({entries});
  for (const menu of document.querySelectorAll('[data-global-menu]')) {
    const surface = menu.dataset.globalMenu;
    menu.replaceChildren(...entries.map(({id, label}) => {
      const item = document.createElement(surface === 'main' ? 'button' : 'a');
      item.setAttribute('role', 'menuitem'); item.tabIndex = -1; item.textContent = label;
      if (surface === 'main') {
        if (id === 'about') item.dataset.about = '';
        else if (id === 'transcription') item.dataset.area = id;
        else if (id === 'research') item.dataset.area = id;
        else if (id === 'connect') item.dataset.area = id;
        else item.dataset.area = id;
      } else {
        const url = new URL(id === 'research' ? 'library.html' : id === 'connect' ? 'connect.html' : './', root);
        if (id === 'about') { url.searchParams.set('about', '1'); item.dataset.globalAbout = ''; }
        else if (id === 'connect') item.dataset.globalConnect = '';
        else if (id !== 'research') { url.searchParams.set('area', id); url.searchParams.set('entry', 'research'); item.dataset.globalArea = id; }
        item.href = url.href;
        if ((surface === 'library' && id === 'research') || (surface === 'molecular' && id === 'molecular') || (surface === 'connect' && id === 'connect')) item.setAttribute('aria-current', 'page');
      }
      return item;
    }));
  }
})();
