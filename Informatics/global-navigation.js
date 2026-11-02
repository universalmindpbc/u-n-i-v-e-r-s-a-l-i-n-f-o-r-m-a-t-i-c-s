'use strict';
// Shared labels, order and destinations for every Universal Informatics menu.
(() => {
  const root = new URL('.', document.currentScript.src);
  const entries = Object.freeze(window.PortalRoutes.data.portals.find(p=>p.id==='informatics').entries.map(item=>Object.freeze({...item})));
  window.InformaticsNavigation = Object.freeze({entries});
  const entryURL=id=>new URL(entries.find(e=>e.id===id).url,location.origin).href;
  for (const menu of document.querySelectorAll('[data-global-menu]')) {
    const surface = menu.dataset.globalMenu;
    menu.replaceChildren(...entries.map(({id, label}) => {
      const item = document.createElement('a');
      item.href = entryURL(id); item.setAttribute('role', 'menuitem'); item.tabIndex = -1; item.textContent = label;
      if (surface === 'main') {
        if (id === 'about') item.dataset.about = '';
        else if (id === 'transcription') item.dataset.area = id;
        else if (id === 'research') item.dataset.area = id;
        else if (id === 'connect') item.dataset.area = id;
        else item.dataset.area = id;
      } else {
        const url = new URL(entryURL(id));
        if (id === 'about') { item.dataset.globalAbout = ''; }
        else if (id === 'connect') item.dataset.globalConnect = '';
        else if (id !== 'research') { item.dataset.globalArea = id; }
        item.href = url.href;
        if ((surface === 'library' && id === 'research') || (surface === 'molecular' && id === 'molecular') || (surface === 'connect' && id === 'connect')) item.setAttribute('aria-current', 'page');
      }
      return item;
    }));
  }
})();
