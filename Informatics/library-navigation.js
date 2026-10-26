'use strict';
(() => {
  const nav = document.querySelector('#atlas-navigation');
  const menu = document.querySelector('#atlas-menu');
  const toggle = document.querySelector('#atlas-toggle');
  const arrow = document.querySelector('#atlas-menu-toggle');
  let timer;
  function close(focus = false) {
    clearTimeout(timer);
    menu.hidden = true;
    for (const button of [toggle, arrow]) button.setAttribute('aria-expanded', 'false');
    if (focus) toggle.focus();
  }
  function open(focus = false, last = false) {
    clearTimeout(timer);
    menu.hidden = false;
    for (const button of [toggle, arrow]) button.setAttribute('aria-expanded', 'true');
    if (focus) {
      const items = menu.querySelectorAll('[role=menuitem]');
      items[last ? items.length - 1 : 0].focus();
    }
  }
  for (const button of [toggle, arrow]) {
    button.onclick = () => button === toggle || menu.hidden ? open(true) : close(true);
    button.onkeydown = event => {
      if (['ArrowDown', 'ArrowUp'].includes(event.key)) {
        event.preventDefault(); open(true, event.key === 'ArrowUp');
      }
    };
  }
  nav.addEventListener('pointerenter', event => { if (event.pointerType === 'mouse') open(); });
  nav.addEventListener('pointerleave', () => { timer = setTimeout(() => { if (!nav.contains(document.activeElement)) close(); }, 160); });
  nav.addEventListener('focusout', () => setTimeout(() => { if (!nav.contains(document.activeElement)) close(); }, 0));
  menu.addEventListener('keydown', event => {
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const items = [...menu.querySelectorAll('[role=menuitem]')];
    const index = items.indexOf(document.activeElement);
    items[event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length].focus();
  });
  document.addEventListener('pointerdown', event => { if (!nav.contains(event.target)) close(); });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !menu.hidden) {
      event.preventDefault(); event.stopImmediatePropagation(); close(true);
    }
  });
  document.addEventListener('click', event => {
    const link = event.target.closest('[data-global-area], [data-global-about], [data-global-connect]');
    if (!link || window.parent === window || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault(); close();
    parent.postMessage({type: link.hasAttribute('data-global-connect') ? 'informatics:connect' : 'informatics:library-navigate', area: link.dataset.globalArea || 'about'}, location.origin);
  });
})();
