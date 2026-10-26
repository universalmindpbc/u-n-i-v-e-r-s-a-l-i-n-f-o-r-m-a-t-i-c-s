'use strict';
(()=>{
 const $=s=>document.querySelector(s),frames={research:$('#research-frame'),connect:$('#connect-frame')};
 const initial=new URLSearchParams(location.search);let forwarded=null;try{forwarded=JSON.parse(initial.get('section'));}catch{}
 const navigate=area=>{document.querySelectorAll('dialog[open]').forEach(d=>d.close());document.dispatchEvent(new CustomEvent('atlas:navigate',{detail:area}));};
 function load(area,extra){const frame=frames[area];if(!frame)return;if(frame.src&&!extra)return;const url=new URL(area==='connect'?'connect.html':'library.html',location.href);if(forwarded?.area===area){url.search=forwarded.search||'';url.hash=forwarded.hash||'';forwarded=null;}if(extra)for(const[k,v]of Object.entries(extra))if(v)url.searchParams.set(k,v);url.searchParams.set('embedded','1');frame.src=url.href;}
 document.addEventListener('atlas:area',e=>load(e.detail));
 document.addEventListener('click',e=>{const link=e.target.closest('[data-library],[data-connect]');if(!link)return;e.preventDefault();e.stopImmediatePropagation();const area=link.hasAttribute('data-connect')?'connect':'research';load(area,area==='research'?{collection:link.dataset.library||'all',target:link.dataset.libraryGene||''}:null);navigate(area);},true);
 const known=()=>[...document.querySelectorAll('#atlas-frame,#molecular-frame,#quantum-frame,#research-frame,#connect-frame')];
 window.addEventListener('message',e=>{if(e.origin!==location.origin||!known().some(f=>f.contentWindow===e.source))return;const data=e.data;if(data?.type==='informatics:connect')navigate('connect');else if(data?.type==='informatics:library-navigate'&&window.InformaticsNavigation.entries.some(x=>x.id===data.area))navigate(data.area);else if(data?.type==='informatics:close-library')navigate('geometry');});
 for(const frame of known())frame.addEventListener('load',()=>{
  if(!frame.getAttribute('src'))return;
  const doc=frame.contentDocument;if(!doc)return;
  doc.addEventListener('pointerdown',()=>document.dispatchEvent(new Event('atlas:close-navigation')));
  doc.addEventListener('click',e=>{const a=e.target.closest('a[data-global-area],a[data-global-about],a[data-global-connect]');if(!a||e.ctrlKey||e.metaKey||e.shiftKey||e.altKey)return;e.preventDefault();e.stopImmediatePropagation();navigate(a.dataset.globalArea||(a.hasAttribute('data-global-connect')?'connect':'about'));},true);
 });
})();
