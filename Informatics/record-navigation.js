'use strict';
(()=>{
 const $=selector=>document.querySelector(selector),nav=$('#record-navigation'),menu=$('#record-menu'),toggle=$('#record-toggle'),search=$('#record-search'),results=$('#record-results'),status=$('#record-status');
 const M9='fibonacci-m9-p24';let records=[],filter='all',selection={id:M9,mode:'animation'},loading=null,timer,pinned=false;
 function updateSelection(){
  const item=records.find(record=>record.id===selection.id),match=selection.id.match(/^fibonacci-m(\d+)-p(\d+)$/);
  const label=selection.id==='m-summary'?'M9 family':item?'M'+item.modulus+' / P'+item.pisano_period:match?'M'+match[1]+' / P'+match[2]:'M9 / P24';
  $('#record-label').textContent=label;toggle.setAttribute('aria-label',label+' — '+(selection.mode==='animation'?'Animation':'Research view')+'. Select a sequence or view');
  menu.querySelectorAll('[data-record]').forEach(button=>{
   const active=button.dataset.record===selection.id&&(button.dataset.mode||'research')===selection.mode;
   if(button.closest('.record-pinned'))button.setAttribute('aria-pressed',String(active));
   button.setAttribute('aria-current',active?'page':'false');
  });
 }
 function matching(record){
  if(record.id===M9||(filter==='family'&&!record.m9_family))return false;
  const query=search.value.trim().toLowerCase().replace(/\s+/g,'');if(!query)return true;
  const modulus=query.match(/^m?(\d+)(?:\/p?(\d+)|p(\d+))?$/),period=query.match(/^p(\d+)$/);
  if(modulus)return record.modulus===Number(modulus[1])&&(!(modulus[2]||modulus[3])||record.pisano_period===Number(modulus[2]||modulus[3]));
  if(period)return record.pisano_period===Number(period[1]);
  return ('m'+record.modulus+'/p'+record.pisano_period).includes(query);
 }
 function render(){
  const matches=records.filter(matching),fragment=document.createDocumentFragment();
  for(const record of matches){const button=document.createElement('button');button.type='button';button.tabIndex=-1;button.dataset.record=record.id;button.textContent='M'+record.modulus+' / P'+record.pisano_period;if(record.m9_family){const note=document.createElement('span');note.textContent='M9 family';button.append(note);}fragment.append(button);}
  results.replaceChildren(fragment);const query=search.value.trim();status.textContent=!query?(filter==='family'?'111 M9-family records · M9 views above':'M1–M1001 · M9 views above'):matches.length?matches.length+' research record'+(matches.length===1?'':'s'):/^m?9(?:\s*\/?\s*p?24)?$/i.test(query)?'M9 / P24 views are above.':'No matching records.';
  menu.querySelectorAll('[data-record-filter]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.recordFilter===filter)));updateSelection();
 }
 async function load(){
  if(loading)return loading;status.textContent='Loading sequences…';$('#record-retry').hidden=true;
  loading=(async()=>{try{const response=await fetch('pilot-v1/index.json?v=m1001-v1');if(!response.ok)throw Error('Sequence index unavailable');const index=await response.json();records=index.records;render();}catch{status.textContent='Sequence list unavailable. Try again.';$('#record-retry').hidden=false;}finally{loading=null;}})();return loading;
 }
 function close({focus=false}={}){clearTimeout(timer);menu.hidden=true;pinned=false;toggle.setAttribute('aria-expanded','false');if(focus)toggle.focus({preventScroll:true});}
 function placeMenu(){menu.style.setProperty('--record-menu-top',Math.ceil(document.querySelector('#experience header').getBoundingClientRect().bottom)+'px');}
 window.addEventListener('resize',()=>{if(!menu.hidden)placeMenu();});
 function open({focus=false}={}){
  placeMenu();clearTimeout(timer);document.dispatchEvent(new Event('atlas:open-records'));menu.hidden=false;toggle.setAttribute('aria-expanded','true');
  if(focus){pinned=true;const active=menu.querySelector('[data-record][aria-current=page]')||menu.querySelector('[data-record]');active.focus({preventScroll:true});if(active.parentElement===results)active.scrollIntoView({block:'nearest'});}
 }
 toggle.addEventListener('click',()=>!menu.hidden&&pinned?close({focus:true}):open({focus:true}));
 toggle.addEventListener('keydown',event=>{if(event.key==='ArrowDown'||event.key==='ArrowUp'){event.preventDefault();open({focus:true});}});
 nav.addEventListener('pointerenter',event=>{if(event.pointerType==='mouse')open();});
 nav.addEventListener('pointerleave',()=>{timer=setTimeout(()=>{if(!pinned&&!menu.contains(document.activeElement))close();},180);});
 nav.addEventListener('focusout',()=>setTimeout(()=>{if(!nav.contains(document.activeElement))close();},0));
 document.addEventListener('pointerdown',event=>{if(!nav.contains(event.target))close();});
 document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!menu.hidden){event.preventDefault();close({focus:true});}});
 document.addEventListener('atlas:close-records',()=>close());
 document.addEventListener('atlas:selection',event=>{selection=event.detail;updateSelection();});
 menu.addEventListener('click',event=>{
  const record=event.target.closest('[data-record]');if(record){close();document.dispatchEvent(new CustomEvent('atlas:choose-record',{detail:{id:record.dataset.record,mode:record.dataset.mode||'research'}}));return;}
  const choice=event.target.closest('[data-record-filter]');if(choice){filter=choice.dataset.recordFilter;render();results.scrollTop=0;}
 });
 menu.addEventListener('keydown',event=>{
  if(event.target===search){if(event.key==='ArrowDown'){event.preventDefault();(results.querySelector('button')||menu.querySelector('[data-record]')).focus();}return;}
  if(!event.target.matches('[data-record]')||!['ArrowDown','ArrowUp','Home','End'].includes(event.key))return;
  event.preventDefault();const choices=[...menu.querySelectorAll('[data-record]')],index=choices.indexOf(event.target),next=event.key==='Home'?0:event.key==='End'?choices.length-1:(index+(event.key==='ArrowDown'?1:-1)+choices.length)%choices.length;choices[next].focus();
 });
 search.addEventListener('input',()=>{render();results.scrollTop=0;});$('#record-retry').addEventListener('click',load);
 load();
})();
