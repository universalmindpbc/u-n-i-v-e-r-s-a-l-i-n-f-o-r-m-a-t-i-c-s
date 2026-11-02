'use strict';
(() => {
  const controls = new Map();
  const values = id => [...document.getElementById(id).selectedOptions].map(o=>o.value).filter(Boolean);
  function set(id, selected) {
    const wanted = new Set(selected.filter(Boolean));
    for (const option of document.getElementById(id).options) option.selected = wanted.has(option.value);
    refresh(id);
  }
  function refresh(id) {
    const ui = controls.get(id); if (!ui) return;
    const options = [...ui.select.options].filter(o=>o.value);
    const signature = JSON.stringify(options.map(o=>[o.value,o.textContent,o.dataset.searchTerms || '']));
    if (signature !== ui.signature) {
      ui.signature = signature; ui.list.replaceChildren();
      for (const option of options) {
        const label = document.createElement('label'), input = document.createElement('input'), text = document.createElement('span');
        input.type='checkbox'; input.value=option.value; text.textContent=option.textContent; label.dataset.search=option.textContent+' '+(option.dataset.searchTerms || '');
        input.addEventListener('change',()=>{ [...ui.select.options].find(o=>o.value===input.value).selected=input.checked; ui.select.dispatchEvent(new Event('change',{bubbles:true})); refresh(id); });
        label.append(input,text); ui.list.append(label);
      }
    }
    const selected = values(id);
    for (const input of ui.list.querySelectorAll('input')) input.checked=selected.includes(input.value);
    const labels = options.filter(o=>o.selected).map(o=>o.textContent);
    ui.summary.textContent = labels.length ? labels.slice(0,2).join(', ')+(labels.length>2 ? ` +${labels.length-2}` : '') : ui.empty;
    ui.summary.setAttribute('aria-label',`${ui.name}: ${labels.length ? labels.join(', ') : ui.empty}`);
    ui.summary.setAttribute('aria-expanded',String(ui.details.open));
    ui.clear.disabled=!selected.length;
  }
  function init(ids) {
    for (const id of ids) {
      const select=document.getElementById(id), holder=select.parentElement;
      const name=holder.firstChild.textContent.trim(), empty=select.options[0].textContent;
      const original=select.value;select.multiple=true;select.hidden=true;set(id,original?[original]:[]);
      const wrapper=document.createElement('div'); wrapper.className='multi-field';
      const heading=document.createElement('span');heading.className='multi-label';heading.id=id+'-label';heading.textContent=name;
      const details=document.createElement('details');details.className='multi-select';
      const summary=document.createElement('summary');summary.id=id+'-toggle';summary.setAttribute('aria-controls',id+'-options');
      const panel=document.createElement('div');panel.className='multi-panel';panel.id=id+'-options';panel.setAttribute('role','group');panel.setAttribute('aria-labelledby',heading.id);
      const search=document.createElement('input');search.type='search';search.placeholder='Find an option…';search.setAttribute('aria-label','Find '+name.toLowerCase());
      const list=document.createElement('div');list.className='multi-options';
      search.addEventListener('input',()=>{const q=search.value.toLocaleLowerCase();for(const label of list.children)label.hidden=!label.dataset.search.toLocaleLowerCase().includes(q);});
      const clear=document.createElement('button');clear.type='button';clear.className='text-button';clear.textContent='Clear selection';
      clear.addEventListener('click',()=>{set(id,[]);select.dispatchEvent(new Event('change',{bubbles:true}));});
      panel.append(search,list,clear);details.append(summary,panel);holder.replaceWith(wrapper);wrapper.append(heading,select,details);
      controls.set(id,{select,name,empty,details,summary,list,clear,signature:''});
      details.addEventListener('toggle',()=>{summary.setAttribute('aria-expanded',String(details.open));if(details.open){for(const [other,ui] of controls)if(other!==id)ui.details.open=false;}});
      details.addEventListener('keydown',event=>{if(event.key==='Escape'&&details.open){event.preventDefault();event.stopPropagation();details.open=false;summary.focus();}});
      refresh(id);
    }
    document.addEventListener('click',event=>{for(const ui of controls.values())if(!ui.details.contains(event.target))ui.details.open=false;});
  }
  function chips(container, onChange) {
    container.replaceChildren();
    for(const [id,ui] of controls)for(const option of ui.select.selectedOptions){
      if(!option.value)continue;
      const button=document.createElement('button');button.type='button';button.className='filter-chip';
      button.textContent=option.textContent+' ×';button.setAttribute('aria-label','Remove '+ui.name+': '+option.textContent);
      button.addEventListener('click',()=>{set(id,values(id).filter(v=>v!==option.value));onChange();document.getElementById(id+'-toggle').focus();});container.append(button);
    }
  }
  globalThis.ResearchMultiSelect={init,values,set,refresh,chips};
})();
