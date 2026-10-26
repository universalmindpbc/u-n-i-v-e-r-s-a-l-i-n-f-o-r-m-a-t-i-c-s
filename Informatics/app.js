'use strict';
(async()=>{
const $=s=>document.querySelector(s),NS='http://www.w3.org/2000/svg';
const response=await fetch('/Informatics/geometry.json?v=1');if(!response.ok)throw Error('Geometry unavailable');const data=await response.json();
const make=(tag,attrs,parent)=>{const el=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));parent.append(el);return el};
const pt=(i,n,r=240)=>[400+r*Math.sin(2*Math.PI*i/n),400-r*Math.cos(2*Math.PI*i/n)];
const path=(points,parent,attrs={})=>make('path',{d:points.map((p,i)=>`${i?'L':'M'}${p.join(',')}`).join(' '),fill:'none',stroke:'#e4e9e9','stroke-width':1.4,'stroke-linejoin':'round','stroke-linecap':'round',...attrs},parent);
const text=(value,p,parent,attrs={})=>{const el=make('text',{x:p[0],y:p[1],fill:'#a5afb3','font-size':14,...attrs},parent);el.textContent=value;return el};
const circle=make('circle',{cx:400,cy:400,r:240,fill:'none',stroke:'#5ca8ff','stroke-width':.7},$('#alpha'));
const labels=[];for(let i=0;i<24;i++){const g=make('g',{},$('#alpha'));const node=data.node_indices.includes(i);path([pt(i,24,246),pt(i,24,node?256:251)],g,{stroke:node?'#c0c9cd':'#465157','stroke-width':.7});text(data.display_alpha[i],pt(i,24,275),g,{fill:node?'#f0f2f2':'#899598','font-size':node?20:14});text(i===0?'24':String(i).padStart(2,'0'),pt(i,24,310),g,{'font-size':8,fill:'#4f6068'});if(node)make('circle',{cx:pt(i,24)[0],cy:pt(i,24)[1],r:2,fill:'#e7eeee'},g);labels.push(g);}
for(const r of [330,350])make('circle',{cx:400,cy:400,r,fill:'none',stroke:'#4b5a62','stroke-width':.4,'stroke-dasharray':r===330?'1 9':'30 170'},$('#guides'));
path([[390,400],[410,400]],$('#guides'),{stroke:'#344249','stroke-width':.6});path([[400,390],[400,410]],$('#guides'),{stroke:'#344249','stroke-width':.6});
const uni=path(data.unicursal_indices.map(i=>pt(i,24)),$('#unicursal'),{'stroke-width':2});
const triangles=data.triangles.map(indices=>path(indices.map(i=>pt(i,24)),$('#stars'),{stroke:'#ff84c5','stroke-width':1,fill:'#ff389c','fill-opacity':.18}));
// Circumradius equals the Alpha circle radius: all six vertices touch it.
const hex=path(Array.from({length:7},(_,i)=>pt(i,6,240)),$('#hexagon'),{stroke:'#62d998','stroke-width':1.5,fill:'#37be79','fill-opacity':.045});
const betaCircle=make('circle',{cx:400,cy:400,r:240,fill:'none',stroke:'#5ca8ff','stroke-width':.6},$('#beta'));
for(let i=0;i<9;i++){const p=pt(i,9);make('circle',{cx:p[0],cy:p[1],r:2,fill:'#a5b1b6'},$('#beta'));text(i||9,pt(i,9,278),$('#beta'),{'font-size':18,fill:data.doubling.includes(i)?'#f0f3f3':'#56646b'});}
const doubling=path(data.doubling.map(i=>pt(i,9)),$('#beta'),{'stroke-width':2.2,stroke:'#ffea42'});
const paths=[circle,uni,...triangles,hex,doubling];paths.forEach(p=>{p.dataset.length=p.getTotalLength();p.style.strokeDasharray=p.dataset.length});
const draw=(p,t)=>p.style.strokeDashoffset=Number(p.dataset.length)*(1-Math.max(0,Math.min(1,t)));
const duration=7000,total=duration*5;let elapsed=0,last=null,playing=false,entered=false;const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const ease=t=>Math.max(0,Math.min(1,t));
function render(){const stage=Math.min(4,Math.floor(elapsed/duration)),fraction=(elapsed%duration)/duration;document.body.dataset.stage=stage;$('#counter').textContent=`0${stage+1} / 05`;$('#progress').style.width=`${elapsed/total*100}%`;document.querySelectorAll('button[data-stage]').forEach(b=>b.setAttribute('aria-pressed',Number(b.dataset.stage)===stage));
$('#alpha').style.opacity=stage===4?.10:1;labels.forEach((g,i)=>g.style.opacity=stage>0?1:ease(fraction*30-i));draw(circle,stage>0?1:fraction*2);
$('#unicursal').style.opacity=stage<1?0:1;draw(uni,stage===1?fraction*1.3:stage>1?1:0);
$('#stars').style.opacity=stage<2?0:.65;triangles.forEach((p,i)=>{const t=stage===2?ease(fraction*2-i):stage>2?1:0;draw(p,t);p.setAttribute('fill-opacity',.18*t);});
$('#hexagon').style.opacity=stage>=3?1:0;draw(hex,stage===3?fraction*1.5:stage>3?1:0);
$('#beta').style.opacity=stage===4?1:0;draw(doubling,stage===4?fraction*1.3:0);
const sequences=['0 · 1 · 1 · 2 · 3 · 5 · 8','9 → 3 → 6 → 9 → 3 → 6','9 · 3 · 6   /   3 · 9 · 6','α · γ','1 → 2 → 4 → 8 → 7 → 5'];$('#sequence').textContent=sequences[stage];
const active=stage===1?uni:stage===4?doubling:null;const tracer=$('#tracer');tracer.style.opacity=active&&fraction<1/1.3?1:0;if(active){const p=active.getPointAtLength(Number(active.dataset.length)*ease(fraction*1.3));tracer.setAttribute('cx',p.x);tracer.setAttribute('cy',p.y);}
$('#play').textContent=playing?'Ⅱ':'▷';$('#play').setAttribute('aria-label',playing?'Pause animation':'Play animation');}
$('#play').onclick=()=>{playing=!playing;if(playing&&elapsed>=total-1)elapsed=0;last=null;render()};$('#replay').onclick=()=>{elapsed=0;playing=true;last=null;render()};document.querySelectorAll('button[data-stage]').forEach(b=>b.onclick=()=>{elapsed=Number(b.dataset.stage)*duration+duration-1;playing=false;render()});
const film=$('#film'),experience=$('#experience'),launch=$('#launch');
const M9='fibonacci-m9-p24',initialParams=new URLSearchParams(location.search);
let area='home',resumeAnimation=true,recordId=initialParams.get('record')||M9,recordMode=initialParams.get('record-mode')||((initialParams.get('area')==='geometry'||initialParams.get('atlas')==='1'||recordId!==M9)?'research':'animation'),recordView=location.hash.slice(1);
const researchFrame=$('#atlas-frame');
function announceRecord(){document.dispatchEvent(new CustomEvent('atlas:selection',{detail:{id:recordId,mode:recordMode}}));}
function syncURL(){const url=new URL(location.href);url.searchParams.delete('atlas');if(area==='home'){url.searchParams.delete('area');url.searchParams.delete('record');url.searchParams.delete('record-mode');url.hash='';}else{url.searchParams.set('area',area);url.searchParams.set('record',recordId);url.searchParams.set('record-mode',recordMode);url.hash=area==='geometry'?recordView:'';}history.replaceState(null,'',url);}
function loadResearch(){
 researchFrame.dataset.record=recordId;researchFrame.dataset.view=recordView;
 if(!researchFrame.getAttribute('src'))researchFrame.src=researchFrame.dataset.src+'&record='+encodeURIComponent(recordId)+(recordView?'#'+recordView:'');
 else researchFrame.contentWindow.AtlasResearch?.selectRecord(recordId,recordView);
}
researchFrame.addEventListener('load',()=>{researchFrame.contentDocument.addEventListener('pointerdown',()=>{closeMenu();document.dispatchEvent(new Event('atlas:close-records'));});if(area==='geometry')loadResearch();});
function chooseRecord(id,mode='research',view){
 if(mode==='animation')id=M9;
 if(id!==recordId)recordView='';if(view!==undefined)recordView=view;
 recordId=id;recordMode=mode;showArea(mode==='animation'?'home':'geometry');$('#record-toggle').focus({preventScroll:true});
}
document.addEventListener('atlas:choose-record',event=>chooseRecord(event.detail.id,event.detail.mode));
document.addEventListener('atlas:record',event=>{const detail=event.detail;if(area!=='geometry'||(!detail.user&&detail.id!==recordId))return;recordId=detail.id;recordMode='research';recordView=detail.view||'';researchFrame.dataset.record=recordId;researchFrame.dataset.view=recordView;syncURL();announceRecord();});
const areas=['about','research','connect','geometry','transcription','molecular','quantum','genomics','sound','vision','language','partners'];
const menu=$('#atlas-menu'),menuToggle=$('#atlas-menu-toggle'),atlasNav=$('#atlas-navigation');let menuTimer;
function closeMenu({focus=false}={}){clearTimeout(menuTimer);menu.hidden=true;menuToggle.setAttribute('aria-expanded','false');$('#atlas-toggle').setAttribute('aria-expanded','false');if(focus)menuToggle.focus();}
function openMenu({focus=false,lastItem=false}={}){document.dispatchEvent(new Event('atlas:close-records'));clearTimeout(menuTimer);menu.hidden=false;menuToggle.setAttribute('aria-expanded','true');$('#atlas-toggle').setAttribute('aria-expanded','true');if(focus){const items=menu.querySelectorAll('[role=menuitem]');items[lastItem?items.length-1:0].focus();}}
function showArea(name,{focus=false}={}){
 if(!areas.includes(name))name='home';
 if(area==='home'&&name!=='home'){resumeAnimation=playing;playing=false;}
 area=name;const auxiliary=['about','research','connect'].includes(name),home=name==='home',geometry=name==='geometry',molecular=name==='molecular',quantum=name==='quantum',transcription=name==='transcription';experience.dataset.area=name;
 experience.classList.toggle('atlas-open',!home);
 $('#atlas-panel').hidden=!geometry;$('#molecular-panel').hidden=!molecular;$('#quantum-panel').hidden=!quantum;$('#transcription-panel').hidden=!transcription;for(const id of ['about','research','connect'])$('#'+id+'-panel').hidden=id!==name;$('#record-navigation').hidden=molecular||quantum||transcription||auxiliary;$('#atlas-sections').hidden=home||geometry||molecular||quantum||transcription||auxiliary;
 experience.querySelector('main').hidden=!home;experience.querySelector('footer').hidden=!home;
 for(const id of ['genomics','sound','vision','language','partners'])$('#area-'+id).hidden=id!==name;
 if(geometry){recordMode='research';loadResearch();}
 if(home){recordId=M9;recordMode='animation';playing=resumeAnimation&&elapsed<total-1;last=null;render();}
 $('#atlas-toggle').textContent='Research Atlas';
 menu.querySelectorAll('[data-area]').forEach(button=>button.setAttribute('aria-current',button.dataset.area===name?'page':'false'));
 syncURL();announceRecord();
 closeMenu();document.dispatchEvent(new Event('atlas:close-records'));document.dispatchEvent(new CustomEvent('atlas:area',{detail:name}));
 if(focus){const heading=$('#area-'+name+' h1');if(heading){heading.tabIndex=-1;heading.focus({preventScroll:true});}else $('#atlas-toggle').focus();}
}
$('#atlas-toggle').setAttribute('aria-haspopup','menu');$('#atlas-toggle').setAttribute('aria-controls','atlas-menu');$('#atlas-toggle').onclick=()=>openMenu({focus:true});
document.addEventListener('atlas:open-records',()=>closeMenu());
document.addEventListener('atlas:close-navigation',()=>{closeMenu();document.dispatchEvent(new Event('atlas:close-records'));});
menuToggle.addEventListener('click',()=>menu.hidden?openMenu({focus:true}):closeMenu({focus:true}));

atlasNav.addEventListener('pointerleave',()=>{menuTimer=setTimeout(()=>{if(!menu.contains(document.activeElement))closeMenu();},160);});
atlasNav.addEventListener('focusout',()=>setTimeout(()=>{if(!atlasNav.contains(document.activeElement))closeMenu();},0));
for(const trigger of [$('#atlas-toggle'),menuToggle])trigger.addEventListener('keydown',e=>{if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();openMenu({focus:true,lastItem:e.key==='ArrowUp'});}});
menu.addEventListener('keydown',e=>{const items=[...menu.querySelectorAll('[role=menuitem]')],index=items.indexOf(document.activeElement);if(['ArrowDown','ArrowUp','Home','End'].includes(e.key)){e.preventDefault();const next=e.key==='Home'?0:e.key==='End'?items.length-1:(index+(e.key==='ArrowDown'?1:-1)+items.length)%items.length;items[next].focus();}});
document.addEventListener('pointerdown',e=>{if(!atlasNav.contains(e.target))closeMenu();});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!menu.hidden){e.preventDefault();closeMenu({focus:true});}if(e.key==='Escape')$('.atlas-brand-wrap').classList.add('tooltip-dismissed');});
$('.atlas-brand-wrap').addEventListener('pointerleave',()=>$('.atlas-brand-wrap').classList.remove('tooltip-dismissed'));
$('.atlas-brand-wrap').addEventListener('focusout',()=>$('.atlas-brand-wrap').classList.remove('tooltip-dismissed'));
document.addEventListener('click',e=>{const button=e.target.closest('button[data-area]');if(button){showArea(button.dataset.area,{focus:true});$('#atlas-sections').scrollTop=0;}if(e.target.closest('[data-about],[data-science-tfbs]'))closeMenu();});
document.addEventListener('science:open-atlas',event=>{
 const view=event.detail;if(!['alpha','beta','gamma','epsilon','geometry-summary'].includes(view))return;
 if(!entered)launch.click();chooseRecord(M9,'research',view);
});
// Research navigation retains the frame; first entry starts muted background video.
function navigateFromResearch(name){
 if(!areas.includes(name))return;
 document.querySelectorAll('dialog[open]').forEach(d=>d.close());
 const firstEntry=!entered;entered=true;document.body.dataset.entered='true';$('#landing').hidden=true;experience.hidden=false;
 if(firstEntry&&!film.src){film.muted=true;film.src=film.dataset.src;film.play().catch(()=>{});syncSound();}
 showArea(name,{focus:true});
}
document.addEventListener('atlas:navigate',event=>navigateFromResearch(event.detail));
function syncSound(){const label=film.muted?'Unmute video':'Mute video';$('#sound').textContent=film.muted?'♪̸':'♫';$('#sound').setAttribute('aria-label',label);$('#sound').title=label;}
$('#sound').onclick=()=>{if(film.paused){film.muted=false;film.play().catch(()=>{});}else film.muted=!film.muted;syncSound();};
function fullscreen(){if(!document.fullscreenElement && experience.requestFullscreen)experience.requestFullscreen().catch(()=>{});}
launch.disabled=false;
launch.onclick=()=>{entered=true;document.body.dataset.entered='true';$('#landing').hidden=true;experience.hidden=false;fullscreen();film.muted=false;film.volume=1;syncSound();if(!film.src)film.src=film.dataset.src;film.play().catch(()=>{$('#sound').textContent='▷';$('#sound').setAttribute('aria-label','Play video with sound');$('#sound').title='Play video with sound';});elapsed=reduced.matches?total-1:0;playing=!reduced.matches;last=null;render();resumeAnimation=!reduced.matches;const params=new URLSearchParams(location.search);showArea(params.get('area')||(recordMode==='research'?'geometry':'home'));$('#atlas-toggle').focus();};
$('#fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen().catch(()=>{});else fullscreen();};
document.addEventListener('fullscreenchange',()=>$('#fullscreen').setAttribute('aria-label',document.fullscreenElement?'Exit full screen':'Enter full screen'));
$('#back').onclick=()=>{closeMenu();document.dispatchEvent(new Event('atlas:close-records'));document.dispatchEvent(new CustomEvent('atlas:area',{detail:null}));film.pause();playing=false;entered=false;if(document.fullscreenElement)document.exitFullscreen().catch(()=>{});experience.hidden=true;$('#landing').hidden=false;document.body.dataset.entered='false';document.dispatchEvent(new Event('informatics:return'));launch.focus();};
window.addEventListener('pagehide',()=>{film.pause();playing=false;});
document.addEventListener('visibilitychange',()=>{last=null;if(document.hidden)film.pause();else if(entered)film.play().catch(()=>{});});
function frame(now){const dt=last===null?0:Math.min(now-last,60);last=now;if(entered&&playing&&!document.hidden){elapsed=Math.min(total-1,elapsed+dt);if(elapsed>=total-1)playing=false;render();}requestAnimationFrame(frame);}render();requestAnimationFrame(frame);
document.dispatchEvent(new Event('atlas:ready'));
if(initialParams.get('entry')==='research'&&[...areas,'transcription'].includes(initialParams.get('area'))){const enter=()=>navigateFromResearch(initialParams.get('area'));if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',enter,{once:true});else enter();}
})().catch(error=>{console.error(error);document.querySelector('#launch-status').textContent='Unable to load geometry. Please reload.';});

// About opens from the discreet static title.
(()=>{
const about=document.querySelector('#about');
const panel=document.querySelector('#about-panel');panel.append(about);about.setAttribute('open','');
document.querySelectorAll('[data-about]').forEach(button=>button.addEventListener('click',()=>document.dispatchEvent(new CustomEvent('atlas:navigate',{detail:'about'}))));
document.querySelector('#about-close').onclick=()=>document.dispatchEvent(new CustomEvent('atlas:navigate',{detail:'geometry'}));
const focusDialog=document.querySelector('#initial-focus'),focusButton=document.querySelector('#initial-focus-open');
focusButton.addEventListener('click',()=>{(document.fullscreenElement||document.body).append(focusDialog);focusDialog.showModal();});
focusDialog.querySelectorAll('[data-close-initial-focus]').forEach(button=>button.addEventListener('click',()=>focusDialog.close()));
focusDialog.addEventListener('click',e=>{if(e.target===focusDialog){const r=focusDialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)focusDialog.close();}});
focusDialog.addEventListener('close',()=>{if(!panel.hidden)focusButton.focus({preventScroll:true});});
if(new URLSearchParams(location.search).has('about'))document.addEventListener('atlas:ready',()=>document.dispatchEvent(new CustomEvent('atlas:navigate',{detail:'about'})),{once:true});
})();
