// I Am contract: 15% programme mix, 800 ms down, 1000 ms restore.
// Retain the actual previous level and mute state; never start programme media.
(() => {
 const host=window.parent!==window&&window.parent.location.origin===location.origin?window.parent:window;
 if(host.UniversalOneAudio){window.UniversalOneAudio=host.UniversalOneAudio;return;}
 const baselines=new Map(),frames=new Map(),owners=new Set();
 function fade(media,target,duration,complete){
  cancelAnimationFrame(frames.get(media));const start=media.volume,time=performance.now();
  const tick=now=>{const p=Math.min(1,Math.max(0,(now-time)/duration));media.volume=Math.max(0,Math.min(1,start+(target-start)*(1-Math.pow(1-p,3))));if(p<1)frames.set(media,requestAnimationFrame(tick));else {frames.delete(media);complete?.();}};
  frames.set(media,requestAnimationFrame(tick));
 }
 const service={set(owner,active){
  if(active)owners.add(owner);else owners.delete(owner);
  if(owners.size){
   const media=[...host.document.querySelectorAll('video'),...document.querySelectorAll('video')];
   for(const video of new Set(media)){if(!baselines.has(video))baselines.set(video,video.volume);fade(video,baselines.get(video)*.15,800);}
  }else{for(const [media,volume]of baselines)fade(media,volume,1000,()=>{if(!owners.size)baselines.delete(media);});}
  host.document.documentElement.dataset.universalOneDucking=String(!!owners.size);
 },getState(){return {active:!!owners.size,media:baselines.size,mix:.15,fadeDownMs:800,fadeUpMs:1000};}};
 host.UniversalOneAudio=service;window.UniversalOneAudio=service;
})();
