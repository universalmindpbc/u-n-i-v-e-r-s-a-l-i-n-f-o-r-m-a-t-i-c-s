import assert from 'node:assert/strict';import fs from 'node:fs';import vm from 'node:vm';
let now=0,next=0;const callbacks=new Map();const video={volume:.8,muted:true,paused:true};const document={querySelectorAll:()=>[video],documentElement:{dataset:{}}};
const ctx={document,location:{origin:'https://staging.universalmind.tv'},performance:{now:()=>now},requestAnimationFrame:fn=>{callbacks.set(++next,fn);return next;},cancelAnimationFrame:id=>callbacks.delete(id)};ctx.window=ctx;ctx.parent=ctx;
vm.runInNewContext(fs.readFileSync(new URL('../../../Informatics/universal-one-audio.js',import.meta.url),'utf8'),ctx);
function step(ms){now+=ms;const pending=[...callbacks.values()];callbacks.clear();pending.forEach(fn=>fn(now));}
const a=ctx.UniversalOneAudio;
a.set('speech',true);step(800);assert(Math.abs(video.volume-.12)<1e-9);assert(video.muted&&video.paused);
a.set('speech',false);step(300);a.set('mic',true);step(800);assert(Math.abs(video.volume-.12)<1e-9);
a.set('second',true);a.set('mic',false);step(800);assert(Math.abs(video.volume-.12)<1e-9);
a.set('second',false);step(1000);assert.equal(video.volume,.8);assert.equal(a.getState().media,0);assert(video.muted&&video.paused);
console.log('15% mix, interrupted fades, overlapping owners and exact volume restoration passed.');
