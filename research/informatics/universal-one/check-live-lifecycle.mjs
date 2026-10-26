import fs from 'node:fs';import vm from 'node:vm';import assert from 'node:assert/strict';
import {needsResearch} from '../../../Informatics/universal-one-routing.js';
const catalogue=JSON.parse(fs.readFileSync(new URL('../../../cloudflare/informatics-research-topics.json',import.meta.url),'utf8')).terms;
const code=fs.readFileSync(new URL('../../../Informatics/universal-one-live.js',import.meta.url),'utf8');
const pause=ms=>new Promise(r=>setTimeout(r,ms));
function harness(){
 const sent=[],states=[],requests=[];let channel,peer,resolveResearch,resolveMic;
 const track={enabled:true,stop(){this.stopped=true;}};const stream={getTracks:()=>[track],getAudioTracks:()=>[track]};
 class Channel extends EventTarget{readyState='open';send(v){sent.push(JSON.parse(v));}close(){this.readyState='closed';}}
 class Peer{connectionState='connected';constructor(){peer=this;}addTrack(){}createDataChannel(){channel=new Channel();return channel;}async createOffer(){return {sdp:'test-offer'};}async setLocalDescription(){}async setRemoteDescription(){channel.dispatchEvent(new Event('open'));}close(){this.connectionState='closed';}}
 const sandbox={window:{RTCPeerConnection:Peer,UniversalOneNeedsResearch:needsResearch},RTCPeerConnection:Peer,navigator:{mediaDevices:{getUserMedia:async()=>stream}},Audio:class{play(){return Promise.resolve();}pause(){}},AbortController,setTimeout,clearTimeout,fetch:async()=>({ok:true,text:async()=>'test-answer'})};
 vm.runInNewContext(code,sandbox);
 const live=sandbox.window.UniversalOneLive({request:async(path,body,signal)=>{requests.push({path,signal,body});if(path==='/research')return new Promise(r=>resolveResearch=r);return {json:async()=>({value:'test-ephemeral',routingCatalogue:catalogue})};},history:()=>[],context:()=>({}),onState:(s,label)=>states.push({s,label}),onMicrophone:()=>{},onUser:()=>{},onAssistant:()=>{}});
 return {live,track,sent,states,requests,emit:e=>channel.onmessage({data:JSON.stringify(e)}),researchResult:()=>resolveResearch({json:async()=>({answer:'Late result',sources:[]})}),delayMic:()=>{sandbox.navigator.mediaDevices.getUserMedia=()=>new Promise(r=>resolveMic=r);},resolveMic:()=>resolveMic(stream)};
}
const h=harness();
try{
 await h.live.start();assert(h.live.getState().active);assert(!h.track.enabled);
 h.emit({type:'response.created'});h.emit({type:'output_audio_buffer.stopped'});await pause(950);
 assert(!h.track.enabled,'An earlier audio-stop event never opens the mic during active response generation');
 h.emit({type:'response.done',response:{status:'completed',output:[]}});await pause(950);assert(h.track.enabled);
 h.emit({type:'response.created'});h.emit({type:'output_audio_buffer.started'});h.live.toggleMute();assert.equal(h.states.at(-1).s,'speaking');assert(!h.track.enabled);
 h.emit({type:'output_audio_buffer.stopped'});h.emit({type:'response.done',response:{status:'completed',output:[]}});await pause(950);assert(!h.track.enabled);h.live.toggleMute();assert(h.track.enabled);
 h.emit({type:'response.created'});const call={type:'function_call',name:'ask_research',call_id:'test-call',arguments:'{"question":"Test the cited research"}'};
 h.emit({type:'response.done',response:{status:'completed',output:[call]}});assert(h.live.getState().pending);assert(!h.track.enabled);
 h.live.interrupt();assert(h.requests.at(-1).signal.aborted);const count=h.sent.length;h.researchResult();await pause(0);assert.equal(h.sent.length,count,'Interrupted retrieval never sends a late answer');
 h.live.end();assert(h.track.stopped);assert(!h.live.getState().microphone);
}finally{h.live.end();}
const late=harness();late.delayMic();const starting=late.live.start();await pause(0);late.live.end();late.resolveMic();await starting;assert(late.track.stopped);assert.equal(late.requests.length,0);assert(!late.live.getState().active);
console.log('Live voice gates, mute, interrupted retrieval and late microphone cancellation passed.');
const hidden=harness();
try{
 await hidden.live.start();hidden.emit({type:'response.done',response:{status:'completed',output:[]}});await pause(950);
 assert(hidden.track.enabled);const requestCount=hidden.requests.length;
 hidden.live.setVisible(false);assert(!hidden.track.enabled);assert(hidden.live.getState().active);assert(hidden.live.getState().connected);assert.equal(hidden.states.at(-1).s,'paused');
 hidden.live.setVisible(true);assert(hidden.track.enabled);assert.equal(hidden.requests.length,requestCount);assert(!hidden.track.stopped);
 hidden.live.toggleMute();hidden.live.setVisible(false);hidden.live.setVisible(true);assert(!hidden.track.enabled,'Returning to the page preserves an explicit microphone mute');
}finally{hidden.live.end();}
const permission=harness();permission.delayMic();const permissionStart=permission.live.start();await pause(0);permission.live.setVisible(false);permission.resolveMic();await pause(0);
assert.equal(permission.requests.length,0,'Permission received in a hidden page waits before connecting');assert(!permission.track.enabled);
permission.live.setVisible(true);await permissionStart;assert(permission.live.getState().active);assert.equal(permission.requests.length,1);permission.live.end();
const cancelled=harness();cancelled.delayMic();const cancelledStart=cancelled.live.start();await pause(0);cancelled.live.setVisible(false);cancelled.resolveMic();await pause(0);cancelled.live.end();await cancelledStart;cancelled.live.setVisible(true);assert(cancelled.track.stopped);assert.equal(cancelled.requests.length,0);
console.log('Transient visibility, permission handoff, explicit mute and cancelled hidden startup passed.');

const routed=harness();
try{
 await routed.live.start();routed.emit({type:'response.done',response:{status:'completed',output:[]}});await pause(950);
 const count=routed.sent.length;
 routed.emit({type:'input_audio_buffer.speech_stopped',item_id:'music'});assert(!routed.track.enabled);assert.equal(routed.sent.length,count,'No answer before transcription routing');
 routed.emit({type:'conversation.item.input_audio_transcription.completed',item_id:'music',transcript:'Electronic music making and gene expression.'});
 assert.equal(routed.sent.at(-1).response.tool_choice.name,'ask_research');
 const after=routed.sent.length;routed.emit({type:'conversation.item.input_audio_transcription.completed',item_id:'music',transcript:'Electronic music making and gene expression.'});assert.equal(routed.sent.length,after,'Duplicate transcription never duplicates a response');
 routed.emit({type:'response.done',response:{status:'completed',output:[]}});await pause(950);
 routed.emit({type:'input_audio_buffer.speech_stopped',item_id:'cancelled'});routed.live.interrupt();
 const cancelledCount=routed.sent.length;routed.emit({type:'conversation.item.input_audio_transcription.completed',item_id:'cancelled',transcript:'Tell me about EMDR'});assert.equal(routed.sent.length,cancelledCount,'Interrupted transcription never sends a late response');
 routed.emit({type:'conversation.item.input_audio_transcription.completed',item_id:'general',transcript:'What is an electron?'});assert.equal(routed.sent.at(-1).response.tool_choice.name,'ask_research');
 routed.emit({type:'conversation.item.input_audio_transcription.completed',item_id:'followup',transcript:'How long were the sessions?'});assert.equal(routed.sent.at(-1).response.tool_choice.name,'ask_research');
 routed.emit({type:'conversation.item.input_audio_transcription.completed',item_id:'thanks',transcript:'Thank you.'});assert.equal(routed.sent.at(-1).response.tool_choice,'auto');
}finally{routed.live.end();}
console.log('Speech is routed before generation; default retrieval, unnamed follow-ups, social bypass, duplicate events and interrupted transcription passed.');
