// I Am Realtime transport, Marin voice and input gates, scoped to Informatics.
(() => {
 window.UniversalOneLive=({request,history,context,onState,onMicrophone,onUser,onAssistant})=>{
  let peer=null,channel=null,stream=null,remote=null,abort=null,active=false,muted=false,gated=true,playing=false,responseActive=false,pending=null,lastResult=null,epoch=0,gateTimer=0,sessionTimer=0,disconnectTimer=0,suspended=false,currentState='idle',currentLabel='Ready when you are.',visibilityWaiters=new Set(),routingCatalogue=[],transcriptionTimer=0,pendingSpeech=null;
  const completed=new Set(),spoken=new Set(),heard=new Set();
  const send=value=>{if(channel?.readyState==='open'){channel.send(JSON.stringify(value));return true;}return false;};
  const sync=()=>stream?.getAudioTracks().forEach(t=>{t.enabled=active&&!muted&&!gated&&!suspended;});
  const state=(name,label)=>{currentState=name;currentLabel=label;onState(suspended&&name!=='error'?'paused':name,suspended&&name!=='error'?'Voice paused':label);};
  function setVisible(visible){suspended=!visible;sync();if(remote)remote.muted=suspended;if(visible){for(const wake of [...visibilityWaiters])wake();}if(active||currentState==='connecting')state(currentState,currentLabel);}
  function waitForVisible(signal){if(!suspended)return Promise.resolve();return new Promise((resolve,reject)=>{const cleanup=()=>{visibilityWaiters.delete(wake);signal.removeEventListener('abort',aborted);};const wake=()=>{cleanup();resolve();};const aborted=()=>{cleanup();reject(Object.assign(Error('Voice start cancelled.'),{name:'AbortError'}));};visibilityWaiters.add(wake);signal.addEventListener('abort',aborted,{once:true});if(signal.aborted)aborted();});}
  function gate(){clearTimeout(gateTimer);gated=true;sync();}
  function listen(){clearTimeout(gateTimer);gateTimer=setTimeout(()=>{if(!active||pending||playing||responseActive)return;gated=false;sync();state(muted?'muted':'listening',muted?'Microphone off':'Listening…');},900);}
  function end(){epoch++;active=false;clearTimeout(gateTimer);clearTimeout(sessionTimer);clearTimeout(disconnectTimer);clearTimeout(transcriptionTimer);abort?.abort();abort=null;pending=null;channel?.close();channel=null;if(peer){peer.onconnectionstatechange=null;peer.close();peer=null;}stream?.getTracks().forEach(t=>t.stop());stream=null;onMicrophone(null);if(remote){remote.pause();remote.srcObject=null;remote=null;}playing=false;responseActive=false;gated=true;muted=false;completed.clear();spoken.clear();heard.clear();lastResult=null;currentState='idle';currentLabel='Ready when you are.';pendingSpeech=null;}
  function fail(label){end();state('error',label);}
  async function research(call,token){
   if(completed.has(call.call_id))return;completed.add(call.call_id);pending=call.call_id;gate();state('thinking','Thinking…');abort=new AbortController();
   try{
    if(call.name!=='ask_research')throw Error('Unknown research operation.');
    const args=JSON.parse(call.arguments);if(typeof args.question!=='string'||!args.question.trim())throw Error('Please repeat the research question.');
    const result=await(await request('/research',{question:args.question,history:history(),context:context()},abort.signal)).json();
    if(token!==epoch||pending!==call.call_id)return;
    lastResult=result;pending=null;
    send({type:'conversation.item.create',item:{type:'function_call_output',call_id:call.call_id,output:JSON.stringify({answer:result.answer,sources:result.sources.map(s=>({label:s.label,title:s.title,url:s.url})),version:result.version})}});
    send({type:'response.create',response:{tool_choice:'none',instructions:'Answer the current question using the research tool result just returned. Preserve specific study findings, names and relevant numbers. Use general knowledge only to explain them. If no relevant evidence was found, state that distinction. Never invent a project claim. Keep the spoken answer concise; source links are in the transcript.'}});
   }catch(error){if(token!==epoch||pending!==call.call_id)return;pending=null;send({type:'conversation.item.create',item:{type:'function_call_output',call_id:call.call_id,output:JSON.stringify({error:error.message})}});send({type:'response.create'});state('thinking','Research request interrupted. Preparing a response…');}
  }
  function receive(message,token){
   if(token!==epoch)return;let e;try{e=JSON.parse(message.data);}catch{return;}
   if(['conversation.item.created','conversation.item.added'].includes(e.type)&&e.item?.role==='user')lastResult=null;
   if(e.type==='input_audio_buffer.speech_started'&&!gated){lastResult=null;state('listening','Listening…');}
   if(e.type==='input_audio_buffer.speech_stopped'&&!gated){
    gate();state('thinking','Thinking…');pendingSpeech=e.item_id;clearTimeout(transcriptionTimer);
    transcriptionTimer=setTimeout(()=>{heard.add(e.item_id);pendingSpeech=null;listen();},15000);
   }
   if(e.type==='conversation.item.input_audio_transcription.failed'){clearTimeout(transcriptionTimer);pendingSpeech=null;heard.add(e.item_id);listen();}
   if(e.type==='conversation.item.input_audio_transcription.completed'&&!heard.has(e.item_id)){
    heard.add(e.item_id);pendingSpeech=null;clearTimeout(transcriptionTimer);
    const question=e.transcript?.trim();if(!question){listen();return;}
    onUser(question);gate();state('thinking','Thinking…');
    // Research every substantive turn before speech; social acknowledgements
    // and voice controls are the only direct-conversation bypass.
    const required=window.UniversalOneNeedsResearch?.(question,routingCatalogue,history())??true;
    send({type:'response.create',response:{tool_choice:required?{type:'function',name:'ask_research'}:'auto'}});
   }
   if(e.type==='response.created'){responseActive=true;gate();state('thinking','Thinking…');}
   if(e.type==='output_audio_buffer.started'){playing=true;gate();state('speaking','Speaking');}
   if(e.type==='output_audio_buffer.stopped'||e.type==='output_audio_buffer.cleared'){playing=false;if(!pending)listen();}
   if(['response.output_audio_transcript.done','response.audio_transcript.done'].includes(e.type)&&e.transcript?.trim()){
    const id=e.item_id||e.response_id;if(!spoken.has(id)){spoken.add(id);onAssistant(e.transcript.trim(),lastResult?.sources||[]);}
   }
   if(e.type==='response.done'){
    responseActive=false;
    const calls=(e.response?.output||[]).filter(item=>item.type==='function_call');
    if(calls.length){void research(calls[0],token);}
    else if(e.response?.status==='failed')fail('Voice response failed. Select Reconnect to try again.');
    else if(!playing&&!pending)listen();
   }
   if(e.type==='error'&&!['response_cancel_not_active','input_audio_buffer_commit_empty'].includes(e.error?.code))fail('Voice connection interrupted. Select Reconnect to try again.');
  }
  async function start(){
   end();const token=epoch;state('connecting','Connecting voice…');abort=new AbortController();
   try{
    if(!navigator.mediaDevices?.getUserMedia||!window.RTCPeerConnection)throw Error('Live voice is unavailable in this browser. Text and dictation remain available.');
    await waitForVisible(abort.signal);if(token!==epoch)return;
    const mic=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true,channelCount:1}});
    if(token!==epoch){mic.getTracks().forEach(t=>t.stop());return;}
    stream=mic;stream.getAudioTracks().forEach(t=>{t.enabled=false;try{t.contentHint='speech';}catch{}});onMicrophone(stream);
    await waitForVisible(abort.signal);if(token!==epoch)return;
    const payload=await(await request('/voice-session',{manualTurns:true},abort.signal)).json();if(token!==epoch)return;routingCatalogue=payload.routingCatalogue||[];
    const connection=new RTCPeerConnection();peer=connection;remote=new Audio();remote.autoplay=true;remote.muted=suspended;
    connection.ontrack=e=>{if(token!==epoch)return;remote.srcObject=e.streams[0];remote.play().catch(()=>{if(token===epoch)fail('Voice playback was blocked. Select Reconnect to try again.');});};
    connection.onconnectionstatechange=()=>{if(token!==epoch)return;clearTimeout(disconnectTimer);if(['failed','closed'].includes(connection.connectionState))fail('Voice connection ended. Select Reconnect to try again.');else if(connection.connectionState==='disconnected')disconnectTimer=setTimeout(()=>{if(token===epoch&&connection.connectionState==='disconnected')fail('Voice connection interrupted. Select Reconnect to try again.');},5000);};
    stream.getTracks().forEach(t=>connection.addTrack(t,stream));channel=connection.createDataChannel('oai-events');channel.onmessage=e=>receive(e,token);channel.addEventListener('close',()=>{if(token===epoch&&active)fail('Voice connection ended. Select Reconnect to try again.');});channel.addEventListener('error',()=>{if(token===epoch&&active)fail('Voice connection interrupted. Select Reconnect to try again.');});
    const ready=new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('The live voice connection timed out.')),15000);channel.addEventListener('open',()=>{clearTimeout(timer);resolve();},{once:true});channel.addEventListener('error',()=>{clearTimeout(timer);reject(Error('The live voice connection failed.'));},{once:true});});ready.catch(()=>{});
    const offer=await connection.createOffer();await connection.setLocalDescription(offer);
    const answer=await fetch('https://api.openai.com/v1/realtime/calls',{method:'POST',body:offer.sdp,headers:{Authorization:`Bearer ${payload.value}`,'Content-Type':'application/sdp'},signal:abort.signal});
    if(!answer.ok)throw Error('OpenAI did not complete the live voice connection.');
    const sdp=await answer.text();if(token!==epoch)return;await connection.setRemoteDescription({type:'answer',sdp});await ready;if(token!==epoch)return;
    await waitForVisible(abort.signal);if(token!==epoch)return;
    active=true;sessionTimer=setTimeout(()=>fail('Voice session complete. Select Reconnect to continue.'),10*60*1000);
    for(const item of (window.UniversalOneContext?.compactHistory(history())||[]))send({type:'conversation.item.create',item:{type:'message',role:item.role,content:[{type:item.role==='assistant'?'output_text':'input_text',text:item.content.slice(0,5000)}]}});
    send({type:'response.create',response:{instructions:'Say exactly: Hi. What research question can I help you with? Then wait for the visitor.'}});
   }catch(error){if(token!==epoch)return;fail(error.name==='NotAllowedError'?'Microphone access was denied. Enable it in your browser, or select Text.':error.message);}
  }
  function toggleMute(){if(!active)return;muted=!muted;sync();state(playing?'speaking':gated?'thinking':muted?'muted':'listening',playing?'Speaking':gated?'Thinking…':muted?'Microphone off':'Listening…');}
  function interrupt(){if(!active)return;clearTimeout(transcriptionTimer);if(pendingSpeech){heard.add(pendingSpeech);pendingSpeech=null;}if(responseActive)send({type:'response.cancel'});send({type:'output_audio_buffer.clear'});if(pending){const call=pending;pending=null;abort?.abort();send({type:'conversation.item.create',item:{type:'function_call_output',call_id:call,output:JSON.stringify({error:'The visitor interrupted this request.'})}});}playing=false;lastResult=null;listen();}
  return {start,end,setVisible,toggleMute,interrupt,getState:()=>({active,muted,gated,suspended,playing,pending:!!pending,microphone:!!stream,inputEnabled:!!stream?.getAudioTracks().some(t=>t.enabled),connected:peer?.connectionState==='connected'})};
 };
})();
