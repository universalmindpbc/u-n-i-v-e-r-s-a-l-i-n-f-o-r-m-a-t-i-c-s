// Transport context is separate from the complete on-page transcript and sources.
const encoder=new TextEncoder();
export const byteLength=value=>encoder.encode(typeof value==='string'?value:JSON.stringify(value)).byteLength;
export function clipText(value,limit){
 if(typeof value!=='string')return '';
 const bytes=encoder.encode(value);return bytes.length<=limit?value:new TextDecoder().decode(bytes.subarray(0,limit),{stream:true});
}
export function compactHistory(value){
 const messages=(Array.isArray(value)?value:[]).filter(m=>['user','assistant'].includes(m?.role)&&typeof m.content==='string').slice(-12);
 const output=[];let remaining=18000;
 for(const m of messages.reverse()){
  const content=clipText(m.content,Math.min(remaining,m.role==='user'?2400:3600));
  const item={role:m.role,content};const size=byteLength(item);
  if(!content||size>remaining)break;
  output.unshift(item);remaining-=size;
 }
 return output;
}
export function compactResearchRequest(body={}){
 const raw=Array.isArray(body.history)?body.history:[],context=body.context||{},atlas=context.atlas||{},library=context.library||{};
 const references=[],seen=new Set();
 const candidates=[...raw].reverse().flatMap(m=>Array.isArray(m?.sources)?m.sources:[]).concat(Array.isArray(body.references)?body.references:[]);
 for(const source of candidates){
  if(typeof source?.id!=='string'||seen.has(source.id))continue;
  seen.add(source.id);references.push({id:clipText(source.id,180),title:clipText(source.title,220),kind:clipText(source.kind,30),section:clipText(source.section,100)});if(references.length===10)break;
 }
 const users=raw.filter(m=>m?.role==='user'&&typeof m.content==='string');
 const origin=clipText(body.conversation?.origin||users[0]?.content,900);
 const result={question:clipText(body.question,12000),history:compactHistory(raw),references,conversation:{origin},context:{page:clipText(context.page,160),area:clipText(context.area||atlas.area,80),atlas:{record:clipText(atlas.record,120),view:clipText(atlas.view,80)},library:{collection:clipText(library.collection,60),chapter:clipText(String(library.chapter??''),30),query:clipText(library.query,400),filters:Object.fromEntries(['community','age','condition'].map(key=>[key,(Array.isArray(library.filters?.[key])?library.filters[key]:[]).filter(v=>typeof v==='string').slice(0,12).map(v=>clipText(v,80))])),includeRelated:library.includeRelated===true}}};
 // Bound the encoded JSON, including Unicode and JSON escaping. Sources remain
 // in the on-page transcript; only identifiers cross the network again.
 while(byteLength(result)>44000){if(result.history.length)result.history.shift();else if(result.references.length)result.references.pop();else result.question=clipText(result.question,Math.floor(byteLength(result.question)/2));}
 return result;
}
export function retrievalQuery(question,history,origin,catalogue=[]){
 const followup=/\b(?:it|its|they|them|those|these|that|this|participants|sample|sessions|duration|limitations|younger|older|compare|comparison|longer|shorter|follow.up|anything else|more detail)\b/i.test(question);
 const newTopic=/\b(?:new question|different (?:topic|subject)|change (?:topic|subject))\b/i.test(question)||/^(?:what (?:is|are)|define|explain)\b/i.test(question)&&!followup;
 if(newTopic||!followup&&catalogue.some(t=>question.toLowerCase().includes(t)))return clipText(question,1600);
 const previous=(history||[]).filter(m=>m.role==='user'&&m.content.trim()!==question.trim());
 const topic=[...previous].reverse().find(m=>catalogue.some(t=>m.content.toLowerCase().includes(t)))?.content||origin;
 return clipText([question,topic&&topic!==question?'Conversation subject: '+clipText(topic,900):''].filter(Boolean).join('\n'),2000);
}
