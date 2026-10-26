// A complete dataset is one gzip-compressed JSON or CSV file. Transfer chunks
// keep each published asset small; they are verified and joined before saving.
const root=new URL('./exports/',import.meta.url);
const buttons=[...document.querySelectorAll('[data-complete-download]')];
const status=document.querySelector('#complete-download-status');
const hex=bytes=>[...new Uint8Array(bytes)].map(v=>v.toString(16).padStart(2,'0')).join('');
let manifest;
async function load(){
 const r=await fetch(new URL('manifest.json',root),{cache:'no-store'});
 if(!r.ok)throw Error('The complete dataset index did not load. Please retry.');
 const m=await r.json();if(m.schema!=='universal-informatics/complete-dataset/v1')throw Error('Dataset index is invalid.');
 return m;
}
for(const button of buttons)button.addEventListener('click',async()=>{
 buttons.forEach(b=>b.disabled=true);
 try{
  manifest=await load();const format=button.dataset.completeDownload,item=manifest.downloads[format],parts=[];let received=0;
  for(const part of item.parts){
   if(!/^part-[a-f0-9]{64}\.bin$/.test(part.file))throw Error('Invalid download reference.');
   status.textContent=`Preparing the complete ${format.toUpperCase()} · ${Math.round(received/item.bytes*100)}%`;
   const response=await fetch(new URL(part.file,root));if(!response.ok)throw Error('The download was interrupted. Please retry.');
   const bytes=await response.arrayBuffer();
   if(bytes.byteLength!==part.bytes||hex(await crypto.subtle.digest('SHA-256',bytes))!==part.sha256)throw Error('The download failed its integrity check. Please retry.');
   parts.push(bytes);received+=bytes.byteLength;
  }
  const blob=new Blob(parts,{type:'application/gzip'}),url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download=item.file;link.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
  status.textContent=`Complete ${format.toUpperCase()} saved · ${manifest.sources.length.toLocaleString()} data files included. Decompress the .gz file to open one ${format.toUpperCase()} file. Version ${manifest.dataset_version.slice(0,12)}.`;
 }catch(error){status.textContent=error.message;}finally{buttons.forEach(b=>b.disabled=false);}
});
load().then(m=>{manifest=m;for(const b of buttons){const n=m.downloads[b.dataset.completeDownload];b.textContent=`Download all · ${b.dataset.completeDownload.toUpperCase()} · ${Math.ceil(n.bytes/1e6)} MB ↓`;}status.textContent=`${m.sources.length.toLocaleString()} data files · includes recorded CUDA-Q and Aquila experiments. One compressed file per format.`;}).catch(e=>{status.textContent=e.message;});
