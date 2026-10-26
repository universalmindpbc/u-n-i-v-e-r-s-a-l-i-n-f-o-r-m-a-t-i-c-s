// Node.js 20+. Verify an export, then optionally upload to a PRIVATE DeSci draft.
// node draft-desci.mjs --bundle EXPORT_DIRECTORY                 (verify only)
// DESCI_API_KEY_FILE=/private/path/key node draft-desci.mjs --list
// ... --bundle EXPORT_DIRECTORY --draft DRAFT_UUID --upload
// This adapter never publishes, signs a transaction or assigns a licence.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
const args=Object.fromEntries(process.argv.slice(2).filter(x=>x.startsWith('--')).map(x=>{const i=process.argv.indexOf(x);return [x.slice(2),process.argv[i+1]?.startsWith('--')?true:process.argv[i+1]??true];}));
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
let key;
async function request(route,body){
 key??=process.env.DESCI_API_KEY||(process.env.DESCI_API_KEY_FILE?await fs.readFile(process.env.DESCI_API_KEY_FILE,'utf8'):'');
 if(!key.trim())throw Error('Set DESCI_API_KEY_FILE or DESCI_API_KEY in your environment.');
 const r=await fetch('https://nodes-api.desci.com'+route,{method:body?'POST':'GET',headers:{'api-key':key.trim()},...(body?{body}:{}),signal:AbortSignal.timeout(120000),redirect:'error'});
 if(!r.ok){const error=Error('DeSci request failed: HTTP '+r.status+'. No credential or response body is printed.');error.status=r.status;throw error;}
 return r.json();
}
if(args.list){const data=await request('/v1/nodes/');console.log(JSON.stringify(data.nodes.map(n=>({id:n.uuid,title:n.title,published:n.isPublished})),null,2));process.exit(0);}
if(typeof args.bundle!=='string')throw Error('Provide --bundle EXPORT_DIRECTORY or --list.');
const root=await fs.realpath(args.bundle),manifestPath=path.join(root,'manifest.json');
const manifestBytes=await fs.readFile(manifestPath),manifest=JSON.parse(manifestBytes),integrity=JSON.parse(await fs.readFile(path.join(root,'integrity.json')));
if(hash(manifestBytes)!==integrity.sha256)throw Error('Manifest hash mismatch.');
const files=[],names=new Set();
for(const e of [...manifest.files,{path:'manifest.json'},{path:'integrity.json'}]){
 const full=path.resolve(root,e.path),relative=path.relative(root,full);
 if(!relative||relative.startsWith('..')||path.isAbsolute(relative))throw Error('Unsafe file path.');
 let walk=root;for(const segment of relative.split(path.sep)){walk=path.join(walk,segment);if((await fs.lstat(walk)).isSymbolicLink())throw Error('Symbolic links are not accepted.');}
 const name=relative.split(path.sep).join('__');if(names.has(name))throw Error('Upload names collide.');names.add(name);
 const bytes=await fs.readFile(full);if(e.sha256&&(hash(bytes)!==e.sha256||bytes.length!==e.bytes))throw Error('File verification failed: '+relative);
 files.push({name,full,bytes:bytes.length});
}
console.log(`Verified ${files.length} files for ${manifest.datasetVersion}.`);
if(!args.upload){console.log('Verification only. No upload performed.');process.exit(0);}
if(typeof args.draft!=='string')throw Error('--upload requires --draft DRAFT_UUID. Create a private draft in DeSci Publish first.');
const nodes=await request('/v1/nodes/'),draft=nodes.nodes.find(n=>n.uuid.replace(/\.$/,'')===args.draft.replace(/\.$/,''));
if(!draft||draft.isPublished!==false)throw Error('Choose an existing private draft owned by this account.');
// A raw IPFS CID carries the file SHA-256. This permits exact reconciliation
// when the provider saves a small file but responds with an HTTP error.
function rawCID(bytes){
 const input=Buffer.concat([Buffer.from([1,0x55,0x12,0x20]),crypto.createHash('sha256').update(bytes).digest()]);
 const alphabet='abcdefghijklmnopqrstuvwxyz234567';let bits=0,value=0,out='b';
 for(const b of input){value=(value<<8)|b;bits+=8;while(bits>=5){bits-=5;out+=alphabet[(value>>>bits)&31];}}
 if(bits)out+=alphabet[(value<<(5-bits))&31];return out;
}
async function savedFile(name,bytes){
 const result=await request('/v1/data/retrieveTree/'+encodeURIComponent(draft.uuid)+'/tree');
 const find=entries=>{for(const entry of entries||[]){if(entry.path==='root/'+name)return entry;const nested=find(entry.contains);if(nested)return nested;}};
 const entry=find(result.tree);
 return entry?.size===bytes.length&&entry.cid===rawCID(bytes)?entry:null;
}
const receiptPath=path.join(root,'desci-draft-receipt.json');
let receipt={provider:'DeSci Publish',draftId:draft.uuid,status:'uploading-private-draft',manifestSHA256:integrity.sha256,files:[]};
try{const old=JSON.parse(await fs.readFile(receiptPath));if(old.draftId!==draft.uuid||old.manifestSHA256!==integrity.sha256)throw Error('Receipt belongs to a different draft or export.');receipt=old;}catch(e){if(e.code!=='ENOENT')throw e;}
await fs.writeFile(receiptPath,JSON.stringify(receipt,null,2)+'\n',{mode:0o600});
for(const f of files){
 if(receipt.files.some(x=>x.name===f.name))continue;
 const bytes=await fs.readFile(f.full);let confirmed=await savedFile(f.name,bytes),response,warning;
 if(!confirmed){
  const form=new FormData();form.append('uuid',draft.uuid);form.append('contextPath','root');form.append('files',new Blob([bytes]),f.name);
  try{response=await request('/v1/data/update',form);}catch(error){
   confirmed=await savedFile(f.name,bytes);
   if(!confirmed)throw Error('Upload outcome is unresolved for '+f.name+'. The local receipt preserves completed files; inspect the private draft before retrying.');
   warning='Provider returned an error; saved file SHA-256 verified through its raw CID.';
  }
 }
 receipt.files.push({name:f.name,bytes:f.bytes,sha256:hash(bytes),manifestCid:response?.manifestCid??null,...(confirmed?{verifiedRawCID:confirmed.cid}:{}),...(warning?{warning}:{})});
 await fs.writeFile(receiptPath,JSON.stringify(receipt,null,2)+'\n',{mode:0o600});
}
receipt.status='uploaded-private-draft';receipt.completedAt=new Date().toISOString();
await fs.writeFile(receiptPath,JSON.stringify(receipt,null,2)+'\n',{mode:0o600});
console.log('Private draft upload complete. Review and publish in DeSci Publish. No public publication was performed.');
