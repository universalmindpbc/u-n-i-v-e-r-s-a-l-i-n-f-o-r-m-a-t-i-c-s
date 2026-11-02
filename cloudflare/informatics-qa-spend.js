// Optional staging QA ceiling; no access or quota privilege. Normal requests unchanged.
// Standard short-context prices verified 2 November 2026:
// https://developers.openai.com/api/docs/models/gpt-6-astra
// Use $12.50/M (cache-write rate) for every input token and $50/M output.
const knownInputs=new WeakMap(),pendingInputs=new WeakMap();
const bytes=value=>new TextEncoder().encode(JSON.stringify(value)).length;
export function qaSpendLedger(limit){
 const n=Number(limit);return Number.isFinite(n)&&n>0&&n<=0.95?{limitUSD:n,chargedUpperUSD:0,calls:[]}:null;
}
export function reserveQACall(ledger,path,body){
 if(!ledger)return null;
 if(!['/responses','/embeddings'].includes(path)||path==='/responses'&&body.model!=='gpt-6-astra')throw Object.assign(new Error('This model operation is outside the bounded QA test.'),{status:422,publicMessage:true});
 let inputUpper=bytes(body)+8192,boundMethod='UTF-8 bytes plus framing';
 const previous=knownInputs.get(ledger);
 // Reuse only provider-reported input usage for an unchanged, exact input prefix.
 // Every appended item and changed request field gets its full UTF-8 byte bound;
 // an additional 8192 tokens covers framing and token-boundary differences.
 if(path==='/responses'&&previous&&Array.isArray(body.input)&&body.input.length>=previous.body.input.length&&previous.body.input.every((v,i)=>JSON.stringify(v)===JSON.stringify(body.input[i]))){
  const changed=Object.fromEntries(Object.entries(body).filter(([k,v])=>k!=='input'&&JSON.stringify(v)!==JSON.stringify(previous.body[k])));
  inputUpper=Math.min(inputUpper,previous.tokens+bytes(body.input.slice(previous.body.input.length))+bytes(changed)+8192);boundMethod='Provider-counted exact prefix plus new UTF-8 bytes and framing';
 }
 if(inputUpper>=272000)throw Object.assign(new Error('QA request exceeds the verified short-context pricing range.'),{status:422,publicMessage:true});
 const outputUpper=path==='/responses'?body.max_output_tokens:0;
 if(!Number.isInteger(outputUpper)||outputUpper<0||outputUpper>1400)throw Object.assign(new Error('QA output tokens need an explicit bounded limit.'),{status:422,publicMessage:true});
 const reservedUSD=inputUpper*12.5/1e6+outputUpper*50/1e6;
 if(ledger.chargedUpperUSD+reservedUSD>ledger.limitUSD){ledger.blocked={path,inputTokenUpper:inputUpper,outputTokenUpper:outputUpper,reservedUSD,boundMethod};throw Object.assign(new Error('QA spending guard stopped before another paid request.'),{status:422,publicMessage:true});}
 const row={path,model:body.model,inputTokenUpper:inputUpper,outputTokenUpper:outputUpper,boundMethod,reservedUSD,chargedUpperUSD:reservedUSD,usage:null};
 if(path==='/responses'&&Array.isArray(body.input))pendingInputs.set(row,JSON.parse(JSON.stringify(body)));
 ledger.chargedUpperUSD+=reservedUSD;ledger.calls.push(row);return row;
}
export function reconcileQACall(ledger,row,usage){
 if(!ledger||!row||!usage)return;
 const input=usage.input_tokens??usage.prompt_tokens,output=usage.output_tokens??0;
 if(!Number.isFinite(input)||!Number.isFinite(output)||input<0||output<0)return;
 const chargedUpperUSD=input*12.5/1e6+output*50/1e6;
 ledger.chargedUpperUSD+=chargedUpperUSD-row.chargedUpperUSD;
 Object.assign(row,{usage,chargedUpperUSD});
 const body=pendingInputs.get(row);if(body)knownInputs.set(ledger,{body,tokens:input});
}
