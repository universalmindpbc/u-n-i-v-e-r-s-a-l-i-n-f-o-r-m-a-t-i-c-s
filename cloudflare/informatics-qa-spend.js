// Optional spending guard for staging verification. It only lowers request limits;
// it grants no access or quota privilege. Requests without the header are unchanged.
// Standard short-context prices verified 2 November 2026:
// https://developers.openai.com/api/docs/models/gpt-6-astra ($10/M input, $12.50/M cache writes, $50/M output).
// UTF-8 bytes plus framing allowance conservatively bound text input tokens.
export function qaSpendLedger(limit){
 const n=Number(limit);return Number.isFinite(n)&&n>0&&n<=0.95?{limitUSD:n,chargedUpperUSD:0,calls:[]}:null;
}
export function reserveQACall(ledger,path,body){
 if(!ledger)return null;
 if(!['/responses','/embeddings'].includes(path)||path==='/responses'&&body.model!=='gpt-6-astra')throw Object.assign(new Error('This model operation is outside the bounded QA test.'),{status:422,publicMessage:true});
 const inputUpper=new TextEncoder().encode(JSON.stringify(body)).length+8192;
 if(inputUpper>=272000)throw Object.assign(new Error('QA request exceeds the verified short-context pricing range.'),{status:422,publicMessage:true});
 const outputUpper=path==='/responses'?body.max_output_tokens:0;
 if(!Number.isInteger(outputUpper)||outputUpper<0||outputUpper>1400)throw Object.assign(new Error('QA output tokens need an explicit bounded limit.'),{status:422,publicMessage:true});
 // Use the highest Astra input/cache-write rate even for embeddings as a conservative bound.
 const reservedUSD=inputUpper*12.5/1e6+outputUpper*50/1e6;
 if(ledger.chargedUpperUSD+reservedUSD>ledger.limitUSD)throw Object.assign(new Error('QA spending guard stopped before another paid request.'),{status:422,publicMessage:true});
 const row={path,model:body.model,inputTokenUpper:inputUpper,outputTokenUpper:outputUpper,reservedUSD,chargedUpperUSD:reservedUSD,usage:null};
 ledger.chargedUpperUSD+=reservedUSD;ledger.calls.push(row);return row;
}
export function reconcileQACall(ledger,row,usage){
 if(!ledger||!row||!usage)return;
 const input=usage.input_tokens??usage.prompt_tokens,output=usage.output_tokens??0;
 if(!Number.isFinite(input)||!Number.isFinite(output))return;
 // Use the higher cache-write rate for all input; ignore caching discounts.
 const chargedUpperUSD=input*12.5/1e6+output*50/1e6;
 ledger.chargedUpperUSD+=chargedUpperUSD-row.chargedUpperUSD;
 Object.assign(row,{usage,chargedUpperUSD});
}
