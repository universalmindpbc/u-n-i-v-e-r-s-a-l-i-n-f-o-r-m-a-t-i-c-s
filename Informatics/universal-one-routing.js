// Every substantive turn is researched. Exact social/voice-control phrases are
// the only bypass; an unfamiliar term or unnamed follow-up is never a bypass.
export const normaliseResearchText=value=>String(value||'').normalize('NFKC').toLowerCase().replace(/(\d)(hz|hertz)\b/gi,'$1 $2').replace(/[–—−-]/g,' ').replace(/(?<!\d)\.|\.(?!\d)/g,' ').replace(/[^\p{L}\p{N}.+]+/gu,' ').trim().replace(/\s+/g,' ');
export function needsResearch(question){
 const text=normaliseResearchText(question);
 return !!text&&!/^(?:(?:hi|hello|hey|namaste)(?: there)?|(?:thank you|thanks)(?: very much| so much)?|good (?:morning|afternoon|evening|night)|goodbye|bye|stop|pause|stop speaking|speak (?:slower|faster)|repeat (?:that|the last answer)|say that again)$/.test(text);
}
