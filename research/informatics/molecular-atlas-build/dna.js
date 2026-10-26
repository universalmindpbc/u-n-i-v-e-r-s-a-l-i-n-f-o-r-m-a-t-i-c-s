import {dnaTemplates,dnaModelSource} from './dna-templates.js';
export {dnaModelSource};
export const DNA_WINDOW=48;
export const complement={A:'T',T:'A',C:'G',G:'C',N:'N'};
export const baseColors={A:0xd9b27a,T:0x82d4cf,C:0x86abd4,G:0xc4a3d9,N:0x89949d};
export function genomicPosition(g,offset){return g.strand===1?g.start+offset:g.end-offset;}
export function dnaWindow(g,offset=0){
 const start=Math.max(0,Math.min(Math.trunc(offset),g.sequence.length-DNA_WINDOW));
 const sequence=g.sequence.slice(start,start+DNA_WINDOW),rise=dnaModelSource.riseAngstrom,twist=dnaModelSource.twistDegrees*Math.PI/180;
 return {kind:'gene',start,sequence,rise,twist,sphere:{center:[0,0,0],radius:Math.hypot(sequence.length*rise/2,12)+3},position:i=>genomicPosition(g,start+i)};
}
export function dnaModel(g,offset=0){
 const window=dnaWindow(g,offset),atoms=[],lines=[
  'HEADER    CONSTRUCTED IDEAL B-DNA MODEL',
  'TITLE     '+g.symbol+' REFERENCE SEQUENCE WINDOW',
  'REMARK 250 MODELLED COORDINATES; NOT AN EXPERIMENTAL GENE STRUCTURE.',
  'REMARK 250 NAB LANGRIDGE B-DNA TEMPLATES; HYDROGEN ATOMS OMITTED.',
  'REMARK 250 GRCH38 CHR'+g.chromosome+' '+window.position(0)+' TO '+window.position(47),
  'REMARK 250 CHAIN A: GENE 5-PRIME TO 3-PRIME; CHAIN B: REVERSE COMPLEMENT.'
 ];
 let serial=0;
 for(const chain of ['A','B']){
  for(let residue=1;residue<=window.sequence.length;residue++){
   const pair=chain==='A'?residue-1:window.sequence.length-residue,base=chain==='A'?window.sequence[pair]:complement[window.sequence[pair]],template=dnaTemplates[chain][base];
   if(!template)throw new Error('Atomic DNA view is unavailable for an unresolved reference base.');
   const angle=pair*window.twist,c=Math.cos(angle),s=Math.sin(angle);
   for(const [name,element,tx,ty,tz] of template){
    // Both strands start at a 5-prime hydroxyl; terminal hydrogens are implicit.
    if(residue===1&&['P','OP1','OP2'].includes(name))continue;
    const position=[tz+(pair-(window.sequence.length-1)/2)*window.rise,tx*c-ty*s,tx*s+ty*c];
    atoms.push({serial:++serial,chain,residue,pair,base,name,element,position});
    lines.push('ATOM  '+String(serial).padStart(5)+' '+(' '+name).padEnd(4)+' '+('D'+base).padStart(3)+' '+chain+String(residue).padStart(4)+'    '+position.map(x=>x.toFixed(3).padStart(8)).join('')+'  1.00  0.00          '+element.padStart(2)+'  ');
   }
  }
  lines.push('TER');
 }
 lines.push('END');
 return {window,atoms,pdb:lines.join('\n')+'\n',source:dnaModelSource};
}
