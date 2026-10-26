'use strict';
(()=>{
const $=selector=>document.querySelector(selector), NS='http://www.w3.org/2000/svg';
const C={alpha:'#5ca8ff',beta:'#ffea42',gamma:'#ff84c5',hex:'#62d998',white:'#e5edf2',muted:'#859aa8',guide:'#30424e'};
const clamp=value=>Math.max(0,Math.min(1,value));
const node=(tag,attrs={},parent)=>{const element=document.createElementNS(NS,tag);for(const [key,value]of Object.entries(attrs))element.setAttribute(key,value);parent.append(element);return element;};
const label=(parent,value,x,y,attrs={})=>{const element=node('text',{x,y,fill:C.white,'font-size':15,'text-anchor':'middle',...attrs},parent);element.textContent=value;element.style.textAnchor=attrs['text-anchor']||'middle';return element;};
const line=(parent,points,color=C.white,attrs={})=>node('path',{d:points.map((p,i)=>`${i?'L':'M'}${p.join(',')}`).join(' '),fill:'none',stroke:color,'stroke-width':1.5,'stroke-linecap':'round','stroke-linejoin':'round',...attrs},parent);
const circle=(parent,x,y,r,color=C.guide,attrs={})=>node('circle',{cx:x,cy:y,r,fill:'none',stroke:color,'stroke-width':1,...attrs},parent);
const polar=(cx,cy,r,index,total)=>[cx+r*Math.sin(index*2*Math.PI/total),cy-r*Math.cos(index*2*Math.PI/total)];
const draw=(element,progress)=>{element.setAttribute('pathLength','1');element.setAttribute('stroke-dasharray','1');element.setAttribute('stroke-dashoffset',String(1-clamp(progress)));};
function reset(svg,title,description){svg.setAttribute('viewBox','0 0 720 560');svg.replaceChildren();const prefix=svg.id.startsWith('sound')?'sound':'vision';node('title',{id:prefix+'-svg-title'},svg).textContent=title;node('desc',{id:prefix+'-svg-desc'},svg).textContent=description;}
const round=value=>Number(value.toFixed(2));
const alpha=[9,1,1,2,3,5,8,4,3,7,1,8,9,8,8,7,6,4,1,5,6,2,8,1],gammaIndices=[0,4,8,12,16,20],melodyDegrees=[1,3,5,8];
const frequencyMapping=hz=>{const digits=String(hz).match(/\d/g).map(Number),sum=digits.reduce((a,b)=>a+b,0),residue=sum%9||9;return{hz,digits,sum,residue,shape:residue===3?'triangle':residue===6?'hexagon':residue===9?'circle':'residue'};};
const phraseBars=[4,8,16,32,64,128], residues=phraseBars.map(n=>n%9);
const semitones=[0,2,4,5,7,9,11,12], names=['C','D','E','F','G','A','B','C'];
const chromaticNames=['C','C♯','D','D♯','E','F','F♯','G','G♯','A','A♯','B'],chromaticBlack=[1,3,6,8,10];
const frequencies=reference=>semitones.map(n=>reference*Math.pow(2,(n-9)/12));
const fifths=['C','G','D','A','E','B','F♯ / G♭','D♭','A♭','E♭','B♭','F'];
const scales={C:['C','D','E','F','G','A','B'],G:['G','A','B','C','D','E','F♯'],D:['D','E','F♯','G','A','B','C♯'],A:['A','B','C♯','D','E','F♯','G♯'],E:['E','F♯','G♯','A','B','C♯','D♯'],B:['B','C♯','D♯','E','F♯','G♯','A♯'],F:['F','G','A','B♭','C','D','E']};
// The same calculations drive the labels, workings and verification.
window.InformaticsSoundMath=Object.freeze({phraseBars,residues,semitones,frequencies,fifths,scales,alpha,gammaIndices,melodyDegrees,frequencyMapping,chromaticNames,chromaticBlack});
let activeArea=null,soundView='sequence',reference=440,key='C';
const blends={sequence:new Set(),melody:new Set(),scale:new Set()};
const descriptions={
 sequence:{label:'SEQUENCE STRUCTURE',title:'Doubling the phrase',description:'A four-bar phrase doubles to 8, 16, 32, 64 and 128 bars. Reducing each length modulo 9 gives a repeating six-value cycle.',equation:'4 → 8 → 7 → 5 → 1 → 2',caption:'Phrase length in bars, mapped onto the nine positions of the Beta circle.',steps:6,accent:C.beta,atlas:'beta',link:'Explore Beta in the Research Atlas →',workings:'<p>In 4/4, each bar contains four quarter-note beats. Here the numbers count bars in a phrase. Doubling is the structural choice being explored.</p><table><thead><tr><th>Bars</th><th>Calculation</th></tr></thead><tbody>'+phraseBars.map((n,i)=>`<tr><td>${n}</td><td>${n} = 9 × ${Math.floor(n/9)} + ${residues[i]}</td></tr>`).join('')+'</tbody></table><p>The M9 / P24 Beta cycle is <code>1, 2, 4, 8, 7, 5</code>. Starting at 4 rotates the same cycle to <code>4, 8, 7, 5, 1, 2</code>.</p><p>Within the Atlas, Beta is calculated by summing the Alpha values strictly between successive Gamma positions and reducing modulo 9. These two calculations produce the same six-value cycle.</p>'},
 melody:{label:'MELODIC STRUCTURE',title:'One · three · five · eight',description:'Select the first, third and fifth degrees of C major, then repeat the tonic at the octave: C, E, G, C. These degree numbers are selected Fibonacci numbers.',equation:'1 · 3 · 5 · 8\nC · E · G · C',caption:'The piano and circle highlight the same values: 1, 3, 5 and 8. Blend Alpha to reveal the full M9 / P24 sequence.',steps:4,accent:C.alpha,atlas:'alpha',link:'Explore Alpha in the Research Atlas →',workings:'<p>The Fibonacci sequence begins <code>0, 1, 1, 2, 3, 5, 8…</code>. The chord selects degrees <code>1, 3, 5, 8</code>; these are selected values within the sequence.</p><p>The circle keeps all 24 Alpha positions in their original order. Every occurrence of <code>1, 3, 5 or 8</code> is highlighted, synchronised with its piano key. Repeated values stay at their separate positions. Blend Alpha to reveal the remaining values; this is a comparison of scale-degree numbers with sequence values.</p><p>A major triad is degrees 1, 3 and 5. Degree 8 repeats its root one octave higher.</p><table><thead><tr><th>Degree</th><th>Just ratio</th><th>12-TET ratio</th></tr></thead><tbody><tr><td>1</td><td>1</td><td>1</td></tr><tr><td>3</td><td>5/4</td><td>1.25992</td></tr><tr><td>5</td><td>3/2</td><td>1.49831</td></tr><tr><td>8</td><td>2</td><td>2</td></tr></tbody></table><p>12-TET means twelve equal divisions of the octave. Its frequency ratio after <code>n</code> semitones is <code>2^(n/12)</code>. Just intonation gives the chord the proportional frequencies <code>4 : 5 : 6 : 8</code>.</p>'},
 scale:{label:'SCALE',title:'A tonic and six companions',description:'Place the tonic at the centre and the six other notes of its major scale around it. Trace the hexagon, then return to the centre.',equation:'1 tonic + 6 notes = 7\n2 · 2 · 1 · 2 · 2 · 2 · 1',caption:'Six scale notes surround the tonic. Blend Gamma to compare the six positions; blend Alpha to see their place in the full period.',steps:7,accent:C.hex,atlas:'gamma',link:'Explore Gamma in the Research Atlas →',workings:'<p>The seven distinct notes of C major are <code>C D E F G A B</code>. C at the centre leaves six notes on the perimeter.</p><p>The steps from one note to the next, including the return to C, are <code>2, 2, 1, 2, 2, 2, 1</code> semitones. Equal distances in the drawing represent positions in the arrangement.</p><p>Move from C major to G major and six note names remain: <code>C D E G A B</code>. F changes to F♯.</p><p>The Gamma overlay uses the Atlas positions <code>24, 4, 8, 12, 16, 20</code>, with Alpha values <code>9, 3, 3, 9, 6, 6</code>. These six equally spaced positions align with the six scale notes around the tonic. Gamma also joins alternating vertices into two opposing equilateral triangles, forming the six-pointed star within the hexagon. The note names and the modular values remain visible as separate labels.</p><p>Blend Alpha to reveal all 24 positions. We hypothesise that these are possible Fibonacci structures within M9 / P24. The overlay makes the proposed spatial relationship available for inspection.</p>'},
 chromatic:{label:'CHROMATIC',title:'The chromatic octave',description:'Follow the twelve chromatic pitch classes around the circle and across the keyboard. From C to the next C, five black keys sit alongside eight white keys.',equation:'2 + 3 = 5\n5 + 8 = 13',caption:'13 keys from C to C, including the repeated upper C. The circle contains 12 distinct pitch classes.',steps:13,accent:C.gamma,atlas:null,link:'',workings:'<p>The chromatic scale includes every semitone in the octave. On the circle, the seven natural notes <code>C D E F G A B</code> are white; the five black-key notes <code>C♯ D♯ F♯ G♯ A♯</code> are pink. Sharp spellings are used here; on an equal-tempered keyboard these keys also have flat names.</p><p>The major-scale steps are <code>2, 2, 1, 2, 2, 2, 1</code> semitones. A black key fills each two-semitone gap. E–F and B–C are already one semitone apart, creating the familiar groups of two and three black keys.</p><p><code>2 + 3 = 5</code> counts the black keys. <code>5 + 8 = 13</code> counts all keys from C to C, including both endpoints. The numbers <code>2, 3, 5, 8, 13</code> are consecutive Fibonacci numbers. This is a counting pattern in the illustrated keyboard span, not a derivation of chromatic tuning.</p><p>The upper C has twice the frequency of the lower C and belongs to the same pitch class. Without the repeated endpoint, there are <code>7 + 5 = 12</code> pitch classes. Choosing different endpoints changes the colour counts: C♯ to C♯ contains seven white keys and six black keys.</p><p>The circle uses twelve-tone equal temperament: every adjacent step has frequency ratio <code>2^(1/12)</code>. Equal angles represent equal semitone intervals. The separate Circle of fifths view explains pure 3:2 fifths, octaves and the tuning adjustment that closes the twelve-note circle.</p><p><a href="https://openmusictheory.github.io/scales.html" target="_blank" rel="noopener">Scales and scale degrees · Open Music Theory ↗</a><br><a href="https://www.phys.unsw.edu.au/jw/notes.html" target="_blank" rel="noopener">Notes, frequencies and equal temperament · UNSW ↗</a></p>'},
 fifths:{label:'CIRCLE OF FIFTHS',title:'Pythagoras and the scale',description:'Pythagorean tuning, traditionally attributed to Pythagoras, builds a scale from 3:2 fifths and 2:1 octaves. Bring the notes into one octave and order them by pitch: C, D, E, F, G, A, B. Repeat C above them to complete the eight-note scale.',equation:'Perfect fifth: 3 : 2\nOctave: 2 : 1',caption:'Pythagorean tuning builds seven distinct scale notes using fifths and octaves. The eighth note repeats the first, one octave higher.',steps:12,accent:C.gamma,atlas:'alpha',link:'Explore Alpha in the Research Atlas →',workings:'<p>The tuning system named after Pythagoras constructs a scale using two operations: multiply by <code>3/2</code> for a fifth, then divide by <code>2</code> when the note lies above the octave.</p><p>Starting from F, the chain <code>F → C → G → D → A → E → B</code> contains the C-major collection. Ordering those notes by pitch gives <code>C D E F G A B</code>.</p><p>In twelve-tone equal temperament each fifth is <code>2^(7/12) ≈ 1.49831</code>. Twelve steps return to the same pitch class.</p><p>Twelve pure 3:2 fifths exceed seven octaves by <code>531441/524288 ≈ 1.01364</code>. This small difference is the Pythagorean comma.</p><p>The numbers in <code>2:1</code> and <code>3:2</code> are consecutive Fibonacci numbers. This is the numerical relationship explored here.</p><p><code>3/2</code> is one ratio of successive Fibonacci numbers. The sequence <code>3/2, 5/3, 8/5, 13/8…</code> approaches the Phi Ratio. Repeating the fixed ratio 3/2 is a separate operation.</p><p><a href="https://www.mtosmt.org/classic/mto.98.4.4/mto.98.4.4.scholtz.html" target="_blank" rel="noopener">Pythagorean tuning: method and ratios ↗</a><br><a href="https://plato.stanford.edu/entries/pythagoras/" target="_blank" rel="noopener">Pythagoras: historical attribution ↗</a></p>'},
 tuning:{label:'TUNING & FREQUENCY',title:'A reference and a ratio',description:'Compare the proposed M9 shapes for 528, 111 and 7.38 Hz. Choose the reference frequency for A4 to see the scale frequencies change while their interval ratios stay the same.',equation:'f = A4 × 2^((n − 9)/12)',caption:'528 → 6: hexagon · 111 → 3: triangle · 7.38 → 9: circle. A proposed visual mapping from the written frequency in hertz.',steps:8,accent:C.alpha,atlas:'alpha',link:'Explore Alpha in the Research Atlas →',workings:'<p>The reference sets A4 to <code>432, 440 or 444 Hz</code>. Every equal-tempered note follows the same ratio rule. Here <code>n</code> is the number of semitones above C4.</p><p>Relative to A4 = 440 Hz, A4 = 432 Hz shifts every note by approximately −31.77 cents. A4 = 444 Hz shifts every note by approximately +15.67 cents. One semitone is 100 cents.</p><p>The separate 528 Hz tone is included as a named solfeggio-frequency example. A single tone does not define the tuning of a whole piece.</p><p>For the M9 diagrams, add the digits of the frequency written in hertz, omitting the decimal point, and reduce the sum modulo 9, displaying zero as 9: <code>528 → 15 → 6</code>, <code>111 → 3</code>, <code>7.38 → 18 → 9</code>. For 7.38 Hz, this operates on its written digits: <code>7 + 3 + 8 = 18</code>. The visual presets associate 6 with a hexagon, 3 with a triangle and 9 with the circle. They show the proposed mapping; the shapes are not waveforms.</p><p>Changing the tuning reference preserves relative intervals. The research question is whether these stimulus settings modulate emotional and biological responses.</p>'}
};
function renderSequence(t){
 const svg=$('#sound-diagram');reset(svg,'Doubling phrase lengths and the Beta cycle','4, 8, 16, 32, 64 and 128 bars reduce modulo 9 to 4, 8, 7, 5, 1 and 2.');
 label(svg,'PHRASE LENGTH',45,42,{'text-anchor':'start','font-size':11,fill:C.muted,class:'diagram-mono'});label(svg,'MODULO 9',535,42,{'font-size':11,fill:C.muted,class:'diagram-mono'});
 circle(svg,535,280,144,C.alpha,{'stroke-width':.7});
 for(let i=0;i<9;i++){const p=polar(535,280,144,i,9),q=polar(535,280,174,i,9);circle(svg,...p,2,C.muted,{fill:C.muted});label(svg,i||9,...q,{fill:[0,3,6].includes(i)?C.muted:C.white,'font-size':18});}
 const points=residues.map(n=>polar(535,280,144,n,9));if(blends.sequence.has('beta'))line(svg,[...points,points[0]],C.beta,{'stroke-width':7,opacity:.42,'data-overlay':'beta'});const track=line(svg,[...points,points[0]],blends.sequence.has('beta')?C.white:C.beta,{'stroke-width':2,'data-phrase-path':'true'});draw(track,t);
 phraseBars.forEach((bars,i)=>{const y=100+i*64,amount=clamp(t*6-i),opacity=.25+.75*amount;label(svg,String(bars),46,y,{'text-anchor':'start','font-size':17,fill:amount?C.white:C.muted});line(svg,[[94,y],[94+bars*1.65*amount,y]],C.beta,{'stroke-width':2,opacity});label(svg,'→ '+residues[i],333,y,{'font-size':15,opacity,fill:C.beta});});
 label(svg,'bars',46,493,{'text-anchor':'start','font-size':11,fill:C.muted});label(svg,'4 → 8 → 7 → 5 → 1 → 2',535,500,{'font-size':14,fill:C.beta,class:'diagram-mono'});
 if(t>0){const length=track.getTotalLength(),p=track.getPointAtLength(length*t);circle(svg,p.x,p.y,4,C.white,{fill:C.white});}
}
function renderMelody(t){
 const svg=$('#sound-diagram');reset(svg,'Melodic structure blended with Alpha M9 / P24','The piano highlights C, E, G and C. Matching values 1, 3, 5 and 8 appear at every matching position of the 24-position Alpha circle.');
 const narrow=matchMedia('(max-width:720px)').matches;svg.setAttribute('viewBox',narrow?'0 0 360 640':'0 0 720 640');
 const blended=blends.melody.has('alpha'),cx=narrow?180:360,cy=190,r=127,active=melodyDegrees[Math.min(3,Math.floor(t*4))];
 label(svg,'ALPHA · M9 / P24',cx,18,{'font-size':13,fill:C.alpha,class:'diagram-mono'});
 circle(svg,cx,cy,r,blended?C.alpha:C.guide,{'stroke-width':.8,'data-alpha-circle':'true'});
 alpha.forEach((value,i)=>{const selected=melodyDegrees.includes(value),lit=selected&&value===active,p=polar(cx,cy,r,i,24),q=polar(cx,cy,r+22,i,24);
  circle(svg,...p,lit?4:selected?2.8:1.5,selected?C.white:C.alpha,{fill:lit?C.alpha:selected?C.white:C.alpha,opacity:selected?1:blended?.7:.15,'data-alpha-value':value,'data-alpha-index':i,'data-selected':String(selected)});
  if(selected||blended)label(svg,value,...q,{'font-size':selected?17:14,fill:lit?C.alpha:selected?C.white:C.alpha,opacity:selected?1:.65});
 });
 label(svg,'1 · 3 · 5 · 8',cx,cy-8,{'font-size':22,fill:C.white,class:'diagram-mono'});label(svg,'matching Alpha values',cx,cy+22,{'font-size':12,fill:C.muted});
 const x=narrow?20:72,width=narrow?40:72,top=458,height=112;
 const melodyPoints=melodyDegrees.map((degree,i)=>[x+(degree-1)*width+width/2,420-(i%2)*38]);
 const melodyPath=line(svg,melodyPoints,C.alpha,{'stroke-width':1.7,'data-melody-connector':'true'});draw(melodyPath,t*4/3);
 melodyPoints.forEach((point,i)=>{const amount=clamp(t*4-i);line(svg,[point,[point[0],top]],C.guide,{'stroke-width':.7,'stroke-dasharray':'2 5'});circle(svg,...point,3,C.alpha,{fill:C.alpha,opacity:.3+.7*amount});label(svg,melodyDegrees[i],point[0],point[1]-18,{'font-size':17,fill:C.alpha});});
 if(t>0&&t<.75){const p=melodyPath.getPointAtLength(melodyPath.getTotalLength()*clamp(t*4/3));circle(svg,p.x,p.y,4,C.white,{fill:C.white});}
 label(svg,'C MAJOR · 1 → 3 → 5 → 8',cx,632,{'font-size':12,fill:C.muted,class:'diagram-mono'});
 names.forEach((name,i)=>{const selected=melodyDegrees.includes(i+1),lit=i+1===active,amount=selected?clamp(t*4-melodyDegrees.indexOf(i+1)):0;
  node('rect',{x:x+i*width,y:top,width:width-2,height,fill:lit?'#5ca8ff35':amount?'#5ca8ff15':'#ffffff06',stroke:selected?C.alpha:C.guide,'stroke-width':1,'data-piano-degree':i+1},svg);
  label(svg,name,x+i*width+width/2,top+height-24,{'font-size':20,fill:selected?C.white:C.muted});label(svg,i+1,x+i*width+width/2,top+height+29,{'font-size':18,fill:selected?C.alpha:C.muted});
 });
 for(const index of [0,1,3,4,5])node('rect',{x:x+(index+1)*width-width*.28,y:top,width:width*.56,height:64,fill:'#03070a',stroke:C.guide,'stroke-width':1},svg);
}
function renderFifths(t){
 const svg=$('#sound-diagram');reset(svg,'Circle of fifths','Twelve equal-tempered pitch classes arranged by fifths, with the seven notes of C major highlighted.');
 const cx=360,cy=273,r=180;circle(svg,cx,cy,r,C.guide);circle(svg,cx,cy,130,C.guide,{'stroke-width':.5,'stroke-dasharray':'2 8'});
 const points=Array.from({length:13},(_,i)=>polar(cx,cy,r,i,12));const track=line(svg,points,C.gamma,{'stroke-width':1.6});draw(track,t);if(t>0&&t<1){const p=track.getPointAtLength(track.getTotalLength()*t);circle(svg,p.x,p.y,4,C.white,{fill:C.white});}
 fifths.forEach((name,i)=>{const p=polar(cx,cy,r,i,12),q=polar(cx,cy,r+36,i,12),chosen=i<=5||i===11,visible=clamp(t*12-i);circle(svg,...p,chosen?4:2,chosen?C.alpha:C.muted,{fill:chosen?C.alpha:C.muted,opacity:.3+.7*visible});label(svg,name,...q,{'font-size':name.length>3?13:19,fill:chosen?C.white:C.muted,opacity:.45+.55*visible});});
 label(svg,'C MAJOR',cx,cy-23,{'font-size':12,fill:C.alpha,class:'diagram-mono'});label(svg,'F · C · G · D · A · E · B',cx,cy+10,{'font-size':16});label(svg,'C · D · E · F · G · A · B · C',cx,cy+43,{'font-size':11,fill:C.muted});
 label(svg,'3 : 2  /  perfect fifth',360,535,{'font-size':12,fill:C.gamma,class:'diagram-mono'});
}
function renderScale(t){
 const svg=$('#sound-diagram');reset(svg,key+' major: Scale with Alpha and Gamma overlays','The tonic sits at the centre. Six scale notes align with Gamma positions 24, 4, 8, 12, 16 and 20; Alpha adds the full 24-position sequence.');
 if(matchMedia('(max-width:720px)').matches)svg.setAttribute('viewBox','100 0 520 575');
 const cx=360,cy=260,r=170,notes=scales[key],points=Array.from({length:7},(_,i)=>polar(cx,cy,r,i,6)),gamma=blends.scale.has('gamma'),fullAlpha=blends.scale.has('alpha');
 circle(svg,cx,cy,r,C.alpha,{'stroke-width':.6});
 if(fullAlpha)alpha.forEach((value,i)=>{const p=polar(cx,cy,r,i,24),q=polar(cx,cy,r+25,i,24);circle(svg,...p,2,C.alpha,{fill:C.alpha,'data-overlay':'alpha','data-alpha-value':value,'data-alpha-index':i});label(svg,value,...q,{'font-size':14,fill:C.alpha,opacity:.8});});
 if(gamma){for(const indices of [[0,8,16,0],[4,12,20,4]]){const triangle=line(svg,indices.map(i=>polar(cx,cy,r,i,24)),C.gamma,{'stroke-width':1.7,fill:C.gamma,'fill-opacity':.035,'data-gamma-triangle':indices.join(',')});draw(triangle,t*1.35);}line(svg,points,C.gamma,{'stroke-width':7,opacity:.45,'data-overlay':'gamma'});gammaIndices.forEach((index,i)=>{const p=points[i];circle(svg,...p,5,C.gamma,{'stroke-width':1.5,'data-gamma-index':index});if(!fullAlpha)label(svg,alpha[index],...polar(cx,cy,r-22,i,6),{'font-size':17,fill:C.gamma});});}
 for(let i=0;i<6;i++){line(svg,[[cx,cy],points[i]],C.guide,{'stroke-width':.7,'stroke-dasharray':'2 5'});label(svg,notes[i+1],...polar(cx,cy,fullAlpha?r+59:r+33,i,6),{'font-size':23,fill:C.white,opacity:.25+.75*clamp(t*7-i),'data-scale-index':i});circle(svg,...points[i],3,C.hex,{fill:C.hex,opacity:.25+.75*clamp(t*7-i)});}
 const perimeter=line(svg,points,C.hex,{'stroke-width':2,'data-scale-path':'true'});draw(perimeter,t*7/6);if(t>0&&t<6/7){const p=perimeter.getPointAtLength(perimeter.getTotalLength()*t*7/6);circle(svg,p.x,p.y,4,C.white,{fill:C.white});}const inward=line(svg,[points[0],[cx,cy]],C.white,{'stroke-width':1.5});draw(inward,t*7-6);
 circle(svg,cx,cy,54,C.hex,{'stroke-width':.8});label(svg,notes[0],cx,cy-7,{'font-size':38});label(svg,'MAJOR',cx,cy+27,{'font-size':10,fill:C.muted,class:'diagram-mono'});
 label(svg,[...notes,notes[0]].join('   '),360,517,{'font-size':19});label(svg,'2     2     1     2     2     2     1',360,548,{'font-size':13,fill:C.hex,class:'diagram-mono'});
}
function renderChromatic(t){
 const svg=$('#sound-diagram');reset(svg,'The chromatic octave: twelve pitch classes and thirteen keys','Twelve equal-tempered pitch classes run clockwise from C. The keyboard spans C to C: five black keys in groups of two and three, and eight white keys including the repeated upper C. These counts form a Fibonacci counting pattern, not a derivation of chromatic tuning.');
 const narrow=svg.parentElement.clientWidth<620;svg.dataset.chromaticLayout=narrow?'stacked':'wide';svg.setAttribute('viewBox',narrow?'0 0 360 790':'0 0 720 510');
 const cx=narrow?180:176,cy=narrow?197:215,r=narrow?126:128,sweep=clamp(t/.76),active=Math.min(12,Math.floor(sweep*12));
 const countProgress=clamp((t-.76)/.24),countOpacity=.2+.8*countProgress;
 label(svg,'12 pitch classes · 13 keys from C to C',narrow?180:360,24,{'font-size':narrow?14:17,fill:C.white});
 circle(svg,cx,cy,r,C.guide,{'stroke-width':.7});
 const track=node('path',{d:`M ${cx},${cy-r} A ${r},${r} 0 1 1 ${cx},${cy+r} A ${r},${r} 0 1 1 ${cx},${cy-r}`,fill:'none',stroke:C.alpha,'stroke-width':1.6,'data-chromatic-path':'true'},svg);draw(track,sweep);
 chromaticNames.forEach((name,i)=>{
  const altered=chromaticBlack.includes(i),color=altered?C.gamma:C.white,amount=clamp(sweep*12-i+1),lit=active%12===i,point=polar(cx,cy,r,i,12),text=polar(cx,cy,r+25,i,12);
  circle(svg,...point,lit?4.5:3,color,{fill:lit&&active===12?C.hex:color,opacity:.25+.75*amount,'data-chromatic-pitch':i,'data-note':name,'data-key-color':altered?'black':'white'});
  label(svg,name,...text,{'font-size':altered?17:20,fill:color,opacity:.4+.6*amount});
 });
 if(sweep>0&&sweep<1){const point=track.getPointAtLength(track.getTotalLength()*sweep);circle(svg,point.x,point.y,4,C.white,{fill:C.white,'data-chromatic-tracer':'true'});}
 label(svg,'12',cx,cy-12,{'font-size':47});label(svg,'pitch classes',cx,cy+27,{'font-size':13,fill:C.muted});
 label(svg,'7 natural + 5 altered',cx,cy+r+62,{'font-size':15,fill:C.white});
 label(svg,'12 equal semitone steps',cx,cy+r+88,{'font-size':11,fill:C.muted});
 const x=narrow?20:366,w=narrow?40:41,top=narrow?491:187,height=124,keyboardCentre=x+4*w;
 // The circle and keyboard share semitone indices; the last C is index 12.
 semitones.forEach((pitch,i)=>{
  const amount=clamp(sweep*12-pitch+1),lit=active===pitch,upper=i===7;
  node('rect',{x:x+i*w,y:top,width:w-2,height,fill:lit?(upper?'#62d99824':'#e5edf220'):'#ffffff06',stroke:upper?C.hex:C.guide,'stroke-width':upper?1.3:1,'data-chromatic-key':pitch,'data-note':names[i],'data-key-color':'white','data-active':String(lit)},svg);
  label(svg,names[i],x+i*w+w/2-1,top+height-24,{'font-size':19,fill:upper?C.hex:C.white,opacity:.4+.6*amount});
 });
 chromaticBlack.forEach((pitch,i)=>{
  const gap=[1,2,4,5,6][i],amount=clamp(sweep*12-pitch+1),lit=active===pitch,keyX=x+gap*w-w*.28;
  node('rect',{x:keyX,y:top,width:w*.56,height:73,fill:lit?'#362032':'#03070a',stroke:C.gamma,'stroke-width':1,opacity:.35+.65*amount,'data-chromatic-key':pitch,'data-note':chromaticNames[pitch],'data-key-color':'black','data-active':String(lit)},svg);
  label(svg,chromaticNames[pitch],x+gap*w,top+44,{'font-size':14,fill:C.gamma,opacity:.4+.6*amount});
 });
 const bracket=(left,right,y,color,below=false,attrs={})=>{const depth=below?-8:8,path=line(svg,[[left,y+depth],[left,y],[right,y],[right,y+depth]],color,{'stroke-width':1.2,opacity:countOpacity,...attrs});draw(path,countProgress);};
 [[1,2,2],[4,6,3]].forEach(([first,last,count])=>{
  const left=x+first*w-w*.28,right=x+last*w+w*.28,centre=(left+right)/2;
  bracket(left,right,top-35,C.gamma,false,{'data-black-key-group':count});label(svg,count,centre,top-58,{'font-size':25,fill:C.gamma,opacity:countOpacity});
 });
 label(svg,'5 black keys',keyboardCentre,top-13,{'font-size':12,fill:C.muted,opacity:countOpacity});
 bracket(x+2,x+8*w-4,top+height+15,C.white,true,{'data-white-key-count':8});label(svg,'8 white keys',keyboardCentre,top+height+36,{'font-size':13,fill:C.white,opacity:countOpacity});
 const upperX=x+7.5*w-1;line(svg,[[upperX,top+height],[upperX,top+height+59],[keyboardCentre+55,top+height+59]],C.hex,{'stroke-width':.8,opacity:countOpacity});
 label(svg,'Upper C repeats the tonic',keyboardCentre,top+height+76,{'font-size':11,fill:C.hex,opacity:countOpacity});
 const equationY=narrow?717:438;
 label(svg,'2 + 3 = 5',narrow?180:466,equationY,{'font-size':narrow?21:24,fill:C.gamma,opacity:countOpacity,'data-chromatic-equation':'black'});
 label(svg,'5 + 8 = 13',narrow?180:620,equationY+(narrow?34:0),{'font-size':narrow?21:24,fill:C.hex,opacity:countOpacity,'data-chromatic-equation':'span'});
 if(!narrow)line(svg,[[348,75],[348,461]],C.guide,{'stroke-width':.6,opacity:.6});
 if(!narrow)label(svg,'13 includes the repeated C · 12 distinct pitch classes',360,488,{'font-size':12,fill:C.muted,opacity:countOpacity});
}

function renderTuning(t){
 const svg=$('#sound-diagram');reset(svg,'Tuning, frequency and proposed M9 shapes','528 Hz has digit sum 15 and M9 value 6, shown as a hexagon. 111 Hz has digit sum 3, shown as a triangle. The digits of 7.38 Hz sum to 18, with zero displayed as 9, shown as a circle.');
 const narrow=matchMedia('(max-width:720px)').matches;if(narrow)svg.setAttribute('viewBox','0 0 360 1210');
 [528,111,7.38].forEach((hz,index)=>{
  const mapping=frequencyMapping(hz),cx=130+230*index,cy=151,r=72,color=index===0?C.hex:index===1?C.gamma:C.alpha;
  const group=node('g',{transform:narrow?`translate(${180-cx},${index*310})`:'','data-frequency':hz,'data-residue':mapping.residue,'data-shape':mapping.shape},svg);
  label(group,hz+' Hz',cx,28,{'font-size':21});circle(group,cx,cy,r,C.guide,{'stroke-width':.7});
  for(let i=0;i<9;i++){const p=polar(cx,cy,r,i,9),q=polar(cx,cy,r+18,i,9);circle(group,...p,1.8,C.muted,{fill:C.muted});label(group,i||9,...q,{'font-size':12,fill:C.muted});}
  let shape;if(mapping.shape==='circle')shape=circle(group,cx,cy,r,color,{'stroke-width':2});else{const sides=mapping.residue;shape=line(group,Array.from({length:sides+1},(_,i)=>polar(cx,cy,r,i,sides)),color,{'stroke-width':2});}shape.setAttribute('data-frequency-shape','true');draw(shape,t);
  label(group,mapping.residue,cx,cy,{'font-size':30,fill:color});label(group,mapping.digits.join(' + ')+' = '+mapping.sum,cx,266,{'font-size':15,fill:color});label(group,mapping.sum+' mod 9 = '+(mapping.sum%9)+(mapping.residue===9?' → display 9':''),cx,294,{'font-size':13,fill:C.muted});
 });
 const lower=node('g',{transform:narrow?'translate(0,620)':''},svg),x0=narrow?25:65,x1=narrow?335:655,seconds=.02,y=380;
 label(lower,'A4 = '+reference+' Hz · 20 ms',x0,333,{'text-anchor':'start','font-size':14,fill:C.alpha});line(lower,[[x0,y],[x1,y]],C.guide);const wave=line(lower,Array.from({length:401},(_,n)=>[x0+(x1-x0)*n/400,y-24*Math.sin(2*Math.PI*reference*seconds*n/400)]),C.alpha,{'stroke-width':1.4});draw(wave,t);
 const hz=frequencies(reference);label(lower,'C MAJOR · 12-TONE EQUAL TEMPERAMENT',narrow?180:360,442,{'font-size':12,fill:C.muted,class:'diagram-mono'});hz.forEach((value,i)=>{const x=narrow?45+(i%4)*90:65+i*84,row=narrow?Math.floor(i/4)*56:0;label(lower,names[i]+(i===7?'5':'4'),x,480+row,{'font-size':17});label(lower,round(value),x,505+row,{'font-size':13,fill:C.alpha,class:'diagram-mono'});});label(lower,'Hz',narrow?180:360,narrow?588:547,{'font-size':12,fill:C.muted});
}
const renderers={sequence:renderSequence,melody:renderMelody,fifths:renderFifths,scale:renderScale,chromatic:renderChromatic,tuning:renderTuning};
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
function transport(){
 const play=$('#sound-play'),svg=$('#sound-diagram');let progress=1,playing=false,last=0,elapsed=0,request=0;
 const duration=()=>({sequence:3000,melody:2400,fifths:3400,scale:2800,chromatic:4600,tuning:2400}[soundView]);
 function update(){play.textContent=playing?'Ⅱ':'▷';play.setAttribute('aria-label',(playing?'Pause ':'Play ')+'sound diagram animation');play.title=playing?'Pause animation':'Play animation';play.setAttribute('aria-pressed',String(playing));svg.dataset.view=soundView;svg.dataset.progress=progress.toFixed(4);svg.dataset.playing=String(playing);renderers[soundView](progress);}
 function pause(){playing=false;cancelAnimationFrame(request);request=0;last=0;update();}
 function frame(now){if(!playing)return;if(last)elapsed+=Math.min(now-last,80);last=now;const cycle=duration()+1100;if(elapsed>=cycle)elapsed%=cycle;progress=clamp(elapsed/duration());update();request=requestAnimationFrame(frame);}
 function start(replay=false){if(activeArea!=='sound'||document.hidden)return;if(replay||progress>=1){elapsed=0;progress=0;}playing=true;last=0;cancelAnimationFrame(request);update();request=requestAnimationFrame(frame);}
 function restart(){pause();elapsed=0;progress=reduced.matches?1:0;update();if(activeArea==='sound'&&!reduced.matches)start(true);}
 play.addEventListener('click',()=>playing?pause():start());$('#sound-replay').addEventListener('click',()=>start(true));
 $('#sound-svg').addEventListener('click',()=>{
 const wasPlaying=playing;pause();const current=progress;progress=1;update();
 const clone=svg.cloneNode(true);progress=current;update();if(wasPlaying)start();
 const box=svg.viewBox.baseVal;clone.setAttribute('xmlns',NS);clone.setAttribute('width',String(box.width));clone.setAttribute('height',String(box.height));
 const background=document.createElementNS(NS,'rect');background.setAttribute('x',String(box.x));background.setAttribute('y',String(box.y));background.setAttribute('width',String(box.width));background.setAttribute('height',String(box.height));background.setAttribute('fill','#05090c');clone.prepend(background);
 node('style',{},clone).textContent='text{font-family:Arial,Helvetica,sans-serif;dominant-baseline:central;letter-spacing:.02em}.diagram-mono{font-family:monospace;letter-spacing:.04em}';
 const blob=new Blob([new XMLSerializer().serializeToString(clone)],{type:'image/svg+xml'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='universal-informatics-'+soundView+'.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 });
 return{pause,restart,refresh:update};
}
const soundTransport=transport();
const visionMotion=$('#vision-motion'),visionArea=$('#area-vision'),visionReduced=matchMedia('(prefers-reduced-motion: reduce)');
function setVisionMotion(playing){visionArea.dataset.motion=playing?'playing':'paused';visionMotion.setAttribute('aria-pressed',String(!playing));visionMotion.textContent=playing?'Pause comparison':'Play comparison';}
visionMotion.addEventListener('click',()=>setVisionMotion(visionArea.dataset.motion!=='playing'));
visionReduced.addEventListener('change',()=>setVisionMotion(!visionReduced.matches));
setVisionMotion(!visionReduced.matches);

function selectSound(view,{focus=false}={}){
 if(!descriptions[view])view='sequence';soundTransport.pause();soundView=view;const info=descriptions[view];
 for(const button of document.querySelectorAll('[data-sound-view]')){const selected=button.dataset.soundView===view;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;if(selected&&focus)button.focus();}
 $('#sound-construction').setAttribute('aria-labelledby','sound-tab-'+view);$('#area-sound').style.setProperty('--sound-accent',info.accent);$('#sound-construction').style.setProperty('--sound-accent',info.accent);
 $('#sound-view-label').textContent=info.label;$('#sound-view-title').textContent=info.title;$('#sound-view-description').textContent=info.description;$('#sound-equation').textContent=info.equation;$('#sound-caption').textContent=info.caption;$('#sound-workings>div').innerHTML=info.workings;$('#sound-workings').open=false;
 const link=$('#sound-atlas-link');link.hidden=!info.atlas;if(info.atlas)link.dataset.scienceAtlas=info.atlas;else delete link.dataset.scienceAtlas;link.textContent=info.link;
 $('#sound-options').replaceChildren();$('#sound-blends').replaceChildren();
 const overlays={sequence:['beta'],melody:['alpha'],scale:['gamma','alpha']}[view]||[];
 const blendCaption=()=>{const selected=blends[view]||new Set();$('#sound-blend-caption').textContent=view==='sequence'&&selected.has('beta')?'White: phrase lengths · Yellow: Beta · The same six positions.':view==='melody'&&selected.has('alpha')?'White: selected melodic values · Blue: the full Alpha sequence.':view==='scale'&&selected.size?'Green: scale · Pink: Gamma · Blue: Alpha. Shared positions, separate labels.':'';};
 for(const sequence of overlays){const button=document.createElement('button');button.type='button';button.dataset.blend=sequence;button.textContent='Blend '+sequence[0].toUpperCase()+sequence.slice(1);button.style.setProperty('--blend-color',C[sequence]);button.setAttribute('aria-pressed',String(blends[view].has(sequence)));button.addEventListener('click',()=>{const selected=blends[view];selected.has(sequence)?selected.delete(sequence):selected.add(sequence);button.setAttribute('aria-pressed',String(selected.has(sequence)));blendCaption();soundTransport.refresh();});$('#sound-blends').append(button);}blendCaption();
 if(view==='scale'){$('#sound-options').innerHTML='<label class="atlas-field">Major scale<select id="sound-key">'+Object.keys(scales).map(name=>`<option value="${name}">${name} major</option>`).join('')+'</select></label>';$('#sound-key').value=key;$('#sound-key').addEventListener('change',e=>{key=e.target.value;soundTransport.restart();});}
 if(view==='tuning'){$('#sound-options').innerHTML='<label class="atlas-field">Tuning reference<select id="sound-reference"><option value="432">A4 = 432 Hz</option><option value="440">A4 = 440 Hz</option><option value="444">A4 = 444 Hz</option></select></label>';$('#sound-reference').value=reference;$('#sound-reference').addEventListener('change',e=>{reference=Number(e.target.value);soundTransport.restart();});}
 soundTransport.restart();
 if(activeArea==='sound'){const url=new URL(location.href);url.searchParams.set('sound',view);history.replaceState(null,'',url);}
}
document.querySelectorAll('[data-sound-view]').forEach(button=>button.addEventListener('click',()=>selectSound(button.dataset.soundView)));
$('.sound-view-nav').addEventListener('keydown',event=>{const keys=['ArrowLeft','ArrowRight','Home','End'];if(!keys.includes(event.key))return;event.preventDefault();const views=Object.keys(descriptions),index=views.indexOf(soundView);const next=event.key==='Home'?0:event.key==='End'?views.length-1:(index+(event.key==='ArrowRight'?1:-1)+views.length)%views.length;selectSound(views[next],{focus:true});});
document.addEventListener('atlas:area',event=>{activeArea=event.detail;soundTransport.pause();if(activeArea==='sound')selectSound(new URLSearchParams(location.search).get('sound')||soundView);});
document.addEventListener('visibilitychange',()=>{document.body.classList.toggle('page-hidden',document.hidden);if(document.hidden)soundTransport.pause();});
window.addEventListener('resize',()=>soundTransport.refresh());
window.addEventListener('pagehide',()=>soundTransport.pause());
selectSound(new URLSearchParams(location.search).get('sound')||'sequence');
})();
