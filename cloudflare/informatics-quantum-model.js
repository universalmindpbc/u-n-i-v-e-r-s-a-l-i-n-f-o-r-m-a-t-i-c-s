// A bounded read-ordering teaching model, not a genome assembler.
export function overlap(a,b){for(let k=Math.min(a.length,b.length);k>0;k--)if(a.slice(-k)===b.slice(0,k))return k;return 0;}
export function compileAssembly(reads){
 if(!Array.isArray(reads)||reads.length<3||reads.length>6||new Set(reads).size!==reads.length||reads.some(r=>typeof r!=='string'||!/^[ACGT]{4,40}$/.test(r)))throw Error('Use 3–6 distinct DNA fragments, each 4–40 A/C/G/T bases.');
 if(reads.some((r,i)=>reads.some((s,j)=>i!==j&&s.includes(r))))throw Error('This example needs fragments that are not fully contained in another fragment.');
 const n=reads.length,weights=reads.map((a,i)=>reads.map((b,j)=>i===j?0:overlap(a,b)));
 let best=-1,orders=[],tested=0;
 const walk=(order,score)=>{if(order.length===n){tested++;if(score>best){best=score;orders=[order];}else if(score===best)orders.push(order);return;}for(let i=0;i<n;i++)if(!order.includes(i))walk([...order,i],score+(order.length?weights[order.at(-1)][i]:0));};walk([],0);
 const order=orders[0],pieces=order.map((i,p)=>({read:i,sequence:reads[i],overlap:p?weights[order[p-1]][i]:0}));
 const sequence=pieces.reduce((s,p)=>s+p.sequence.slice(p.overlap),'');
 const penalty=(n-1)*Math.max(...weights.flat())+1,linear=Array(n*n).fill(-2*penalty),terms=new Map();
 const add=(i,j,v)=>{if(!v)return;const key=[Math.min(i,j),Math.max(i,j)].join(',');terms.set(key,(terms.get(key)||0)+v);};
 for(let i=0;i<n;i++)for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){add(i*n+p,i*n+q,2*penalty);add(p*n+i,q*n+i,2*penalty);}
 for(let p=0;p<n-1;p++)for(let i=0;i<n;i++)for(let j=0;j<n;j++)if(i!==j)add(i*n+p,j*n+p+1,-weights[i][j]);
 const quadratic=[...terms].map(([ij,value])=>({i:+ij.split(',')[0],j:+ij.split(',')[1],value})).filter(t=>t.value);
 const h=linear.map(x=>-x/2),J=quadratic.map(t=>({i:t.i,j:t.j,value:t.value/4}));
 for(const t of quadratic){h[t.i]-=t.value/4;h[t.j]-=t.value/4;}
 const constant=2*n*penalty,offset=constant+linear.reduce((s,x)=>s+x/2,0)+quadratic.reduce((s,t)=>s+t.value/4,0);
 return {version:'read-ordering-v1',method:'Exact classical enumeration',reads,weights,order,pieces,sequence,overlapScore:best,optimalOrders:orders.length,tested,disconnected:pieces.slice(1).some(p=>p.overlap===0),assumptions:['Error-free fragments in a known common orientation.','Each supplied fragment appears once; no reverse complements or repeat resolution.','Maximum overlap is an optimisation objective, not proof of a true biological sequence.'],qubo:{variables:n*n,index:'read * fragment_count + position; zero-based',constant,linear,quadratic,penalty},ising:{convention:'x=(1-z)/2; z is the Pauli-Z eigenvalue',offset,h,J},execution:{classical:true,gpu:false,qpu:false,wolfram:false}};
}
