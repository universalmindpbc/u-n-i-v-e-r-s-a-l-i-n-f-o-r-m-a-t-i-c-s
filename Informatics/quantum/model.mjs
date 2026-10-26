// Two neutral atoms, physical order gg, gr, rg, rr. Time in microseconds.
export const VERSION='rydberg-pair-v1';
export const BASIS=Object.freeze(['gg','gr','rg','rr']);
export const BASE=Object.freeze({distance_um:5.7,duration_us:4,omega_rad_us:6.3,detuning_rad_us:31.5});
export const BOUNDS=Object.freeze({distance_um:[4.5,9],duration_us:[2,6],omega_rad_us:[1,10],detuning_rad_us:[10,50]});
export function validate(input){
 if(!input||typeof input!=='object'||Array.isArray(input))throw Error('An experiment needs four numeric parameters.');
 const spec={};
 for(const [key,[lo,hi]] of Object.entries(BOUNDS)){
  const x=input[key];
  if(typeof x!=='number'||!Number.isFinite(x)||x<lo||x>hi)throw Error(`${key} must be between ${lo} and ${hi}.`);
  spec[key]=x;
 }
 return spec;
}
export const isBaseline=s=>Object.keys(BASE).every(k=>s[k]===BASE[k]);
export function pulse(t,s){
 const f=Math.max(0,Math.min(1,t/s.duration_us)),r=.025;
 return {omega:s.omega_rad_us*Math.min(1,f/r,(1-f)/r),delta:s.detuning_rad_us*(f<=r?-1:f>=1-r?1:-1+2*(f-r)/(1-2*r))};
}
export function solve(input,refinement=1){
 const s=validate(input),v=5.42e6/s.distance_um**6;
 // Align both pulse corners exactly to the grid; bound the largest phase step.
 const n=Math.ceil(s.duration_us*(v+2*s.detuning_rad_us+2*s.omega_rad_us)/.06/40)*40*refinement,dt=s.duration_us/n;
 let y=new Float64Array([1,0,0,0,0,0,0,0]);
 const a=new Float64Array(8),b=new Float64Array(8),c=new Float64Array(8),d=new Float64Array(8),tmp=new Float64Array(8),trace=[];
 function rhs(t,z,out){
  const {omega,delta}=pulse(t,s),h=omega/2,diag=[0,-delta,-delta,v-2*delta];
  for(let i=0;i<4;i++){
   const j=i^1,k=i^2;
   out[2*i]=diag[i]*z[2*i+1]+h*(z[2*j+1]+z[2*k+1]);
   out[2*i+1]=-diag[i]*z[2*i]-h*(z[2*j]+z[2*k]);
  }
 }
 const probs=()=>Array.from({length:4},(_,i)=>y[2*i]**2+y[2*i+1]**2);
 const stride=Math.max(1,Math.floor(n/160));
 trace.push({t:0,p:probs()});
 for(let step=0;step<n;step++){
  const t=step*dt;rhs(t,y,a);
  for(let k=0;k<8;k++)tmp[k]=y[k]+dt*a[k]/2;rhs(t+dt/2,tmp,b);
  for(let k=0;k<8;k++)tmp[k]=y[k]+dt*b[k]/2;rhs(t+dt/2,tmp,c);
  for(let k=0;k<8;k++)tmp[k]=y[k]+dt*c[k];rhs(t+dt,tmp,d);
  for(let k=0;k<8;k++)y[k]+=dt*(a[k]+2*b[k]+2*c[k]+d[k])/6;
  if((step+1)%stride===0||step===n-1)trace.push({t:(step+1)*dt,p:probs()});
 }
 const probabilities=probs(),normError=Math.abs(probabilities.reduce((a,b)=>a+b,0)-1);
 return {spec:s,probabilities,trace,normError,steps:n,interaction_rad_us:v};
}
export function verifiedSolve(input){
 const coarse=solve(input),fine=solve(input,2),convergence=Math.max(...fine.probabilities.map((v,i)=>Math.abs(v-coarse.probabilities[i])));
 if(convergence>1e-6||fine.normError>1e-6)throw Error('The numerical accuracy check failed. No result has been saved.');
 return {...fine,convergence,solver:'RK4 with step-halving check',solver_version:'rk4-pair-v1',basis:BASIS,assumptions:'Closed system; initial gg; phase zero; C6 = 5.42e-24 rad m^6/s. Excludes noise, preparation, readout error and atom loss.'};
}
