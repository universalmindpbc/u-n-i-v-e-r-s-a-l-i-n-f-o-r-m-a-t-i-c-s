"""Prepare independently executable Wolfram comparisons for generated records."""
import sys,json,hashlib
from pathlib import Path
from expand_atlas import signature,OUT,ROOT
lo,hi=map(int,sys.argv[1:3])
expected=[signature(json.loads((OUT/f'm{m}.json').read_text())) for m in range(lo,hi+1)]
wl=lambda v:'{'+','.join(wl(x) if isinstance(x,list) else json.dumps(x) for x in v)+'}'
code='expected='+wl(expected)+'''; sig[x_]:=Hash[StringRiffle[ToString/@x,","],"SHA256","HexString"]; results=Table[Module[{m=e[[1]],p,a,n,s,t,h,actual},p=SelectFirst[Range[6m],Mod[Fibonacci[#],m]==0&&Mod[Fibonacci[#+1],m]==Mod[1,m]&];a=Mod[Fibonacci[Range[0,p-1]],m];n=Flatten[Position[Mod[a,3],0]]-1;s=If[Length[n]<2,{},Table[Total[a[[Mod[Range[n[[k]]+1,If[k<Length[n],n[[k+1]],n[[1]]+p]-1],p]+1]]],{k,Length[n]}]];t=If[Mod[p,3]==0,Select[Table[Range[i,p-1,p/3],{i,0,p/3-1}],SubsetQ[n,#]&],{}];h=If[Mod[p,6]==0,Select[Table[Range[i,p-1,p/6],{i,0,p/6-1}],SubsetQ[n,#]&],{}];actual={m,p,sig[a],sig[n],sig[s],sig[Mod[s,m]],sig[Flatten[t]],sig[Flatten[h]]};{m,MapThread[SameQ,{actual,e}]}],{e,expected}];ExportString[<|"range"->{First[expected][[1]],Last[expected][[1]]},"records"->Length[results],"comparisons"->Total[Length[#[[2]]]&/@results],"passed"->And@@Flatten[results[[All,2]]],"failures"->Select[results,!And@@#[[2]]&]|>,"RawJSON"]'''
out=ROOT/'research/informatics/data/m1-m1001-v1'/f'verify-{lo}-{hi}.wl';out.write_text(code)
print(json.dumps({'code':code,'sha256':hashlib.sha256(code.encode()).hexdigest(),'file':str(out)}))
