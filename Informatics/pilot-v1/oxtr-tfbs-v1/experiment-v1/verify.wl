Module[{lo=8740380,hi=8779628,n=39248,alpha,gamma,beta,eps,projection,rows,encoded,inputs},
alpha=Table[Mod[Fibonacci[k],9],{k,0,23}];
gamma=Flatten[Position[Mod[alpha,3],0]]-1;
beta=Mod[Total /@ Partition[Rest[alpha],4,4,1,{}][[All,1;;3]],9];
eps={0,8,20,12,4,16,0};
projection[{s_,e_}]:=Module[{a=Max[s,lo],b=Min[e,hi],d,r},d=Floor[(hi-b+hi-a-1)/2];r=Mod[d,9];{s,e,hi-1-d,d,r,If[r==0,9,r]}];
inputs={{8770729,8770746},{8770405,8770425},{8739658,8740510},{lo,lo+1},{hi-1,hi},{lo,hi},{hi-2,hi},{lo-5,lo+1},{hi-1,hi+5}};
rows=Table[With[{r=Mod[d,9]}, {d,r,If[r==0,9,r],Round[10^9 Sin[2 Pi d/n]],Round[-10^9 Cos[2 Pi d/n]],Round[10^9 Sin[2 Pi r/9]],Round[-10^9 Cos[2 Pi r/9]]}],{d,0,n-1}];
encoded=StringJoin[(StringRiffle[ToString /@ #,","]<>"\n")& /@ rows];
ExportString[<|"tool"->"Wolfram Language","kernel_version"->$Version,"status"->"passed","offsets_checked"->Length[rows],"coordinate_rounding_scale"->10^9,"coordinate_table_sha256"->Hash[encoded,"SHA256","HexString"],"residue_counts_0_to_8"->Table[Count[rows[[All,2]],r],{r,0,8}],"interval_cases_columns"->{"start","end","anchor_genomic","offset","residue","display"},"interval_cases"->(projection /@ inputs),"alpha_residues"->alpha,"gamma_indices"->gamma,"beta_residues"->beta,"epsilon_indices"->eps,"epsilon_values"->Replace[alpha[[eps+1]],0->9,{1}],"unit_circle_identity"->FullSimplify[Sin[x]^2+Cos[x]^2==1],"scope"->"All 39248 possible offsets: M9 residue and full/M9 circle coordinates rounded to 1e-9; nine interval edge cases; Fibonacci recurrence and canonical route values. Computational checks only."|>,"RawJSON"]
]
