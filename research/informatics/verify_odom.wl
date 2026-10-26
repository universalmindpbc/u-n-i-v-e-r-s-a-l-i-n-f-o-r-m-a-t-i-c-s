(* Exact geometry: E,F,D on the unit circle, A and B side midpoints. *)
e={0,1};f={-Sqrt[3]/2,-1/2};d={Sqrt[3]/2,-1/2};
a=(e+f)/2;b=(e+d)/2;c={Sqrt[15]/4,1/4};
phi=(1+Sqrt[5])/2;
ab=EuclideanDistance[a,b];bc=EuclideanDistance[b,c];ac=EuclideanDistance[a,c];
{RootReduce[ab/bc],RootReduce[ac/ab],RootReduce[bc/ab],
 RootReduce[1+1/phi-phi],c.c==1,
 EuclideanDistance[e,f]==EuclideanDistance[f,d]==EuclideanDistance[d,e]}
(* Expected: {phi, phi, 1/phi, 0, True, True}. *)
