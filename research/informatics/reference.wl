(* Universal Informatics: M9/P24 visual reference. The path is proposed,
   not a physical electron-motion result. Evaluate in Mathematica. *)
alpha = Mod[Fibonacci[Range[0, 23]], 9];
period = SelectFirst[Range[1, 54], Mod[{Fibonacci[#], Fibonacci[# + 1]}, 9] == {0, 1} &];
nodes = {0, 4, 8, 12, 16, 20};
unicursal = {0, 8, 20, 12, 4, 16, 0};
labels = ReplaceAll[alpha[[Mod[unicursal, 24] + 1]], Rule[0, 9]];
point[n_, p_] := {Sin[2 Pi n/p], Cos[2 Pi n/p]};
beta = Mod[Total /@ Partition[Delete[alpha, List /@ (nodes + 1)], 3], 9];
doubling = Mod[2^Range[0, 6], 9];
Graphics[{Gray, Circle[], White, Thick, Line[point[#, 24] & /@ unicursal],
  Table[Text[Style[ReplaceAll[alpha[[n + 1]], Rule[0, 9]], 14], 1.12 point[n, 24]], {n, 0, 23}]},
  Background -> Black, ImageSize -> 700, PlotRange -> 1.25]
Graphics[{Gray, Circle[], White, Thick, Line[point[#, 9] & /@ doubling],
  Table[Text[Style[n, 14], 1.12 point[n, 9]], {n, 1, 9}]},
  Background -> Black, ImageSize -> 700, PlotRange -> 1.25]
