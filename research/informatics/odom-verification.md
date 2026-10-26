# Odom construction — Wolfram verification, 16 October 2026

The connected Wolfram Language evaluator returned:

- Triangle side lengths: `{Sqrt[3], Sqrt[3], Sqrt[3]}`.
- C on unit circle: `True`.
- AB / BC: `(1 + Sqrt[5])/2`.
- AC / AB: `(1 + Sqrt[5])/2`.
- BC / AB: `(-1 + Sqrt[5])/2`.
- Ratio checks against phi, phi and 1/phi: `{True, True, True}`.
- `1 + 1/phi - phi`: `0`.
- `Limit[Fibonacci[n+1]/Fibonacci[n], n -> Infinity]`: `(1 + Sqrt[5])/2`.

The evaluator emitted numerical-precision diagnostics on the first simplification and indeterminate-intermediate diagnostics on the limit evaluation, while returning the exact results above. The separate RootReduce ratio computations returned the exact radical values. Reproducible geometry input is in `verify_odom.wl`.

The visible formula is AB / BC = AC / AB = phi. Segment labels 1 and 1/phi use AB as the unit length. These labels specify relative lengths; they do not change the circle or claim that its circumference equals phi. AC = AB + BC, but a sum of unspecified lengths is not a dimensionless ratio.
