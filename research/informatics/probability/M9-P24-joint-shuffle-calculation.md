# M9 P24: joint Gamma and Beta shuffle probability

Calculated 18 October 2026. Exact integer enumeration, independently cross-checked.

## Probability

**Approximately 1 in 195.1 billion.**

Shuffle the 24 Alpha values around the circle. This is the chance of recovering
the original Gamma arrangement and deriving Beta as **4, 8, 7, 5, 1, 2**, with
the same starting position.

Comparison rule agreed with Ben: retain each record's own original alignment.
For M9 P24, Beta begins with 4 in the interval from position 0 to position 4.

## Question and model

Uniformly shuffle the 24 existing entries of one Fibonacci modulo-9 cycle onto
24 fixed positions, numbered 0 through 23 clockwise from north. Preserve every
value and its frequency. Display residue 0 as 9.

The original cycle is:

`9, 1, 1, 2, 3, 5, 8, 4, 3, 7, 1, 8, 9, 8, 8, 7, 6, 4, 1, 5, 6, 2, 8, 1`.

For the principal result, require:

1. Gamma values `9, 3, 3, 9, 6, 6` at positions `0, 4, 8, 12, 16, 20`, respectively.
2. Derive each Beta value by summing the three Alpha entries strictly between
   consecutive Gamma nodes and reducing modulo 9. Include the final gap from
   position 20 back to position 0.
3. The resulting six Beta values are exactly `4, 8, 7, 5, 1, 2`, beginning
   with the gap from position 0 to position 4 and proceeding clockwise.

Both the original Gamma arrangement and the original Beta alignment are fixed.
Other cyclic starting points are counted separately in the alternatives below.

The north reference and Gamma placement remain fixed. The 24-position framework
and residue frequencies are conditions of the experiment. This calculation does
not assign a probability to the emergence of the Pisano period from arbitrary
inputs or require a shuffled arrangement to satisfy the Fibonacci recurrence.

## Exact count

The frequencies in Alpha are two each of residues `0, 2, 3, 4, 5, 6, 7`, and five
each of `1, 8`. Consequently there are

\[
T=\frac{24!}{(2!)^7(5!)^2}=336,614,801,287,564,800
\]

distinct value arrangements. These are equally likely under a uniform shuffle
of the 24 cards, because every value arrangement has the same number of labelled
card permutations.

After fixing Gamma, the remaining values have frequencies

`1:5, 2:2, 4:2, 5:2, 7:2, 8:5`.

There are

\[
R=\frac{18!}{(5!)^2(2!)^4}=27,788,080,320
\]

remaining arrangements. Gamma alone therefore has probability

\[
P(G)=R/T=1/12,113,640.
\]

The earlier Appendix D expression `8 / (24 × 23 × 22 × 21 × 20 × 19)` is correct
for this Gamma-only event. Its denominator product is **96,909,120**, giving
**1 / 12,113,640**, rather than the previously stated approximately 1 in 12.6 million.

### Counting Beta without assuming independence

For the fixed Beta target `4, 8, 7, 5, 1, 2`, each gap has six possible unordered
triple types drawn from `1, 2, 4, 5, 7, 8` whose sum has the required residue.
For example, residue 4 admits:

`(1,1,2), (1,4,8), (1,5,7), (2,4,7), (4,4,5), (7,7,8)`.

Exhaustively checking all `6^6 = 46,656` combinations of these triple types leaves
**198** combinations using exactly the available frequencies. For each gap,
a triple with multiplicities `c_v` has `3! / product(c_v!)` internal orderings.
Multiplying the six gap weights and summing over the 198 combinations gives

\[
N_{\mathrm{fixed\ phase}}=1,725,543.
\]

Thus the joint probability is

\[
\begin{aligned}
P(G\cap B)
&=P(G)\,P(B\mid G)\\
&=\frac{1}{12,113,640}\times
  \frac{1,725,543}{27,788,080,320}\\
&=\frac{263}{51,305,410,956,800}\\
&\approx5.1261649618\times10^{-12}.
\end{aligned}
\]

This is approximately **1 in 195,077,608,201**, or **1 in 195.1 billion**.
The conditional Beta probability is approximately 1 in 16,103.963. It is obtained
from the remaining multiset directly, without treating the six gap sums as
independent random values.

## Matching-rule alternatives

Each of the six cyclic Beta targets has the same count, verified separately.
Accepting all six starting points gives `6 × 1,725,543 = 10,353,258` arrangements.

| Gamma requirement | Beta requirement | Joint probability, approximately |
|---|---|---:|
| Six fixed Gamma positions, allowing the 9/3/6 values in any order | Clockwise doubling cycle, any starting point | 1 in 361,254,830 |
| Exact Gamma values at their original positions | Clockwise doubling cycle, any starting point | 1 in 32,512,934,700 |
| Exact Gamma values at their original positions | Exact original `4,8,7,5,1,2` alignment — selected comparison rule | **1 in 195,077,608,201** |

The first alternative allows `6!/(2!2!2!) = 90` orders of the Gamma values at
the same positions. The last alternative accepts one of the six Beta phases.
None of these rows allows the six Gamma positions to rotate away from the fixed
indices. Different matching rules define different events.

## Verification and interpretation

Three exact counting procedures agree:

1. Dynamic programming over remaining value multiplicities and gap residues.
2. Exhaustive enumeration of all 46,656 combinations of allowed triple types.
3. Independent enumeration using 18 individually labelled cards and bitmasks,
   converting the result back to distinct value arrangements.

The labelled-card check finds 8,521,200 partitions into ordered gaps with
unordered triples. Multiplication by `(3!)^6` and division by `(5!)^2(2!)^4`
recovers 1,725,543 distinct arrangements for the fixed Beta phase. An unrestricted
version of the dynamic count recovers all 27,788,080,320 remaining arrangements.

These are exact descriptive shuffle probabilities for specified matches. The
calculation does not adjust for selecting M9 P24 or these patterns after
inspection. A formal significance assessment requires a defined selection and
comparison procedure. Biological relevance and computational usefulness are
separate empirical questions.

The record, source hash, counts, exact fractions and verification results are in
`m9p24_exact_shuffle.json`. The standard-library Python implementation is
`m9p24_exact_shuffle.py`. No website, Notion page, or canonical dataset was changed.
