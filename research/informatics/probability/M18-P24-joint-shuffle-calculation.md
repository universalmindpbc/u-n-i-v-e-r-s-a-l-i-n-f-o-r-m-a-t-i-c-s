# M18 P24: joint Gamma and Beta shuffle probability

Calculated 18 October 2026. Exact integer enumeration, independently cross-checked.

**Approximately 1 in 5.88 trillion.**

Shuffle the 24 Alpha values around the circle. This is the chance of recovering
the original Gamma arrangement and deriving Beta as **4, 8, 16, 14, 10, 2**, with
the same starting position.

## Native comparison

The selected records are M9 P24 and M18 P24. Each comparison retains its native
values, Gamma positions and Beta alignment. M27, M36 and M45 are not selected;
their probabilities remain null. This selection records the current research
scope, not an assertion that other records lack geometry.

The M18 Alpha cycle is:

`18, 1, 1, 2, 3, 5, 8, 13, 3, 16, 1, 17, 18, 17, 17, 16, 15, 13, 10, 5, 15, 2, 17, 1`.

Zero is displayed as 18. Shuffle the existing multiset uniformly over the fixed
24 clockwise positions. Require native Gamma values `18, 3, 3, 18, 15, 15`
at clock positions `24, 4, 8, 12, 16, 20`, respectively. Sum the three interior
Alpha values in each gap, excluding Gamma endpoints, and reduce each sum modulo
18. Require the ordered Beta sequence `4, 8, 16, 14, 10, 2`, beginning with
the interval from clock position 24 to 4. No rotation or reversal is accepted.

The reduction of this Beta sequence modulo 9 is a separate comparison; it is
not used to count native M18 matches.

## Exact count

Alpha frequencies are `0:2, 1:4, 2:2, 3:2, 5:2, 8:1, 10:1, 13:2, 15:2, 16:2, 17:4`.
The total number of equally likely distinct arrangements is:

`T = 24! / ((2!)^7 × (4!)^2) = 8,415,370,032,189,120,000`.

After fixing Gamma, the remaining frequencies are
`1:4, 2:2, 5:2, 8:1, 10:1, 13:2, 16:2, 17:4`:

`R = 18! / ((2!)^4 × (4!)^2) = 694,702,008,000`.

The six target Beta values admit `6, 7, 7, 6, 7, 7` unordered triple types.
Exhaustively checking their `6² × 7⁴ = 86,436` combinations leaves **74**
combinations using exactly the available frequencies. Weight each group by
`3! / product(multiplicity!)` to count its internal orders. The total is
**1,431,756 matching arrangements**.

`P(Gamma and Beta) = 1,431,756 / 8,415,370,032,189,120,000`

`= 491 / 2,885,929,366,320,000 ≈ 1 / 5,877,656,550,549.898`.

The joint count accounts for dependence between Gamma and Beta. The period,
Alpha frequencies and comparison targets are fixed conditions. This is an
exact configuration shuffle benchmark, not a WSS or biological-significance score.

## Independent verification

1. A memoised remaining-frequency count evaluates the six fixed Beta targets.
2. Exhaustive enumeration of all 86,436 triple-type combinations recovers the
   same count, with no random sampling.
3. A separate bitmask count treats the 18 remaining cards as individually labelled.
   It finds 282,816 unordered labelled gap partitions. Multiplication by `(3!)^6`
   and division by `(4!)^2 × (2!)^4` again gives 1,431,756.

The unrestricted group count also recovers R. All six Beta starting phases are
counted separately for audit; only the original alignment is published.

Run the downloaded `m18-probability.py` beside `m18.json`, or:

```sh
python3 research/informatics/probability/m18p24_exact_shuffle.py --record Informatics/pilot-v1/m18.json
```
