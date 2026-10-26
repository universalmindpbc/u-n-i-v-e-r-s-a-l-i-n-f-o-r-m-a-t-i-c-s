#!/usr/bin/env python3
"""Exact M18 P24 Gamma/Beta shuffle counts. Python standard library only.

Model: uniformly shuffle the existing 24 Alpha entries onto fixed, clockwise
positions 0..23. Preserve multiplicities; display 0 as 18. This is a descriptive
exact-match probability, not a selection-adjusted significance test.

Run from any directory:
    python3 m18-probability.py --record m18.json > calculation.json

Reads the approved Atlas record, without modifying any dataset or website.
"""

from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, product
from math import comb, factorial, prod
from pathlib import Path
import hashlib
import json


def multinomial(counts):
    return factorial(sum(counts)) // prod(factorial(c) for c in counts)


def probability(numerator, denominator):
    value = Fraction(numerator, denominator)
    return {
        "exact_fraction": str(value),
        "decimal": float(value),
        "percent": float(value * 100),
        "one_in_approx": float(1 / value),
    }


def calculate(source):
    source = Path(source)
    source_bytes = source.read_bytes()
    record = json.loads(source_bytes)
    alpha = tuple(record["alpha"]["residues"])
    nodes = tuple(record["gamma"]["alpha_indices"])
    gamma = tuple(alpha[i] for i in nodes)
    beta = tuple(sum(alpha[4*j+1:4*j+4]) % 18 for j in range(6))
    assert record["modulus"] == 18 and record["pisano_period"] == 24
    assert nodes == (0, 4, 8, 12, 16, 20)
    assert gamma == (0, 3, 3, 0, 15, 15)
    assert beta == tuple(record["beta"]["interval_residues"]) == (4, 8, 16, 14, 10, 2)
    assert all(beta[(i+1) % 6] == 2*beta[i] % 18 for i in range(6))

    remaining = tuple(alpha[i] for i in range(24) if i not in nodes)
    values = tuple(sorted(set(remaining)))
    counts = tuple(remaining.count(v) for v in values)
    assert values == (1, 2, 5, 8, 10, 13, 16, 17) and counts == (4, 2, 2, 1, 1, 2, 2, 4)
    total = multinomial(Counter(alpha).values())
    remaining_total = multinomial(counts)
    gamma_orders = multinomial(Counter(gamma).values())
    assert gamma_orders == 90
    assert Fraction(remaining_total, total) == Fraction(1, 12_113_640)
    assert Fraction(gamma_orders * remaining_total, total) == Fraction(1, comb(24, 6))

    options = {r: [] for r in range(18)}
    for triple in combinations_with_replacement(values, 3):
        used = tuple(triple.count(v) for v in values)
        options[sum(triple) % 18].append((used, multinomial(used)))

    def dynamic_count(target):
        @lru_cache(None)
        def visit(left, k):
            if k == 6:
                return int(not any(left))
            candidates = options[target[k]] if target else sum(options.values(), [])
            result = 0
            for used, weight in candidates:
                after = tuple(a-b for a, b in zip(left, used))
                if min(after) >= 0:
                    result += weight * visit(after, k+1)
            return result
        return visit(counts, 0)

    # Check that unrestricted block enumeration recovers all 18-value shuffles.
    assert dynamic_count(None) == remaining_total
    phases = [beta[i:] + beta[:i] for i in range(6)]
    phase_counts = [dynamic_count(target) for target in phases]
    assert len(set(phases)) == 6 and len(set(phase_counts)) == 1
    fixed_count = phase_counts[0]
    cycle_count = sum(phase_counts)

    # Independent enumeration: all 6^2 * 7^4 combinations of allowed triple multisets.
    # Retain only combinations using precisely the available values, then count
    # all internal orderings of each three-value gap.
    qualifying_combinations = 0
    direct_count = 0
    for picks in product(*(options[r] for r in beta)):
        used = tuple(sum(pick[0][i] for pick in picks) for i in range(len(values)))
        if used == counts:
            qualifying_combinations += 1
            direct_count += prod(pick[1] for pick in picks)
    assert direct_count == fixed_count

    # Second independent check: treat the 18 cards as individually labelled.
    # Count disjoint unordered triples by bitmask, multiply by 3! per gap, then
    # remove the labelling multiplicity to recover distinct value arrangements.
    masks = {r: [] for r in range(18)}
    for inds in combinations(range(18), 3):
        masks[sum(remaining[i] for i in inds) % 18].append(sum(1 << i for i in inds))

    @lru_cache(None)
    def labelled_partition_count(mask, k):
        if k == 6:
            return int(mask == 0)
        return sum(
            labelled_partition_count(mask ^ triple, k+1)
            for triple in masks[beta[k]] if mask & triple == triple
        )

    labelled_partitions = labelled_partition_count((1 << 18)-1, 0)
    unlabelled = Fraction(
        labelled_partitions * factorial(3)**6,
        prod(factorial(c) for c in counts),
    )
    assert unlabelled == fixed_count

    result = {
        "calculation_version": "m18p24-exact-shuffle-v1",
        "date": "2026-10-18",
        "source": source.name,
        "mathematical_input_sha256": hashlib.sha256(json.dumps({"alpha": alpha, "gamma_indices": nodes, "gamma_values": gamma, "beta": beta}, sort_keys=True).encode()).hexdigest(),
        "selected_comparison": {
            "rule": "Retain each record's original Gamma placement and original Beta alignment.",
            "probability_key": "joint_exact_gamma_fixed_beta_phase",
            "heading": "Probability",
            "display": "Approximately 1 in 5.88 trillion",
            "explanation": "Shuffle the 24 Alpha values around the circle. This is the chance of recovering the original Gamma arrangement and deriving Beta as 4, 8, 16, 14, 10, 2, with the same starting position.",
        },
        "model": {
            "sampling": "Uniform random permutation of existing 24 Alpha entries, preserving all multiplicities.",
            "positions": "Fixed indices 0..23, clockwise from north; 0 displayed as 18.",
            "gamma_fixed_indices": nodes,
            "gamma_fixed_values": [v or 18 for v in gamma],
            "beta_rule": "Sum three interior Alpha values in each gap, exclude Gamma endpoints, reduce modulo 18, include wraparound.",
            "beta_fixed_phase": beta,
            "beta_any_phase": phases,
            "beta_direction": "Clockwise doubling only; reversed cycles are not included.",
            "not_required": "Unshuffled Fibonacci order, original unreduced gap sums, or a biological outcome.",
            "interpretation": "Descriptive exact-match benchmark; no adjustment for prior selection of modulus, pattern, or matching rule.",
        },
        "counts": {
            "all_distinct_alpha_arrangements": total,
            "arrangements_given_exact_gamma": remaining_total,
            "gamma_value_orders_if_labels_are_free": gamma_orders,
            "valid_triple_types_per_fixed_beta_gap": [len(options[r]) for r in beta],
            "qualifying_six_triple_combinations": qualifying_combinations,
            "exact_gamma_and_fixed_beta_phase": fixed_count,
            "valid_remaining_arrangements_per_beta_phase": phase_counts,
            "exact_gamma_and_any_beta_phase": cycle_count,
            "labelled_unordered_gap_partitions_fixed_beta": labelled_partitions,
        },
        "probabilities": {
            "exact_gamma_only": probability(remaining_total, total),
            "beta_fixed_phase_given_gamma": probability(fixed_count, remaining_total),
            "beta_any_phase_given_gamma": probability(cycle_count, remaining_total),
            "joint_exact_gamma_fixed_beta_phase": probability(fixed_count, total),
            "joint_exact_gamma_any_beta_phase": probability(cycle_count, total),
            "joint_gamma_positions_only_any_beta_phase": probability(gamma_orders*cycle_count, total),
        },
        "verification": {
            "approved_native_record_assertions": "passed",
            "unrestricted_count_equals_multinomial": "passed",
            "six_rotations_counted_separately": "passed",
            "exhaustive_triple_multiset_enumeration": "passed",
            "independent_labelled_bitmask_enumeration": "passed",
            "monte_carlo_used": False,
        },
    }
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=Path(__file__).with_name("m18.json"))
    args = parser.parse_args()
    print(json.dumps(calculate(args.record), indent=2))
