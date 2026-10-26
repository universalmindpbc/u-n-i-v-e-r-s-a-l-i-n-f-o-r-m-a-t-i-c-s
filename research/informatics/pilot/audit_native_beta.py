"""Audit native Beta doubling and the Atlas chevron across M1-M1001.

Run from any directory. This writes a research audit, not new Atlas records.
An independent implementation verifies every period, Beta and classification.
"""
import argparse
import hashlib
import json
from pathlib import Path

import build_pilot as atlas

ROOT = Path(__file__).resolve().parents[3]


def independent_period(modulus):
    # Unreduced integer Fibonacci numbers, stopping at the first return pair.
    first, second = 0, 1
    residues = []
    while True:
        residues.append(first % modulus)
        first, second = second, first + second
        if (first % modulus, second % modulus) == (0, 1 % modulus):
            return residues
        assert len(residues) <= modulus * modulus


def independent_beta(alpha, modulus):
    nodes = [i for i, value in enumerate(alpha) if value % 3 == 0]
    if len(nodes) < 2:
        return None
    prefix = [0]
    for value in alpha + alpha:
        prefix.append(prefix[-1] + value)
    endpoints = nodes[1:] + [nodes[0] + len(alpha)]
    return [(prefix[end] - prefix[start + 1]) % modulus
            for start, end in zip(nodes, endpoints)]


def shortest_repeat(values):
    # Prefix-function period detection, independent of the Atlas divisor search.
    if not values:
        return []
    prefix = [0] * len(values)
    for i in range(1, len(values)):
        j = prefix[i - 1]
        while j and values[i] != values[j]:
            j = prefix[j - 1]
        if values[i] == values[j]:
            j += 1
        prefix[i] = j
    length = len(values) - prefix[-1]
    return values[:length] if len(values) % length == 0 else values[:]


def independent_chevron(values, modulus):
    # Compare relative vertex angles directly using exact integer arithmetic.
    # No floating point, edge-angle implementation or reduction to M9 is used.
    cycle = shortest_repeat(values)
    if len(cycle) != 6 or len(set(cycle)) != 6:
        return False
    reference = [1, 2, 4, 8, 7, 5]
    return any(
        all((9 * (cycle[i] - cycle[0])
             - sign * modulus * (reference[(start + i) % 6] - reference[start]))
            % (9 * modulus) == 0 for i in range(6))
        for sign in (-1, 1) for start in range(6)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'research/informatics/native-beta-m1-m1001-audit.json')
    args = parser.parse_args()
    records = []
    exact_m9_beta_moduli = []
    published_checked = 0
    for modulus in range(1, 1002):
        alpha = atlas.period(modulus)
        nodes = [i for i, value in enumerate(alpha) if value % 3 == 0]
        beta = atlas.beta_for(alpha, nodes, modulus)
        values = beta['interval_residues'] if beta else None
        if values == [4, 8, 7, 5, 1, 2]:
            exact_m9_beta_moduli.append(modulus)
        doubling = bool(beta and beta['native_doubling'])
        chevron = atlas.chevron(beta, modulus)
        assert alpha == independent_period(modulus), modulus
        assert values == independent_beta(alpha, modulus), modulus
        independent_doubling = bool(values and len(set(values)) > 1
                                    and all((nxt - 2 * value) % modulus == 0
                                            for value, nxt in zip(values, values[1:] + values[:1])))
        assert doubling == independent_doubling, modulus
        assert chevron == independent_chevron(values, modulus), modulus
        if beta:
            assert beta['minimal_cycle'] == shortest_repeat(values), modulus
        published = ROOT / f'Informatics/pilot-v1/m{modulus}.json'
        if modulus <= 48:
            current = json.loads(published.read_text())
            assert current['alpha']['residues'] == alpha, modulus
            assert current['beta'] == beta, modulus
            shape = next(shape for shape in current['geometry_summary']['shapes']
                         if shape['name'] == 'Chevron')
            assert (shape['status'] == 'Present') == chevron, modulus
            published_checked += 1
        row = {
            'modulus': modulus, 'pisano_period': len(alpha),
            'm9_family': modulus % 9 == 0,
            'gamma_node_count': len(nodes), 'beta_defined': beta is not None,
            'beta_length': len(values) if values is not None else None,
            'minimal_beta_length': len(beta['minimal_cycle']) if beta else None,
            'native_doubling': doubling, 'native_chevron': chevron,
            'both': doubling and chevron,
            'beta_residues_sha256': hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest(),
            'independent_verification': 'Passed',
        }
        if doubling or chevron:
            row['native_beta_cycle'] = values
            row['minimal_native_beta_cycle'] = beta['minimal_cycle']
        records.append(row)
    matches = [row['modulus'] for row in records if row['both']]
    report = {
        'scope': 'All integer moduli M1-M1001 inclusive; native primary Gamma-derived Beta only.',
        'gamma_rule': 'Native Alpha residues divisible by 3, including zero.',
        'beta_rule': 'Sum each open cyclic interval between consecutive Gamma nodes, then reduce modulo the native modulus. Preserve the original starting interval.',
        'doubling_rule': 'A nonconstant cycle in which every next value equals twice its predecessor modulo the native modulus, including the closing step.',
        'chevron_rule': 'Six distinct vertices in the minimal Beta cycle; the exact M9 reference angular path on the native value circle, allowing shape rotation and reflection.',
        'interpretation': 'This is an exhaustive count under the Atlas rules in the stated range. It is not a random-shuffle probability, biological significance score, or result for moduli above 1001. Shape equivalence does not alter the fixed alignment used in probability calculations.',
        'atlas_rules_sha256': hashlib.sha256(Path(atlas.__file__).read_bytes()).hexdigest(),
        'verification': {'independent_implementations_agree': True,
                         'native_records_checked': len(records),
                         'published_records_match': published_checked},
        'summary': {
            'exact_native_m9_beta_moduli': exact_m9_beta_moduli,
            'moduli_checked': len(records), 'both_count': len(matches), 'both_moduli': matches,
            'additional_matches_beyond_m9_m18': [m for m in matches if m not in (9, 18)],
            'm9_family_checked': sum(row['m9_family'] for row in records),
            'm9_family_both_moduli': [m for m in matches if m % 9 == 0],
            'doubling_without_chevron_moduli': [row['modulus'] for row in records if row['native_doubling'] and not row['native_chevron']],
            'chevron_without_doubling_moduli': [row['modulus'] for row in records if row['native_chevron'] and not row['native_doubling']],
        },
        'records': records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'summary': report['summary'], 'verification': report['verification'],
                      'output': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
