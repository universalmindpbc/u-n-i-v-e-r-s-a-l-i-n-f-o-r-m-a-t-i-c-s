"""Phase-preserving M9 views for Alpha, Beta, Gamma and Epsilon.

Alpha/Gamma use 24-index blocks. Epsilon applies the reference route within
each block, separately from qualification on the whole native period circle.
"""
import argparse
import json
from pathlib import Path
from build_pilot import period, beta_for
import drawing
from drawing import Drawing, pt, BLUE, PINK, FG, MUTED

ROUTE = [0, 8, 20, 12, 4, 16, 0]
REFERENCE = {
    'alpha': [x or 9 for x in period(9)],
    'beta': [4, 8, 7, 5, 1, 2],
    'gamma': [9, 3, 3, 9, 6, 6],
    'epsilon': [9, 3, 6, 9, 3, 6, 9],
}
SUMMARY = 'All 111 multiples of nine in M1–M1001 (11.1% of moduli) recover M9 P24 through exact modular reduction.'
FOLD = 'Reduce values modulo 9 and fold indices modulo 24, preserving the original start. Repeated 24-position blocks overlap in the diagram.'
EPSILON_RULE = 'Reduce Alpha modulo 9, then apply the fixed M9 Epsilon route within each 24-position block: 0 → 8 → 20 → 12 → 4 → 16 → 0. This is a cycle-based construction; native full-circle Epsilon is assessed separately.'


def verify_period(m):
    a = period(m)
    assert len(a) % 24 == 0
    repeats = len(a) // 24
    reduced = [v % 9 or 9 for v in a]
    nodes = [i for i, v in enumerate(a) if v % 3 == 0]
    beta = beta_for(a, nodes, m)
    checks = {
        'alpha': reduced == REFERENCE['alpha'] * repeats,
        'beta': [v % 9 for v in beta['interval_residues']] == REFERENCE['beta'] * repeats,
        'gamma': [reduced[i] for i in nodes] == REFERENCE['gamma'] * repeats,
        'gamma_indices': nodes == [24 * k + i for k in range(repeats) for i in range(0, 24, 4)],
        'epsilon_cycle_routes': all([reduced[24 * k + i] for i in ROUTE] == REFERENCE['epsilon'] for k in range(repeats)),
    }
    assert all(checks.values()), (m, checks)
    whole_route = [len(a) * k // 6 for k in [0, 2, 5, 3, 1, 4, 0]]
    return {'modulus': m, 'pisano_period': len(a), 'repeat_count': repeats,
            'checks': checks, 'original_alignment_preserved': True,
            'native_full_circle_epsilon_reduces_to_reference': [reduced[i] for i in whole_route] == REFERENCE['epsilon']}


def audit():
    rows = [verify_period(m) for m in range(9, 1002, 9)]
    return {'audit_id': 'm9-sequence-family-audit-v1', 'modulus_range': [1, 1001],
            'family_members_checked': len(rows), 'matching_family_members': len(rows),
            'coverage_fraction': '111/1001', 'coverage_percent': 111 / 1001 * 100,
            'summary': SUMMARY, 'identity': '(F_n mod (9k)) mod 9 = F_n mod 9',
            'classification': 'Exact modular reduction and cycle-based construction',
            'reference_cycles': REFERENCE, 'folding_rule': FOLD, 'epsilon_rule': EPSILON_RULE,
            'probability_relationship': 'The random-shuffle probability is a separate calculation. Modular recurrence is not a series of independent chance events.',
            'records': rows}


def projections(d, family_audit):
    m, p = d['modulus'], d['pisano_period']
    if m % 9:
        return None
    row = next(x for x in family_audit['records'] if x['modulus'] == m)
    repeats = row['repeat_count']
    views = {'beta': d['beta_m9_view']}
    for kind, native in [('alpha', d['alpha']['display_values']), ('gamma', d['gamma']['clockwise_values'])]:
        reduced = [v % 9 or 9 for v in native]
        assert reduced == REFERENCE[kind] * repeats
        views[kind] = {'version': 'm9-sequence-view-v1', 'native_values': native,
                       'reduced_values': reduced, 'display_cycle': REFERENCE[kind],
                       'native_circle_positions': p, 'reduced_circle_positions': 24,
                       'operation': FOLD}
    blocks = []
    for k in range(repeats):
        indices = [24 * k + i for i in ROUTE]
        native = [d['alpha']['display_values'][i] for i in indices]
        reduced = [v % 9 or 9 for v in native]
        assert reduced == REFERENCE['epsilon']
        blocks.append({'block': k + 1, 'alpha_indices': indices,
                       'native_values': native, 'reduced_values': reduced})
    views['epsilon'] = {
        'version': 'm9-sequence-view-v1', 'display_cycle': REFERENCE['epsilon'],
        'reduced_values': REFERENCE['epsilon'][:-1] * repeats + [9],
        'block_traversals': blocks, 'reference_alpha_indices': ROUTE,
        'native_circle_positions': p, 'reduced_circle_positions': 24,
        'native_369_qualifies': bool(d['epsilon']),
        'native_full_circle_comparison': d['epsilon_m9_comparison'],
        'operation': EPSILON_RULE,
    }
    for kind, view in views.items():
        if kind == 'beta':
            continue
        view.update({'native_modulus': m, 'target_modulus': 9,
                     'repeat_count': repeats, 'original_alignment_preserved': True,
                     'reference_cycle_match': True, 'folded_period': 24,
                     'family_audit': {'audit_id': family_audit['audit_id'], 'summary': SUMMARY,
                                      'audit_file': 'pilot-v1/m9-sequence-family-audit.json'},
                     'reduced_svg': f'pilot-v1/views/m{m}/{kind}-m9.svg',
                     'reduced_png': f'pilot-v1/views/m{m}/{kind}-m9.png'})
    return views


def draw(d, kind, view, out):
    drawing.OUT = out / f"views/m{d['modulus']}"
    drawing.OUT.mkdir(parents=True, exist_ok=True)
    title = kind.title() + ' · Reduced to M9'
    g = Drawing(title, view['operation'])
    g.circle((400, 365), 238, BLUE if kind == 'alpha' else '#34434e')
    if kind == 'alpha':
        for i, value in enumerate(REFERENCE['alpha']):
            g.circle(pt(i, 24), 2, BLUE, fill=BLUE)
            g.text(value, pt(i, 24, 267), 18)
            g.text(i or 24, pt(i, 24, 291), 10, MUTED)
    elif kind == 'gamma':
        triangles = [[0, 8, 16, 0], [4, 12, 20, 4]]
        for route in triangles:
            g.line([pt(i, 24) for i in route[:-1]], PINK, 2, True, '#261525')
        for route in triangles:
            g.line([pt(i, 24) for i in route], PINK, 2)
        for i, value in zip(range(0, 24, 4), REFERENCE['gamma']):
            g.circle(pt(i, 24), 4, PINK, fill=PINK)
            g.text(value, pt(i, 24, 270), 18)
            g.text(i or 24, pt(i, 24, 292), 10, MUTED)
    else:
        points = [pt(i, 24) for i in ROUTE]
        g.line(points, FG, 3)
        for start, end in zip(points, points[1:]):
            g.arrow(start, end, FG)
        for step, (i, value) in enumerate(zip(ROUTE[:-1], REFERENCE['epsilon'][:-1]), 1):
            g.circle(pt(i, 24), 4, FG, fill=FG)
            g.text(value, pt(i, 24, 270), 24)
            g.text(f'{step:02}', pt(i, 24, 292), 11, MUTED)
    repeats = view['repeat_count']
    g.text(f"M{d['modulus']} / P{d['pisano_period']}", (400, 32), 12, MUTED)
    g.text(title, (400, 690), 24)
    g.text('24-position cycle · original start preserved', (400, 723), 14, MUTED)
    g.text('One cycle' if repeats == 1 else f'{repeats} repeated cycles · overlapping view', (400, 752), 14, MUTED)
    g.save(kind + '-m9')


def sync(data_dir):
    dataset = json.loads((data_dir / 'dataset.json').read_text())
    family = audit()
    dataset['m9_sequence_family_audit'] = family
    (data_dir / 'm9-sequence-family-audit.json').write_text(json.dumps(family, ensure_ascii=False, indent=2) + '\n')
    updated = []
    for d in dataset['records']:
        views = projections(d, family)
        if not views:
            continue
        standalone = data_dir / f"m{d['modulus']}.json"
        native = json.loads(standalone.read_text())
        native['m9_sequence_views'] = d['m9_sequence_views'] = views
        for kind in ('alpha', 'gamma', 'epsilon'):
            draw(d, kind, views[kind], data_dir)
        standalone.write_text(json.dumps(native, ensure_ascii=False, indent=2) + '\n')
        updated.append(d['modulus'])
    (data_dir / 'dataset.json').write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + '\n')
    return {'updated_moduli': updated, 'verified_family_members': len(family['records']),
            'all_four_sequences_match_in_m9_cycles': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    print(json.dumps(sync(parser.parse_args().data_dir), indent=2))
