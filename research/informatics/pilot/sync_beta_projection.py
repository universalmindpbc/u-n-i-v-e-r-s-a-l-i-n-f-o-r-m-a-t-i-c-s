"""Publish the native/M9 Beta view for every applicable Atlas record.

Native arithmetic and images remain unchanged. Reduced views use the existing
phase-preserving comparison; repeated traversals share one drawn chevron.
"""
import argparse
import json
from pathlib import Path
import drawing
from drawing import Drawing, pt, FG, MUTED, YELLOW
from build_pilot import minimal, period, beta_for

FAMILY_TEXT = 'All 111 multiples of nine in M1–M1001 (11.1% of moduli) share this reduced Beta cycle through exact modular reduction.'
FAMILY_SUMMARY = {
    'audit_id': 'beta-m9-family-audit-v1', 'modulus_range': [1, 1001],
    'total_moduli_in_range': 1001, 'family_members_checked': 111,
    'matching_family_members': 111, 'coverage_fraction': '111/1001',
    'coverage_percent': 111 / 1001 * 100, 'coverage_display': '11.1%',
    'classification': 'Exact modular reduction; descriptive coverage',
    'summary': FAMILY_TEXT, 'audit_file': 'pilot-v1/beta-m9-family-audit.json',
}


def family_audit():
    rows = []
    for m in range(9, 1002, 9):
        a = period(m)
        nodes = [i for i, value in enumerate(a) if value % 3 == 0]
        beta = beta_for(a, nodes, m)
        reduced = [value % 9 for value in beta['interval_residues']]
        cycle = [4, 8, 7, 5, 1, 2]
        assert reduced == cycle * (len(reduced) // 6)
        # Independent direct sums of the reduced Alpha intervals.
        wrapped = a + a
        ends = nodes[1:] + [nodes[0] + len(a)]
        assert reduced == [sum(value % 9 for value in wrapped[start + 1:end]) % 9
                           for start, end in zip(nodes, ends)]
        rows.append({'modulus': m, 'pisano_period': len(a),
                     'native_beta_length': len(reduced), 'reduced_cycle': cycle,
                     'repeat_count': len(reduced) // 6,
                     'native_doubling': beta['native_doubling'], 'exact_match': True})
    assert len(rows) == 111
    return {**FAMILY_SUMMARY, 'reference_cycle': [4, 8, 7, 5, 1, 2],
            'identity': '(F_n mod (9k)) mod 9 = F_n mod 9',
            'scope': 'Arithmetic audit of the 111 multiples of nine through M999; the full published Atlas remains M1–M48.',
            'probability_relationship': 'The M9 random-shuffle probability is a separate conditional calculation. This audit describes exact modular identities, not independent chance events.',
            'original_alignment_preserved': True, 'records': rows}


def projection(record):
    m = record['modulus']
    comparison = record['reduction_to_m9']
    if m % 9 or not record['beta'] or not comparison['applicable']:
        return None
    native = record['beta']['interval_residues']
    reduced = [value % 9 for value in native]
    cycle = minimal(reduced)
    assert reduced == comparison['beta_residues']
    assert cycle == [4, 8, 7, 5, 1, 2]
    assert comparison['phase_preserved'] and comparison['checks']['beta']
    view = next(view for view in record['views'] if view['id'] == 'beta')
    return {
        'version': 'beta-m9-view-v1',
        'native_modulus': m, 'target_modulus': 9,
        'native_values': native, 'reduced_values': reduced,
        'display_cycle': cycle, 'repeat_count': len(reduced) // len(cycle),
        'native_circle_positions': m, 'reduced_circle_positions': 9,
        'original_alignment_preserved': True,
        'start_interval_clock_positions': record['beta']['start_interval_clock_positions'],
        'reference_cycle_match': True,
        'family_audit': FAMILY_SUMMARY,
        'operation': 'Reduce each native Beta residue modulo 9, preserving order and the original start.',
        'drawing_rule': 'Plot the six-value cycle once; the repeat count records overlapping traversals. The complete reduced sequence remains available alongside the diagram and in the data.',
        'native_svg': view['svg'], 'native_png': view['png'],
        'reduced_svg': f'pilot-v1/views/m{m}/beta-m9.svg',
        'reduced_png': f'pilot-v1/views/m{m}/beta-m9.png',
    }


def draw(record, view, data_dir):
    drawing.OUT = data_dir / f"views/m{record['modulus']}"
    drawing.OUT.mkdir(parents=True, exist_ok=True)
    cycle, repeats = view['display_cycle'], view['repeat_count']
    title = 'Beta · Reduced to M9'
    caption = 'Original start preserved · ' + ('one cycle' if repeats == 1 else f'{repeats} repeated cycles')
    g = Drawing(title, view['operation'] + ' ' + view['drawing_rule'])
    g.circle((400, 365), 238, '#34434e')
    points = [pt(value, 9) for value in cycle + cycle[:1]]
    g.line(points, YELLOW, 3)
    for value in range(9):
        g.text(value or 9, pt(value, 9, 270), 20, YELLOW if value in cycle else MUTED)
    for start, end in zip(points, points[1:]):
        g.arrow(start, end, YELLOW)
    g.circle(points[0], 5, YELLOW, fill=YELLOW)
    g.text('start', pt(cycle[0], 9, 297), 11, MUTED)
    g.text(f"M{record['modulus']} / P{record['pisano_period']}", (400, 32), 12, MUTED)
    g.text(title, (400, 690), 24)
    g.text(caption, (400, 723), 14, MUTED)
    g.text(' → '.join(map(str, cycle)), (400, 752), 16, FG)
    g.save('beta-m9')


def sync(data_dir):
    dataset = json.loads((data_dir / 'dataset.json').read_text())
    audit = family_audit()
    dataset['beta_m9_family_audit'] = audit
    (data_dir / 'beta-m9-family-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    updated = []
    for record in dataset['records']:
        view = projection(record)
        if not view:
            continue
        native_path = data_dir / f"m{record['modulus']}.json"
        native = json.loads(native_path.read_text())
        assert native['beta'] == record['beta']
        for target in (native, record):
            target['beta_m9_view'] = view
        draw(record, view, data_dir)
        native_path.write_text(json.dumps(native, ensure_ascii=False, indent=2) + '\n')
        updated.append({'modulus': record['modulus'], 'repeat_count': view['repeat_count']})
    (data_dir / 'dataset.json').write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + '\n')
    return updated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(sync(args.data_dir), indent=2))
