"""Expose the M9 family's existing Other Geometry formations as summary data.

Run after geometry generation and before export_csv.py. Does not alter geometry,
probability selection, or evidence assessments.
"""
import argparse
import json
from pathlib import Path


def summarize(record):
    p = record['pisano_period']
    gamma = set(record['gamma']['alpha_indices'])
    formations = [f for f in record['formation_catalogue'] if f['shape'] == 'Hexagon']
    # Deduplicate node sets independently of the catalogue's starting labels.
    expected = {frozenset((offset + k * (p // 6)) % p for k in range(6))
                for offset in range(p)} if p % 6 == 0 else set()
    actual = {frozenset(f['alpha_indices']) for f in formations}
    assert actual == expected and len(actual) == len(formations)
    primary = [f for f in formations if set(f['alpha_indices']) <= gamma]
    assert all(f['in_primary_gamma'] == (f in primary) for f in formations)
    current = next(s for s in record['geometry_summary']['shapes']
                   if s['name'] == '2D stellated octahedron')
    assert len(primary) == current['count']
    return {
        'schema_version': '1.0.0',
        'shape': '2D stellated octahedron',
        'all_alpha_count': len(formations),
        'primary_gamma_count': len(primary),
        'native_beta_defined': record['beta'] is not None,
        'native_beta_doubling': record['beta']['native_doubling'] if record['beta'] else None,
        'formation_ids': [f['id'] for f in formations],
        'primary_gamma_formation_ids': [f['id'] for f in primary],
        'counting_rule': 'One pair of alternating equilateral triangles per distinct regular six-node set on the full Alpha circle. Each set is counted once, with its orientation retained.',
        'period_rule': 'The full Alpha count is P / 6 when P is divisible by 6, otherwise zero.',
        'gamma_rule': 'Primary Gamma counts only formations whose six vertices belong to the selected Gamma node set.',
        'source': 'formation_catalogue; each referenced Hexagon supplies the same six vertices for the paired triangles in Other Geometry.',
        'research_status': 'Descriptive geometry. Native doubling is the current starting hypothesis for mitosis; other Beta sequences remain available for future investigation.',
    }


def sync(data_dir):
    dataset = json.loads((data_dir / 'dataset.json').read_text())
    summary = []
    for record in dataset['records']:
        if record['modulus'] not in (9, 18, 27, 36, 45):
            continue
        native_path = data_dir / f"m{record['modulus']}.json"
        native = json.loads(native_path.read_text())
        assert native['formation_catalogue'] == record['formation_catalogue']
        counts = summarize(native)
        for target in (record, native):
            target['other_geometry_summary'] = counts
        native_path.write_text(json.dumps(native, ensure_ascii=False, indent=2) + '\n')
        summary.append({'modulus': record['modulus'], **counts})
    (data_dir / 'dataset.json').write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps([{'modulus': s['modulus'], 'all_alpha': s['all_alpha_count'],
                       'primary_gamma': s['primary_gamma_count']} for s in sync(args.data_dir)], indent=2))
