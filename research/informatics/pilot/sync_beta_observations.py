"""Add the approved M9/M18 native Beta observations to the versioned records."""
import argparse
import json
from pathlib import Path

NOTES = {
    9: 'Across M1–M1001, the native Beta sequence 4–8–7–5–1–2 occurs only at M9 P24. M9 and M18 are the only records whose complete native Beta sequence forms a doubling cycle with six distinct values and the reference chevron under the Atlas construction rules.',
    18: 'M18 P24 has the native doubling sequence 4–8–16–14–10–2. Reduction modulo 9 recovers 4–8–7–5–1–2 in its original alignment. This relationship follows the arithmetic of reduction within the M9 family.',
}


def sync(data_dir):
    dataset = json.loads((data_dir / 'dataset.json').read_text())
    for record in dataset['records']:
        m = record['modulus']
        if m not in NOTES:
            continue
        native_file = data_dir / f'm{m}.json'
        native = json.loads(native_file.read_text())
        assert native['beta'] == record['beta']
        observation = {
            'version': 'native-beta-observation-v1',
            'title': 'Native sequence comparison',
            'text': NOTES[m],
            'classification': 'Computed mathematical observation',
            'rule_version': record['geometry_summary']['rule_version'],
            'audit_modulus_range': [1, 1001],
            'audit_moduli_count': 1001,
            'native_doubling_and_reference_chevron_moduli': [9, 18],
            'native_beta_cycle': record['beta']['interval_residues'],
            'reduced_m9_beta_cycle': [v % 9 for v in record['beta']['interval_residues']],
            'original_alignment_preserved': True,
            'source': 'Two independent native-period, Beta and chevron implementations; research/informatics/pilot/audit_native_beta.py.',
        }
        if m == 9:
            observation['exact_native_beta_match_moduli'] = [9]
        for target in (native, record):
            target['native_beta_observation'] = observation
        native_file.write_text(json.dumps(native, ensure_ascii=False, indent=2) + '\n')
    (data_dir / 'dataset.json').write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    sync(args.data_dir)
    print('M9 and M18 native Beta observations synchronized.')
