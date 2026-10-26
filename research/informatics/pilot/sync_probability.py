"""Publish the selected M9 and M18 comparisons with native alignment preserved.

Run after geometry/literature generation and before export_csv.py.
M27, M36 and M45 remain unscored: null means not selected, not zero.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def publish(data_dir, calculator, modulus):
    spec = importlib.util.spec_from_file_location(f'm{modulus}_probability', calculator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audit = module.calculate(data_dir / f'm{modulus}.json')
    native = json.loads((data_dir / f'm{modulus}.json').read_text())
    result = audit['probabilities']['joint_exact_gamma_fixed_beta_phase']
    assert result['exact_fraction'] == {9: '263/51305410956800', 18: '491/2885929366320000'}[modulus]
    numerator, denominator = result['exact_fraction'].split('/')
    counts = audit['counts']
    probability = {
        'schema_version': '1.0.0',
        'id': f'm{modulus}p24-fixed-gamma-beta-shuffle-v1',
        'record_id': native['id'],
        'calculated_on': audit['date'],
        'status': 'Calculated',
        'heading': 'Probability',
        'display': audit['selected_comparison']['display'],
        'summary': audit['selected_comparison']['explanation'],
        'research_insight': {
            'status': 'Open research',
            'quote': 'We are exploring equilateral triangles, honeycomb theorem, and carbon structures in relationship to this Atlas for useful directions for computation in OXTR research.',
        },
        'method': {
            'id': 'uniform-alpha-shuffle-original-alignment-v1',
            'comparison_rule': "Preserve each record's original Gamma values, positions and Beta alignment.",
            'sampling': audit['model']['sampling'],
            'coordinate_system': audit['model']['positions'],
            'alpha_frequencies': {str(v): native['alpha']['residues'].count(v) for v in sorted(set(native['alpha']['residues']))},
            'gamma_indices': native['gamma']['alpha_indices'],
            'gamma_clock_positions': native['gamma']['clock_positions'],
            'gamma_values': native['gamma']['clockwise_values'],
            'beta_values': native['beta']['interval_residues'],
            'beta_start_interval_clock_positions': native['beta']['start_interval_clock_positions'],
            'beta_interior_indices': [gap['interior_indices'] for gap in native['beta']['intervals']],
            'beta_rule': audit['model']['beta_rule'],
            'allow_rotated_gamma_positions': False,
            'allow_reordered_gamma_values': False,
            'allow_rotated_beta_cycle': False,
            'allow_reversed_beta_cycle': False,
        },
        # Counts are strings so JavaScript consumers retain exact integers.
        'counts': {
            'all_arrangements': str(counts['all_distinct_alpha_arrangements']),
            'arrangements_given_gamma': str(counts['arrangements_given_exact_gamma']),
            'matching_arrangements': str(counts['exact_gamma_and_fixed_beta_phase']),
            'qualifying_triple_combinations': counts['qualifying_six_triple_combinations'],
        },
        'result': {
            'numerator': numerator, 'denominator': denominator,
            'fraction': result['exact_fraction'],
            'value': result['decimal'], 'one_in_approx': result['one_in_approx'],
        },
        'calculation': {
            'steps': [
                'Shuffle the same 24 Alpha values, keeping every value and its frequency.',
                f"Place Gamma values {', '.join(map(str, native['gamma']['clockwise_values']))} at clock positions 24, 4, 8, 12, 16, 20.",
                f'Sum the three Alpha values between each pair of Gamma nodes, excluding the endpoints, and reduce each sum modulo {modulus}.',
                f"Match Beta to {', '.join(map(str, native['beta']['interval_residues']))}, starting with the interval from clock position 24 to 4.",
            ],
            'formula': f"{counts['exact_gamma_and_fixed_beta_phase']:,} / {counts['all_distinct_alpha_arrangements']:,} = {int(numerator):,} / {int(denominator):,}",
            'counting_note': f"For the six Beta intervals, {counts['qualifying_six_triple_combinations']} combinations of three-value groups use exactly the available values. Counting the possible orders within each group gives {counts['exact_gamma_and_fixed_beta_phase']:,} matching arrangements.",
            'verification_note': 'Three exact counting methods agree: remaining-value enumeration, complete three-value-group enumeration, and a separate count treating each card individually.',
        },
        'verification': {
            'calculation_version': audit['calculation_version'],
            'mathematical_input_sha256': audit['mathematical_input_sha256'],
            'checks': audit['verification'],
        },
        'interpretation': {
            'type': 'exact_configuration_shuffle_probability',
            'selection_adjustment_applied': False,
            'statistical_significance_status': native['research']['statistical_significance_status'],
            'biological_significance_status': native['research']['biological_significance_status'],
        },
        'files': {
            'data': f'm{modulus}-probability.json',
            'code': f'm{modulus}-probability.py',
        },
    }
    dataset = json.loads((data_dir / 'dataset.json').read_text())
    matching = [r for r in dataset['records'] if r['id'] == native['id']]
    assert len(matching) == 1
    aggregate = matching[0]
    for key in ('alpha', 'gamma', 'beta'):
        assert aggregate[key] == native[key]
    aggregate['probability'] = probability
    native['probability'] = probability
    write_json(data_dir / 'dataset.json', dataset)
    write_json(data_dir / f'm{modulus}.json', native)
    write_json(data_dir / f'm{modulus}-probability.json', probability)
    (data_dir / f'm{modulus}-probability.py').write_text(calculator.read_text())
    return probability


SCOPE = {
    'selected_record_ids': ['fibonacci-m9-p24', 'fibonacci-m18-p24'],
    'not_selected_record_ids': ['fibonacci-m27-p72', 'fibonacci-m36-p24', 'fibonacci-m45-p120'],
    'basis': 'Current M9 and M18 geometry comparison, selected by the project lead on 2026-10-18.',
    'null_meaning': 'Not selected for this comparison; no probability assigned. Null is not zero.',
    'comparison_rule': "Preserve each selected record's native Gamma values, positions and Beta alignment.",
}


def sync(data_dir, calculator):
    # Selection is explicit, not inferred from family membership or a score.
    publications = [publish(data_dir, calculator, 9),
                    publish(data_dir, calculator.with_name('m18p24_exact_shuffle.py'), 18)]
    dataset = json.loads((data_dir / 'dataset.json').read_text())
    dataset['probability_scope'] = SCOPE
    for record in dataset['records']:
        if record['id'] not in SCOPE['not_selected_record_ids']:
            continue
        native_path = data_dir / f"m{record['modulus']}.json"
        native = json.loads(native_path.read_text())
        for target in (native, record):
            target['probability'] = None
            target['probability_scope'] = {
                'status': 'Not selected',
                'comparison': 'Current M9 and M18 geometry comparison',
                'null_meaning': SCOPE['null_meaning'],
            }
        write_json(native_path, native)
    write_json(data_dir / 'dataset.json', dataset)
    return publications


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--calculator', type=Path, default=Path(__file__).resolve().parents[1] / 'probability/m9p24_exact_shuffle.py')
    args = parser.parse_args()
    for result in sync(args.data_dir, args.calculator):
        print(f"{result['record_id']}: {result['display']}; native Gamma and Beta alignment preserved.")
    print('M27, M36 and M45: null; not selected for the current comparison.')
