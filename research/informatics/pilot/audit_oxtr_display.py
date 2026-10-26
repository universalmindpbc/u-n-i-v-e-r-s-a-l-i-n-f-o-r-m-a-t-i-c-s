"""Audit the plotted source selection and density without changing observations.

Run from any directory. Counts describe pooled records, not confirmed binding sites.
"""
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter
from csv_records import write_record

ROOT = Path(__file__).resolve().parents[3]
SUP = ROOT / 'Informatics/pilot-v1/oxtr-tfbs-v1'
EXP = SUP / 'experiment-v1'


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def audit():
    manifest = read(EXP / 'manifest.json')
    reference = manifest['reference']
    length = reference['length_bp']
    originals = {
        'studies': {**{r['record_id']: r for r in read(SUP / 'sequence-mapping.json')},
                    **{r['id']: r for r in read(SUP / 'study-additions-v1/braintf-peaks.json')}},
        'encode': {r['id']: r for r in read(SUP / 'encode-clusters.json')},
        'jaspar': {r['id']: r for r in read(SUP / 'jaspar-predictions.json.gz') if r['score'] >= 400},
    }
    result = {'version': 'oxtr-display-audit-2026-10-21-v2', 'reference': reference,
              'comparison': 'Same fixed gene-body-plus-flanks window for all sources; no fitted bounds or rotation.',
              'bin_definition': 'Historical 72-section summary retained for comparison and zoom range selection; displayed marks no longer aggregate neighbouring bases.',
              'position_count_definition': 'One count per identical anchor_offset_bp. Counts are evidence records, not distinct confirmed binding sites.',
              'coverage_definition': 'Union of clipped source intervals, counting overlapping bases once. This is record coverage, not a map of confirmed OXTR regulation.',
              'sources': {}}
    for source, spec in manifest['files'].items():
        path = EXP / spec['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']
        rows = read(path)
        assert {r['id'] for r in rows} == set(originals[source])
        for row in rows:
            original = originals[source][row['id']]
            for key in ['start_0based', 'end_exclusive', 'chromosome']:
                assert row[key] == original[key], (source, row['id'], key)
            if source == 'jaspar':
                for key in ['factor', 'matrix_id', 'score', 'strand']:
                    assert row[key] == original[key]
            if source == 'encode':
                assert row['factor'] == original['target']
        bins = [0] * 72
        for row in rows:
            bins[min(71, int(row['mapping']['full_circle_turns'] * 72))] += 1
        intervals = sorted((r['mapping']['offset_start'], r['mapping']['offset_end_exclusive']) for r in rows)
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        coverage = sum(b - a for a, b in merged)
        result['sources'][source] = {
            'records': len(rows), 'coordinates_match_archived_sources': True,
            'file_sha256': spec['sha256'],
            'factors_or_reported_labels': len({r['factor'] for r in rows}),
            'motif_models': len({r['matrix_id'] for r in rows if r.get('matrix_id')}),
            'cell_types': len({c for r in rows for c in r.get('cell_types', [])}),
            'experiments': len({e['experiment'] for r in rows for e in r.get('experiments', [])}),
            'unique_exact_intervals': len(set(intervals)),
            'distinct_plotted_centres': len(Counter(r['mapping']['anchor_offset_bp'] for r in rows)),
            'maximum_observations_at_one_centre': max(Counter(r['mapping']['anchor_offset_bp'] for r in rows).values()),
            'bin_counts': bins, 'empty_bins': bins.count(0),
            'interval_union_bp': coverage, 'uncovered_bp': length - coverage,
            'interval_union_percent': round(100 * coverage / length, 3),
        }
    assert result['sources']['encode']['empty_bins'] == 41
    assert result['sources']['jaspar']['motif_models'] == 872
    (EXP / 'display-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    write_record(EXP / 'display-audit-fields.csv', result)
    print(json.dumps(result))


if __name__ == '__main__':
    audit()
