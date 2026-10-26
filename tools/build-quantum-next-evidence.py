"""Create a traceable coverage check for the next OXTR prediction experiment.

Reads existing public source files. Does not fit a model or submit a job.
"""
import gzip, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
source = Path(sys.argv[1])
meta_path = source / 'source/ENCSR932ZMX.json'
peaks_path = source / 'source/ENCFF651DQL.bed.gz'
inputs_path = source / 'results/inputs.json'
meta = json.loads(meta_path.read_text())
assert meta['accession'] == 'ENCSR932ZMX' and meta['target']['label'] == 'CTCF'
assert meta['biosample_ontology']['term_name'] == 'GM23338'
file = next(f for f in meta['files'] if isinstance(f,dict) and f.get('accession') == 'ENCFF651DQL')
assert file['assembly'] == 'GRCh38' and file['output_type'] == 'conservative IDR thresholded peaks'
peaks = []
with gzip.open(peaks_path, 'rt') as stream:
    for row in stream:
        fields = row.rstrip().split('\t')
        if fields[0] == 'chr3' and int(fields[1]) < 8779628 and int(fields[2]) > 8740380:
            peaks.append({'start': int(fields[1]), 'end': int(fields[2])})
peaks.sort(key=lambda p: p['start'])
samples = []
for row in json.loads(inputs_path.read_text())['rows']:
    overlaps = [p for p in peaks if p['start'] < row['end'] and p['end'] > row['start']]
    samples.append({'id': row['id'], 'start': row['start'], 'end': row['end'], 'overlap_count': len(overlaps)})
assert len(peaks) == 5 and len(samples) == 2 and all(s['overlap_count'] == 0 for s in samples)
record = {'schema': 'universal-informatics/next-test-evidence/v1', 'checked_at': datetime.now(timezone.utc).isoformat(),
 'purpose': 'Post-run data coverage audit; not a biological prediction result. No measurements or selected inputs changed.',
 'coordinates': 'GRCh38 chr3, 0-based start, end-exclusive; UI displays start + 1 and end.',
 'locus': {'start': 8740380, 'end': 8779628},
 'assay': {'accession': meta['accession'], 'factor': 'CTCF', 'cell': 'GM23338', 'cell_description': 'induced pluripotent stem cells',
           'replicates': len(meta['replicates']), 'file': file['accession'], 'output_type': file['output_type'],
           'audit_warnings': sorted({a['category'] for a in meta.get('audit',{}).get('WARNING',[])})},
 'peaks': peaks, 'samples': samples,
 'sources': [{'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [meta_path, peaks_path, inputs_path]],
 'prediction_test_run': False, 'hardware_jobs_submitted': 0}
out = Path('Informatics/quantum/data/oxtr-next-test-evidence.json')
out.write_text(json.dumps(record, indent=2)+'\n')
print('Verified five CTCF regions; neither setup sample overlaps them. No jobs submitted.')
