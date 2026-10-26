"""Publish the saved Gemini reviews without altering experiment history."""
import json
import hashlib
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
here = Path(__file__).parent
sys.path.insert(0, str(root / 'research/informatics/scientific-swarm'))
from swarm import checked_review
from publication import write_csv

design_input = json.loads((here / 'design-input.json').read_text())
final_input = json.loads((here / 'review-input.json').read_text())
design = checked_review(json.loads((root / 'research/informatics/scientific-swarm/proof/design-review.json').read_text()), design_input)
final = checked_review(json.loads((here / 'review-output.json').read_text()), final_input)
paths = ['experiments/oxtr-agent-follow-through-pilot/r1/experiment.json',
         'experiments/oxtr-qrc-01-qualification/r1/experiment.json']
files = [{'path': p, 'sha256': hashlib.sha256((root / 'Informatics/quantum' / p).read_bytes()).hexdigest()} for p in paths]
assert final_input['archive_sha256'] == files[0]['sha256']
record = {
    'schema': 'universal-informatics/scientific-review-display/v1',
    'study_id': 'oxtr-qrc-01',
    'review_status': 'retrospective_model_review_complete',
    'plain_language_finding': 'Gemini accepted Astra’s report: this pilot found no clear improvement in OXTR binding predictions from Fibonacci spacing and Phi Ratio. A larger prediction test is still needed for the Alpha, Beta, Gamma and Epsilon quantum responses.',
    'scope': 'Scientific interpretation review by a second model. This is not independent experimental reproduction, a biological validation or a statistical confidence score.',
    'study_files': files,
    'reviews': {'design': design, 'final': final},
    'review_inputs': {'design': design_input, 'final': final_input},
    'provenance': {
        'design': 'Retrospective review of the saved design and proposed next test. Its record_sha256 identifies a separate cloud QA execution of the same pilot; the source URL in that original input is a reference link, not the file matching that hash.',
        'final': 'Review of the exact published genomic report identified by archive_sha256. The earlier two-window qualification is supporting context, not a binding-prediction result.',
        'timestamps': 'Application-recorded UTC completion times; no independent timestamp certification.',
        'historical_records': 'Original study timestamps and deferred-review fields remain unchanged. This later review is an additional record.'
    }
}
out = root / 'Informatics/quantum/reviews/oxtr-study-01/r1'
out.mkdir(parents=True, exist_ok=True)
(out / 'review.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
write_csv(out / 'review.csv', record)
print(json.dumps({'passed': True, 'study_files': files, 'reviewed_at': final['reviewed_at']}))
