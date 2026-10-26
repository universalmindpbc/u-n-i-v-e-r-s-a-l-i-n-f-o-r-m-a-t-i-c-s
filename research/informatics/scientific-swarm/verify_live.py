"""Bounded review of saved public work; no scientific job submission."""
import argparse
import hashlib
import json
from pathlib import Path
from providers import GeminiReviewer, ReviewLedger
from swarm import checked_review

p = argparse.ArgumentParser()
p.add_argument('--phase', choices=['design', 'final'], required=True)
p.add_argument('--key-file', required=True)
p.add_argument('--ledger', required=True)
p.add_argument('--output', required=True)
p.add_argument('--attempt', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[3]
source = root / 'research/informatics/quantum-unified-20261026/task-result.json'
raw = source.read_bytes(); recorded = json.loads(raw)
plan = {
    'scope': 'Retrospective audit of completed public OXTR work; no new jobs.',
    'question': recorded['question'],
    'hypotheses': ['Fibonacci and Phi Ratio may improve occupancy prediction',
                   'M9 P24 Alpha/Beta/Gamma/Epsilon responses are candidate future predictive features'],
    'controls': ['Ordinary DNA inputs', 'Shuffled input control', 'Classical motif baseline'],
    'current_evidence_boundary': 'The two-window 16-run quantum batch qualifies measurements. It does not evaluate biological prediction or coherence lifetime. The later 152-window classical OXTR test is a separate pilot.',
    'next_test': 'Prespecify more labelled genomic windows, partition adjacent windows together, freeze all representations and model selection, then compare held-out prediction against ordinary and shuffled controls.',
    'authority': {'new_Aquila_jobs': False, 'new_paid_GPU_jobs': False},
    'success': 'Improvement on independent biological test regions with uncertainty and matched baselines; null findings remain reportable.'}
material = {'question': recorded['question'], 'plan': plan,
            'evidence': {'record_sha256': hashlib.sha256(raw).hexdigest(),
                         'source': 'https://universalmind.tv/informatics/quantum/experiments/oxtr-agent-follow-through-pilot/r1/experiment.json'}}
if a.phase == 'final':
    # Send the full evidence categories while omitting large raw numeric arrays.
    # Their complete source remains content-addressed in this packet.
    analysis = {k:v for k,v in recorded['analysis'].items() if k != 'predictions'}
    prepared = recorded['prepared_quantum']
    case_keys = ['id', 'sample', 'reported_CTCF_peak', 'representation', 'execution',
                 'max_export_observable_error', 'max_implementation_probability_error',
                 'qasm_sha256', 'ahs_sha256']
    positive = [r for r in recorded['analysis']['predictions'] if r['reported_CTCF_peak']]
    material.update({'execution': {
        'cohort': recorded['cohort'], 'analysis': analysis,
        'positive_test_windows': positive,
        'prior_quantum': recorded['prior_quantum'], 'spend': recorded['spend'],
        'prepared_quantum': {k:v for k,v in prepared.items() if k not in ['cases', 'physical_model']},
        'prepared_cases': [{k:r[k] for k in case_keys} for r in prepared['cases']]},
        'report': recorded['narrative'],
        'report_context': {'completed_at': recorded['completed_at'],
            'review_at_original_completion': recorded['review'],
            'current_review': 'Retrospective review; preserve the original historical report status. New review is recorded separately.',
            'sampling_definition': 'Eight labelled 256-base windows overlap five reported peak regions. Peak regions and DNA windows are different counting units.'}})
ledger = ReviewLedger(a.ledger)
reviewer = GeminiReviewer(Path(a.key_file).read_text().strip(), ledger, max_output_tokens=4096)
output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
(output / (a.phase + '-input.json')).write_text(json.dumps(material, indent=2) + '\n')
try:
    review = checked_review(reviewer.review('connection-' + a.attempt, a.phase, material), material)
    (output / (a.phase + '-review.json')).write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({'phase': a.phase, 'connected': True, 'decision': review['decision'],
                      'model': review['model'], 'usage': review['usage']}))
except Exception as e:
    print(json.dumps({'connected': False, 'type': type(e).__name__, 'message': str(e)[:200]}))
    raise SystemExit(1)
