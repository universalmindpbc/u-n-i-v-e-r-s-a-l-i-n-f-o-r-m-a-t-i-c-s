"""Review the exact public report within the existing shared test allowance."""
import sys,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'research/informatics/scientific-swarm'))
from providers import GeminiReviewer,ReviewLedger,INSTRUCTIONS,SCHEMA
from swarm import checked_review
out=Path(__file__).parent
p=root/'Informatics/quantum/experiments/oxtr-agent-follow-through-pilot/r1/experiment.json'
x=json.loads(p.read_text());a=x['measurements']['classical'];q=x['measurements']['local_cudaq']
material={'scope':'Review the interpretation of this exact published OXTR report against the supplied method, metrics and recorded program checks. This is retrospective model review, not independent reproduction. Use two or three concise checks. Return fewer than 350 words.',
 'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'question':x['question'],
 'cohort':x['methods']['cohort'], 'analysis':{k:v for k,v in a.items() if k not in ['predictions','models']},
 'models':{k:{kk:vv for kk,vv in v.items() if kk!='validation_candidates'}for k,v in a['models'].items()},
 'quantum':{'prior':x['measurements']['reused_gpu_aquila']['meaning'], 'local_scope':q['scope'],
 'cases':[{'sample':c['sample'],'representation':c['representation'],'reported_CTCF_peak':c['reported_CTCF_peak'],'max_export_observable_error':c['max_export_observable_error'],'max_implementation_probability_error':c['max_implementation_probability_error']}for c in q['cases']],
 'compiled_artifact_hash':q['compiler_sha256'],'template_hash':q['template_sha256']},
 'report':x['interpretation'],'execution':x['execution'],
 'historical_note':'The report records Gemini as deferred at original completion. This new review is a separate later record and leaves the historical text intact. Eight labelled windows overlap five peak regions.'}
prompt=json.dumps({'phase':'final','material':material},ensure_ascii=False,allow_nan=False)
reserve=(len((INSTRUCTIONS+prompt+json.dumps(SCHEMA)).encode())+1024)*4+768*20
print(json.dumps({'reserved_USD':reserve/1e6,'material_bytes':len(prompt.encode())}))
if '--run' not in sys.argv:raise SystemExit()
ledger=ReviewLedger('/private/tmp/astra-gemini-secrets/review-usage.sqlite')
reviewer=GeminiReviewer(Path('/private/tmp/astra-gemini-secrets/api-key').read_text().strip(),ledger,max_output_tokens=768)
record=checked_review(reviewer.review('published-oxtr-archive-20261026','final',material),material)
(out/'review-input.json').write_text(json.dumps(material,indent=2)+'\n')
(out/'review-output.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'decision':record['decision'],'model':record['model'],'recorded':record['reviewed_at'],'usage':record['usage']}))
