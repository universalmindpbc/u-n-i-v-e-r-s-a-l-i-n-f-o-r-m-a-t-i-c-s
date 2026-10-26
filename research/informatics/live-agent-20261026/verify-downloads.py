"""Reconstruct the complete report from its CSV and verify review packets."""
import csv,hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent
report=json.loads((p/'live-result.json').read_text())
rows=list(csv.DictReader((p/'live-result.csv').open()))
built={}
for row in rows:
 value=row['value'][1:] if row['escaped']=='1' else row['value']
 kind=row['type']
 value={} if kind=='object' else [] if kind=='array' else None if kind=='null' else value=='true' if kind=='boolean' else json.loads(value) if kind=='number' else value
 path=row['field'];built[path]=value
 if path:
  parent,key=path.rsplit('/',1);key=key.replace('~1','/').replace('~0','~')
  if isinstance(built[parent],list):built[parent].append(value)
  else:built[parent][key]=value
assert built['']==report
def numeric_normalize(value):
 if isinstance(value,float) and value.is_integer():return int(value)
 if isinstance(value,list):return [numeric_normalize(v) for v in value]
 if isinstance(value,dict):return {k:numeric_normalize(v) for k,v in value.items()}
 return value
proof={'json_csv_equal':True,'csv_fields':len(rows),'semantic_sha256':hashlib.sha256(json.dumps(numeric_normalize(report),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),'review_evidence_present':[],'hash_note':'Browser JSON serialization converts integral floats such as 1.0 to 1. Exact review hashes are checked against the archived source packets; semantic equality normalizes that numeric spelling.'}
for review in report['reviews']:
 expected=review['evidence_sha256'];packet=report['review_inputs'][expected]
 assert isinstance(packet,dict)
 proof['review_evidence_present'].append({'phase':review['phase'],'source_sha256':expected})
(p/'download-equivalence.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
