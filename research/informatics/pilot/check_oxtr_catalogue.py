"""Cross-format and provenance validation for the OXTR supplement."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
from csv_records import read_record
ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'Informatics/pilot-v1';SUP=DATA/'oxtr-tfbs-v1'
def read(name):return json.loads((SUP/name).read_text())
def table(name):
    with (SUP/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
c=read('catalogue.json');e=read('literature-evidence.json')
assert read_record(SUP/'catalogue-fields.csv')==c
flat=table('literature-evidence.csv');assert len(flat)==len(e)==45
for row,r in zip(flat,e):
    for k,v in r.items():assert row[k]==('' if v is None else str(v)),(r['id'],k)
assert len(c['publications'])==26 and len({p['doi'] for p in c['publications']})==26
assert len([p for p in c['publications'] if p['existing_register_id']])==1
assert len(json.loads((DATA/'research-register.json').read_text())['publications'])==14
hits=json.loads(gzip.decompress((SUP/'jaspar-predictions.json.gz').read_bytes()))
flat=table('jaspar-predictions.csv');assert len(flat)==len(hits)==217857
for row,r in zip(flat,hits):
    for k,v in r.items():assert row[k]==str(v),(r['id'],k)
    assert len(r['sequence_oriented'])==r['end_exclusive']-r['start_0based']
assert len(table('jaspar-default-threshold.csv'))==sum(r['score']>=400 for r in hits)==9164
models={p['matrix_id']:p for p in read('jaspar-pfm-templates.json')}
assert len(models)==1019 and {r['matrix_id'] for r in hits}<=set(models)
for row in table('jaspar-pfm-templates.csv'):
    assert int(row['count'])==models[row['matrix_id']]['counts'][row['base']][int(row['position_1based'])-1]
clusters=read('encode-clusters.json');provenance=read('encode-provenance.json')
assert len(clusters)==255 and len({r['target'] for r in clusters})==133
for r in clusters:assert sum(p['cluster_id']==r['id'] for p in provenance)==r['source_count']
assert len(provenance)==751
assert len(read('oreganno-regions.json'))==37
assert ''.join(r['base_plus_strand'] for r in table('reference-dna.csv'))==read('reference-dna.json')['dna_plus_strand']
for r in table('file-manifest.csv'):
    p=SUP/r['file'];assert p.stat().st_size==int(r['bytes']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
manifest=json.loads((DATA/'dataset.json').read_text())
for r in manifest['records']:
    p=ROOT/'Informatics'/r['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['id']
assert len(manifest['records'])==1001
for name in ['dataset','index','m1-m1001']:assert read_record(DATA/'csv-shared'/(name+'.csv'))==json.loads((DATA/(name+'.json')).read_text())
for name in ['oxtr-tfbs-json.zip','oxtr-tfbs-csv.zip']:
    with zipfile.ZipFile(SUP/name) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        if '-csv' in name:assert not any(n.endswith(('.json','.json.gz')) for n in z.namelist())
for name,prefix,files in [('m1-m1001-data.zip','pilot-v1/',['catalogue.json','jaspar-predictions.json.gz','encode-provenance.json']),('m1-m1001-csv.zip','',['catalogue-fields.csv','jaspar-predictions.csv','encode-provenance.csv'])]:
    with zipfile.ZipFile(DATA/name) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for file in files:assert z.read(prefix+'oxtr-tfbs-v1/'+file)==(SUP/file).read_bytes()
        if '-csv' in name:assert not any(n.endswith(('.json','.json.gz')) for n in z.namelist())
guide=read('binding-outcomes.json')
assert c['binding_interpretation']==guide==read_record(SUP/'binding-outcomes-fields.csv')
outcomes=read('mapped-binding-outcomes.json');flat_outcomes=table('mapped-binding-outcomes.csv')
assert len(outcomes)==len(flat_outcomes)==9547
assert sum(r['status']=='binding_detected' for r in outcomes)==2
assert sum(r['status']=='regional_association_detected' for r in outcomes)==381
assert sum(r['status']=='prediction_only' for r in outcomes)==9164
assert not any(r['status']=='binding_not_detected' for r in outcomes)
for a,b in zip(outcomes,flat_outcomes):assert a==b
print(json.dumps({'binding_outcomes_verified':9547,'passed':True,'literature_rows_roundtripped':45,'predictions_roundtripped':217857,'pfms_verified':1019,'encode_provenance_links':751,'mathematical_records_unchanged':1001,'existing_register_publications':14,'complete_archives_verified':4}))
