"""Validate source retention, coordinate mapping and conservative overlap joins."""
import csv
import gzip
import hashlib
import json
from collections import Counter
from normalize_oxtr_sites import SUP, exact_matches, overlap

def read(n):
    b=(SUP/n).read_bytes()
    return json.loads(gzip.decompress(b) if n.endswith('.gz') else b)

project=read('human-site-project.json')
for n,h in project['input_sha256'].items():
    assert hashlib.sha256((SUP/n).read_bytes()).hexdigest()==h,n
observations=read('normalized-observations.json.gz')
intervals=read('reference-intervals.json.gz')
assert len(observations)==218320
assert Counter(r['source_collection'] for r in observations)==dict(literature=45,encode=255,oreganno=37,jaspar=217857,braintf=126)
assert len({r['observation_id'] for r in observations})==len(observations)
lookup={r['observation_id']:r for r in observations}
counts=Counter(r['reference_interval_id'] for r in observations if r['reference_interval_id'])
assert {r['reference_interval_id']:r['observation_count'] for r in intervals}==counts
assert len({(r['species'],r['assembly'],r['chromosome'],r['start_0based'],r['end_exclusive']) for r in intervals})==len(intervals)
with (SUP/'normalized-observations.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=csv.DictReader(f)
    count=0
    for r,o in zip(rows,observations,strict=True):
        count+=1
        for k,v in o.items():
            if isinstance(v,(list,dict)):assert json.loads(r[k])==v,(o['observation_id'],k)
            else:assert r[k]==('' if v is None else str(v)),(o['observation_id'],k)
    assert count==len(observations)
dna=read('reference-dna.json')
expected={'TFBS012':(8770729,8770746,'-'),'TFBS013':(8770405,8770425,'-')}
for m in read('sequence-mapping.json'):
    assert m['matches_in_window']==1
    assert (m['start_0based'],m['end_exclusive'],m['strand'])==expected[m['record_id']]
    s=dna['dna_plus_strand'][m['start_0based']-dna['start_0based']:m['end_exclusive']-dna['start_0based']]
    assert s.translate(str.maketrans('ACGT','TGCA'))[::-1]==m['sequence']
assert len(exact_matches('AAA','AAAA'))==2,'Overlapping matches cannot be lost'
assert exact_matches('AGC','GCT')==[(0,3,'-')]
a=dict(species='Homo sapiens',assembly='GRCh38',chromosome='chr3',start_0based=10,end_exclusive=20)
assert overlap(a,{**a,'start_0based':20,'end_exclusive':30})==0,'Touching half-open intervals do not overlap'
assert overlap(a,{**a,'species':'Mus musculus'})==0
assert overlap(a,{**a,'assembly':'GRCh37'})==0
for r in read('literature-overlaps.json'):
    assert overlap(lookup[r['literature_observation_id']],lookup[r['other_observation_id']])==r['overlap_bp']>0
assert lookup['literature:TFBS043']['reference_interval_id']==lookup['oreganno:OREG1492306']['reference_interval_id']
assert lookup['literature:TFBS010']['reference_interval_id'] is None,'Do not turn relative positions into guessed genomic intervals'
assert lookup['literature:TFBS003']['reference_interval_id'] is None,'Reported group of sites remains unresolved'
assert 'laboratory_tested' in lookup['literature:TFBS029']['evidence_categories']
assert 'binding_evidence' not in lookup['literature:TFBS029']['evidence_categories'],'Negative binding outcome must remain negative'
assert 'functional_evidence' not in lookup['literature:TFBS023']['evidence_categories']
assert project['summary']['unique_confirmed_human_site_total'] is None
assert project['original_fifteen']['individual_coordinates_recovered']==0
print(json.dumps({'passed':True,'source_imports_verified':6,'observations_roundtripped':count,'reference_intervals':len(intervals),'published_sequence_mappings':2,'overlap_pairs':len(read('literature-overlaps.json'))}))
