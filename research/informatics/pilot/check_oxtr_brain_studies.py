"""Check selection, evidence boundaries and JSON/CSV/archive parity."""
import csv
import gzip
import json
from pathlib import Path
import zipfile
from csv_records import read_record

ROOT = Path(__file__).resolve().parents[3]
SUP = ROOT / 'Informatics/pilot-v1/oxtr-tfbs-v1'
STUDY = SUP / 'study-additions-v1'
read = lambda name: json.loads((STUDY / name).read_text())
peaks = read('braintf-peaks.json')
locus = read('braintf-locus-review.json')
qc = {r['source_row']: r for r in read('braintf-qc-rows.json')}
assert len(peaks) == 126
assert len({r['factor'] for r in peaks}) == 43
assert len({r['context'] for r in peaks}) == 13
assert sum(r['promoter_overlap'] for r in peaks) == 114
assert len(locus) == 390
assert sum(r['included_in_study_browser'] for r in locus) == 126
assert len({r['id'] for r in peaks}) == 126
for p in peaks:
    r = next(r for r in locus if (r['source_file'], r['source_row_1based']) == (p['source_file'], p['source_row_1based']))
    assert r['assay_class'] == 'TF_or_chromatin_protein'
    assert r['replication_qc_statuses'] == ['pass'] and len(r['qc_rows']) == 1
    q = qc[r['qc_rows'][0]]
    assert min(q['FRiP_rep1'], q['FRiP_rep2']) == p['minimum_FRiP'] >= .01
    assert r['author_HOT_statuses'] == ['NOT'] and r['source_metadata_conflict'] is None
    assert (p['start_0based'], p['end_exclusive']) == (r['start'], r['end'])
    assert p['start_0based'] < 8779628 and p['end_exclusive'] > 8740380
    assert p['chromosome'] == 'chr3' and p['assembly'] == 'GRCh38'
    assert p['source_url'].startswith('https://raw.githubusercontent.com/aanderson54/Loupe_BrainTF/')
    assert len(p['source_sha256']) == 64
    assert p['qc_sheet'] == 'Supplementary Table 2 / ChIP-seq'
assert all(not p['factor'].startswith('H3') for p in peaks)
assert all(not (p['factor'] in ('YBX1','YBX3') and p['context']=='OL_NEUN') for p in peaks)
for name in ['braintf-peaks','braintf-locus-review','braintf-qc-rows','enhancer-regions','source-file-hashes','sources']:
    with (STUDY / (name+'.csv')).open(encoding='utf-8-sig',newline='') as f:
        table = list(csv.DictReader(f))
    data = read(name+'.json')
    assert len(table)==len(data)
    for r,p in zip(table,data):
        for k,v in p.items():
            assert (json.loads(r[k]) == v) if isinstance(v,(dict,list)) else r[k] == ('' if v is None else str(v)), (name,k)
assert read_record(STUDY/'review-fields.csv') == read('review.json')
catalogue=json.loads((SUP/'catalogue.json').read_text())
assert catalogue['study_additions'] == read('review.json')
assert [r['pmid'] for r in catalogue['publications'][-2:]] == ['38831039','39957428']
assert [r['id'] for r in catalogue['evidence'][-2:]] == ['TFBS044','TFBS045']
assert all(r['normalized_location']['start_0based'] is None for r in catalogue['evidence'][-2:])
enhancers=read('enhancer-regions.json')
assert [r['name'] for r in enhancers if r['positive_reporter_assay']] == ['OCE1','OCE5','OCE7']
assert next(r for r in enhancers if r['name']=='OCE7')['reported_coordinates_hg38'] == 'chr3:8757366-8759184'
assert catalogue['coordinate_experiment']['counts']['studies']==128
observations=json.loads(gzip.decompress((SUP/'normalized-observations.json.gz').read_bytes()))
assert {r['source_record_id'] for r in observations if r['source_collection']=='braintf'} == {p['id'] for p in peaks}
for file,prefix,names in [
    (SUP/'oxtr-tfbs-json.zip','',['braintf-peaks.json','review.json','enhancer-regions.json']),
    (SUP/'oxtr-tfbs-csv.zip','',['braintf-peaks.csv','review-fields.csv','enhancer-regions.csv']),
    (SUP.parent/'m1-m1001-data.zip','pilot-v1/oxtr-tfbs-v1/',['braintf-peaks.json','review.json']),
    (SUP.parent/'m1-m1001-csv.zip','oxtr-tfbs-v1/',['braintf-peaks.csv','review-fields.csv'])]:
    with zipfile.ZipFile(file) as z:
        for name in names:
            assert z.read(prefix+'study-additions-v1/'+name)==(STUDY/name).read_bytes()
print(json.dumps({'passed':True,'screened_occupancy_records':126,'protein_targets':43,'contexts':13,'promoter_overlaps':114,'review_records':390,'new_publications':2,'precise_new_TF_sites_asserted':0,'archive_parity':4}))
