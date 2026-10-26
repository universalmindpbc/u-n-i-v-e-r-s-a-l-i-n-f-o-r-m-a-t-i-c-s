"""Verify the CSV-only export reconstructs every Atlas record without a JSON reader."""
import csv, io, json, sys, tempfile, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'research/informatics/pilot'))
from csv_records import read_record, write_record, BASE_URL
DATA = ROOT/'Informatics/pilot-v1'
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)/'edge.csv'
    value = {'empty':'','null':None,'empty-list':[],'empty-object':{},'types':[True,False,None,0,1.25], 'a/b~c':['x;y','a"b','line\nbreak',''], 'bools':[True,False], 'nulls':[None,None]}
    write_record(path,value); assert read_record(path)==value
with (DATA/'m1-m1001.csv').open(encoding='utf-8-sig',newline='') as f:
    summary={int(row['modulus']):row for row in csv.DictReader(f)}
assert list(summary)==list(range(1,1002))
with (DATA/'m1-m1001-images.csv').open(encoding='utf-8-sig',newline='') as f: images=list(csv.DictReader(f))
assert len(images)==8938 and len({r['url'] for r in images})==8938
with zipfile.ZipFile(DATA/'m1-m1001-csv.zip') as z:
    assert not any(n.endswith('.json') for n in z.namelist())
    for m,row in summary.items():
        d=json.loads((DATA/f'm{m}.json').read_text()); file=DATA/row['record_csv_file']
        assert read_record(file)==d,m
        assert z.read(row['record_csv_file'])==file.read_bytes(),m
        for view in d['views']:
            prefix=view['id']+'_native' if view['id'] in ('alpha','beta','gamma','epsilon') else 'combined'
            for fmt in ('png','svg'): assert row[f'{prefix}_{fmt}_url']==BASE_URL+view[fmt]
        for kind in ('alpha','beta','gamma','epsilon'):
            for fmt in ('png','svg'):
                value=row[f'{kind}_reduced_m9_{fmt}_url']
                assert value==(BASE_URL+d['m9_sequence_views'][kind][f'reduced_{fmt}'] if m%9==0 else '')
        assert row['json']==BASE_URL+f'pilot-v1/m{m}.json'
    for f in (DATA/'csv-shared').glob('*.csv'):
        assert read_record(f)==json.loads((DATA/(f.stem+'.json')).read_text())
        assert z.read(str(f.relative_to(DATA)))==f.read_bytes()
    for file in ('m1-m1001.csv','m1-m1001-images.csv','research-publications.csv'):
        assert z.read(file)==(DATA/file).read_bytes()
    assert 'source/csv_records.py' in z.namelist() and 'source/atlas-geometry.js' in z.namelist()
with zipfile.ZipFile(DATA/'m1-m1001-data.zip') as z:
    assert z.read('pilot-v1/m1-m1001.csv')==(DATA/'m1-m1001.csv').read_bytes()
    assert 'source/csv_records.py' in z.namelist()
print(json.dumps({'passed':True,'records_reconstructed_exactly':1001,'json_files_needed_by_csv_bundle':0,'native_and_m9_image_columns_verified':True,'distinct_image_urls':len(images),'shared_tables_verified':10,'csv_and_json_bundle_overviews_identical':True},indent=2))
