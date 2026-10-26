import zipfile, json, csv, io, runpy
from pathlib import Path
p=Path('Informatics/pilot-v1'); csv.field_size_limit(10000000)
zj=zipfile.ZipFile(p/'m1-m1001-data.zip');zc=zipfile.ZipFile(p/'m1-m1001-csv.zip')
parse=runpy.run_path('research/informatics/pilot/csv_records.py')['parse']
counts={'jsonRecords':0,'csvRecords':0,'jsonBytesMatched':0,'csvFieldsMatched':0}
def check(value,key,rows):
 row=rows[key]
 if row['type'].startswith('list:'):
  decoded=[parse(v,row['type'][5:]) for v in next(csv.reader(io.StringIO(row['value']),delimiter=';'))]
  assert decoded==value,(key,row)
 elif isinstance(value,dict):
  assert row['type']=='object' and int(row['value'])==len(value)
  for k,v in value.items():check(v,key+'/'+k.replace('~','~0').replace('/','~1'),rows)
 elif isinstance(value,list):
  assert row['type']=='array' and int(row['value'])==len(value)
  for i,v in enumerate(value):check(v,key+'/'+str(i),rows)
 else:
  v=parse(row['value'],row['type'])
  assert v==value,(key,v,value)
 counts['csvFieldsMatched']+=1
for m in range(1,1002):
 name=f'm{m}.json';raw=(p/name).read_bytes();assert zj.read('pilot-v1/'+name)==raw;counts['jsonRecords']+=1;counts['jsonBytesMatched']+=len(raw)
 rows={r['field']:r for r in csv.DictReader(io.StringIO(zc.read(f'csv-records/m{m}.csv').decode('utf-8-sig')))}
 check(json.loads(raw),'',rows);counts['csvRecords']+=1
print(json.dumps(counts))
Path('research/informatics/connectors/verification/atlas-archives.json').write_text(json.dumps(counts,indent=2)+'\n')
