"""Resolve public bibliographic records linked by the authorized Daytab review."""
import json, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path('/private/tmp/um-daytab-20261026')
mapping = json.loads((HERE / 'daytab-source-map.json').read_text())
metadata = json.loads((CACHE / 'primary-metadata.json').read_text())
ids = {x['pmid'] for group in mapping.values() for x in group}
missing = ids - set(metadata)
if missing:
    query = 'SRC:MED AND (' + ' OR '.join('EXT_ID:' + x for x in sorted(missing)) + ')'
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query':query,'format':'json','resultType':'core','pageSize':1000})
    with urllib.request.urlopen(url, timeout=60) as response:
        result = json.load(response)
    (CACHE / 'daytab-additional-primary.json').write_text(json.dumps(result))
    for record in result['resultList']['result']:
        metadata[record['id']] = {'metadata':record,'queries':['Daytab direct source link']}
    (CACHE / 'primary-metadata.json').write_text(json.dumps(metadata))
print(json.dumps({'resolvedDaytabPMIDs':len(ids),'newPrimaryRecords':len(missing),'unresolved':sorted(ids-set(metadata))}))
