import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
BASE='https://staging.universalmind.tv/informatics/'
def read(path):
    req=urllib.request.Request(BASE+path,headers={'User-Agent':'UniversalInformatics-StagingVerification/1.0'})
    return urllib.request.urlopen(req,timeout=90)

manifest=json.loads((ROOT/'Informatics/exports/manifest.json').read_text())
with read('exports/manifest.json') as r:
    assert json.load(r)==manifest

def verify(part):
    h=hashlib.sha256();count=0
    with read('exports/'+part['file']) as r:
        while chunk:=r.read(1024*1024):
            h.update(chunk);count+=len(chunk)
    assert count==part['bytes'],part['file']
    assert h.hexdigest()==part['sha256'],part['file']
    return {'file':part['file'],'bytes':count,'sha256':h.hexdigest(),'matched':True}

parts=[p for download in manifest['downloads'].values() for p in download['parts']]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    checks=list(pool.map(verify,parts))
report={'passed':True,'datasetVersion':manifest['dataset_version'],'sourceCount':len(manifest['sources']),'parts':checks}
(OUT/'exports-live.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'parts':len(checks),'sourceCount':len(manifest['sources'])}))
