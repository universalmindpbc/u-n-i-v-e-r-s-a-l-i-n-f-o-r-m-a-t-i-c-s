import concurrent.futures,hashlib,json,re,subprocess,urllib.request,urllib.error
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent
paths=subprocess.check_output(['git','diff','--name-only','c247cb476509aa5d51db40b419d245a6b79dcd61','93d35407','--','Informatics'],cwd=ROOT,text=True).splitlines()
paths=[p for p in paths if not p.endswith(('.bin','.php'))]
def check(path):
    relative=path.replace('Informatics/','informatics/',1)
    query='?v=18' if path=='Informatics/library.js' else '?v=geometry-20261026' if '/library-data/' in path else '?v=unified-2' if path in ['Informatics/quantum/biology.css','Informatics/quantum/biology.js'] else '?verification=93d35407'
    url='https://staging.universalmind.tv/'+relative+query
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'UniversalInformatics-StagingVerification/1.0'}),timeout=60) as response:
        body=response.read()
    raw=hashlib.sha256(body).hexdigest()
    if path.endswith('.html'):
        body=re.sub(rb'<script type="module" src="https://static\.cloudflareinsights\.com/beacon\.min\.js/[^<]+</script>\n',b'',body)
    actual=hashlib.sha256(body).hexdigest()
    expected=hashlib.sha256((ROOT/path).read_bytes()).hexdigest();assert actual==expected,path
    return {'path':path,'url':url,'sha256':actual,'rawSHA256':raw,'normalization':'Existing Cloudflare analytics script only' if raw!=actual else None,'matched':True}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:assets=list(pool.map(check,paths))
base='https://informatics-universal-one.universalmind.workers.dev'
def request(url):return urllib.request.Request(url,headers={'User-Agent':'UniversalInformatics-StagingVerification/1.0'})
with urllib.request.urlopen(request(base+'/api/v1/inventory'),timeout=60) as response:production=json.load(response)
assert production['counts']['publications']==811
try:urllib.request.urlopen(request(base+'/api/v1/publications/vision-hexadirectional-2018'),timeout=60)
except urllib.error.HTTPError as error:assert error.code==404
else:raise AssertionError('Staging addition appeared in production')
report={'passed':True,'sourceCommit':'93d35407091981ea5e1abfd2986212ef1ccd5575','assets':assets,'productionCounts':production['counts'],'newRecordAbsentFromProduction':True}
(OUT/'assets-live.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'assets':len(assets),'productionUnchanged':True}))
