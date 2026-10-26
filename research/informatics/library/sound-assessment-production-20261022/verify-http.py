import concurrent.futures, datetime, hashlib, json, pathlib, subprocess
root=pathlib.Path('/private/tmp/um-informatics-library-20261021')
source='a5b386988b206477cd017435295c0e291ce2f8a1'
baseline='93e74a360654d8ff3c9baa09f548d82d8284c661'
files=['library.html','library.css','library.js']
def verify(item):
    env,name=item
    url=f'https://{"staging." if env=="staging" else ""}universalmind.tv/informatics/{name}?release={source[:8]}'
    response=subprocess.run(['curl','--fail','--silent','--show-error','--location','--max-time','45',url,'--write-out','\n%{http_code}'],capture_output=True,check=True)
    body,status=response.stdout.rsplit(b'\n',1)
    expected=subprocess.check_output(['git','show',f'{source}:Informatics/{name}'],cwd=root)
    assert status==b'200' and body==expected,(url,status,'source mismatch')
    return dict(environment=env,url=url,status=int(status),sha256=hashlib.sha256(body).hexdigest(),matchesApprovedSource=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results=list(pool.map(verify,[(env,name) for env in ['production','staging'] for name in files]))
manifest={line.split('\t',1)[1]:line.split('\t',1)[0] for line in (root/'.panthur-deploy-manifest.tsv').read_text().splitlines()}
assert all(manifest['informatics/'+name]==hashlib.sha256((root/'Informatics'/name).read_bytes()).hexdigest() for name in files)
changed=subprocess.check_output(['git','diff','--name-only',baseline,source,'--','Informatics'],cwd=root,text=True).splitlines()
assert sorted(changed)==sorted('Informatics/'+n for n in files)
result=dict(checkedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),approvedSourceCommit=source,previousProductionCommit=baseline,changedPublicFiles=changed,productionManifestMatches=True,results=results,passed=True)
(root/'research/informatics/library/sound-assessment-production-20261022/http-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
