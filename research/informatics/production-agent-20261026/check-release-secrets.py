import json,re,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[3]
paths=subprocess.check_output(['git','diff','--name-only','97962ec3a0c86e55153ba43eb1828cc1d4e11e69','HEAD'],cwd=root,text=True).splitlines()
paths+= [str(p.relative_to(root)) for p in (Path(__file__).resolve().parent).rglob('*') if p.is_file()]
credentials=[]
for path in ['/private/tmp/uo-secrets/worker.json','/private/tmp/astra-owner-service/server-secret.json','/private/tmp/astra-owner-service/worker-secrets.json','/private/tmp/astra-owner-service/credentials.json','/private/tmp/astra-test-session.json']:
 p=Path(path)
 if p.exists():
  for key,value in json.loads(p.read_text()).items():
   if isinstance(value,str) and len(value)>24: credentials.append(value)
patterns=[re.compile(r'AIza[0-9A-Za-z_-]{35}'),re.compile(r'AKIA[0-9A-Z]{16}'),re.compile(r'sk-(?:proj-)?[0-9A-Za-z_-]{45,}'),re.compile(r'postgres(?:ql)?://[^\s\"\']+:[^\s\"\']+@[^\s\"\']+')]
checked=0;matches=[];historical_identifiers=[]
for path in sorted(set(paths)):
 p=root/path
 if not p.is_file() or p.suffix.lower() in {'.png','.jpg','.jpeg','.gif','.webp','.pdf','.mp3','.mp4','.woff2','.bin'}:continue
 try:text=p.read_text()
 except UnicodeDecodeError:continue
 checked+=1
 if any(value in text for value in credentials):matches.append(path)
 for index,pattern in enumerate(patterns):
  for match in pattern.finditer(text):
   prefix=text[max(0,match.start()-40):match.start()].replace('\\n','')
   if index==1 and path in ['Informatics/library-data/library.json','Informatics/library-data/sound-edition.json'] and 'X-Amz-Credential=' in prefix and '20190608T031604Z' in text:
    historical_identifiers.append({'path':path,'type':'Public AWS access-key identifier in an expired 2019 publisher citation URL; no secret key'})
   else:matches.append(path)
report={'files_scanned':checked,'matching_paths':matches,'passed':not matches,'secret_values_printed':False,'non_secret_citation_identifiers':historical_identifiers}
(Path(__file__).resolve().parent/'secret-scan.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report));assert not matches
