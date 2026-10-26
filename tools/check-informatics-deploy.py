"""Exercise existing FTP uploader ordering and manifest failure handling, offline."""
from pathlib import Path
import subprocess,tempfile,os
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'deploy-panthurlive.sh').read_text().split('deploy_informatics() {',1)[1].split('\nDEPLOY_MODE=',1)[0]
with tempfile.TemporaryDirectory(prefix='atlas-deploy-test-') as temp:
 root=Path(temp);(root/'Informatics/pilot-v1').mkdir(parents=True)
 for name in ['m1.json','m2.json','index.json']:(root/'Informatics/pilot-v1'/name).write_text(name)
 (root/'Informatics/index.html').write_text('entry')
 subprocess.run(['git','init','-q',temp],check=True);subprocess.run(['git','-C',temp,'add','Informatics'],check=True);subprocess.run(['git','-C',temp,'-c','user.name=Test','-c','user.email=test@invalid','commit','-qm','fixture'],check=True)
 (root/'bin').mkdir()
 stub=root/'bin/curl'
 stub.write_text('''#!/usr/bin/env python3
import sys,json,os,urllib.parse
from pathlib import Path
assert sys.argv[1:]==['--fail-early','--config','-']
config=sys.stdin.read()
assert 'parallel' not in config
for block in config.split('next\\n'):
 values={}
 for line in block.splitlines():
  if ' = ' in line:
   k,v=line.split(' = ',1)
   if v.startswith('"'):values[k]=json.loads(v)
 assert values['user']=='test:dummy'
 assert values['url'].startswith('ftp://test.invalid/')
 remote=urllib.parse.unquote(values['url'].split('test.invalid/',1)[1])
 with open(os.environ['TEST_UPLOAD_LOG'],'a') as f:f.write(remote+'\\n')
 if remote==os.environ.get('FAIL_UPLOAD'):sys.exit(1)
''')
 stub.chmod(0o755)
 script=root/'test.sh';script.write_text('set -euo pipefail\nROOT_DIR='+repr(temp)+'\nexport PATH="$ROOT_DIR/bin:$PATH"\nexport TEST_UPLOAD_LOG="$ROOT_DIR/uploads"\nFTP_HOST=test.invalid\nFTP_USER=test\nFTP_PASS=dummy\nREMOTE_ROOT=\nMANIFEST_FILE="$ROOT_DIR/manifest.tsv"\ndeploy_informatics() {'+source+'\ndeploy_informatics\n')
 def run(env=None):
  r=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
  if r.returncode: print(r.stdout,r.stderr)
  return r
 assert run().returncode==0
 uploaded=(root/'uploads').read_text().splitlines();assert len(uploaded)==4;assert uploaded.index('informatics/pilot-v1/m2.json')<uploaded.index('informatics/pilot-v1/index.json')
 (root/'uploads').write_text('');assert run().returncode==0;assert not (root/'uploads').read_text()
 (root/'manifest.tsv').write_text('old\tunrelated/file\n');before=(root/'manifest.tsv').read_bytes()
 assert run({**os.environ,'FAIL_UPLOAD':'informatics/pilot-v1/m2.json'}).returncode!=0;assert (root/'manifest.tsv').read_bytes()==before
 # A dirty tracked asset blocks the entire batch, before any upload.
 (root/'uploads').write_text('');(root/'Informatics/index.html').write_text('dirty')
 assert run().returncode!=0;assert not (root/'uploads').read_text()
 print('Panthur upload checks passed: changed-only, data before entry points, failed upload preserves manifest, dirty source blocked.')
