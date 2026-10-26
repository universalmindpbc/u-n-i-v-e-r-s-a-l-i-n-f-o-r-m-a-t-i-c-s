"""Prepare a verifiable snapshot of public datasets and their reproducible source.

Usage: python archive-informatics-snapshot.py NEW_DIRECTORY
Only committed public source enters the archive. No external upload occurs here.
"""
from pathlib import Path
import gzip, hashlib, io, json, subprocess, sys, tarfile
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'Informatics'
out=Path(sys.argv[1]).resolve()
out.mkdir(mode=0o700)
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
source=json.loads((SITE/'exports/manifest.json').read_text())
files=[]
def register(p,role):files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':hashfile(p),'role':role})
for kind,entry in source['downloads'].items():
    target=out/entry['file']
    with target.open('wb') as f:
        for part in entry['parts']:
            p=SITE/'exports'/part['file']
            assert hashfile(p)==part['sha256']
            f.write(p.read_bytes())
    assert target.stat().st_size==entry['bytes'] and hashfile(target)==entry['sha256']
    h=hashlib.sha256();length=0
    with gzip.open(target,'rb') as f:
        while block:=f.read(1024*1024):h.update(block);length+=len(block)
    assert h.hexdigest()==entry['uncompressed_sha256'] and length==entry['uncompressed_bytes']
    register(target,'Complete lossless '+kind.upper()+' dataset, gzip compressed')
    print('Verified',target.name,target.stat().st_size,flush=True)
p=out/'dataset-manifest.json';p.write_bytes((SITE/'exports/manifest.json').read_bytes());register(p,'Original source and asset inventory with individual hashes')
p=out/'DATA-GUIDE.txt';p.write_bytes((SITE/'exports/README.md').read_bytes());register(p,'Dataset schema and reconstruction instructions')
tracked=subprocess.check_output(['git','ls-tree','-r','--name-only','-z',commit],cwd=ROOT).decode().split('\0')
suffixes={'.html','.js','.css','.mjs','.py','.md','.txt','.wl'}
selected=[name for name in tracked if name.startswith('Informatics/') and not name.startswith('Informatics/read/') and Path(name).suffix in suffixes]
selected += ['tools/build-informatics-reading.mjs','tools/check-informatics-discovery.py','tools/archive-informatics-snapshot.py','tools/export-informatics.py','research/informatics/pilot/csv_records.py','robots.txt','robots-staging.txt','sitemap.xml','llms.txt','Informatics/.htaccess']
members=[]
for name in sorted(set(selected)):
    p=ROOT/name
    assert p.is_file() and not p.is_symlink(),name
    committed=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest()==hashfile(p), 'Commit the public source before archiving: '+name
    members.append({'path':name,'bytes':p.stat().st_size,'sha256':hashfile(p)})
target=out/'public-source.tar.gz'
with target.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w|') as tar:
    for m in members:
        content=(ROOT/m['path']).read_bytes();info=tarfile.TarInfo(m['path']);info.size=len(content);info.mode=0o644;info.mtime=0
        tar.addfile(info,io.BytesIO(content))
register(target,'Public client code, scientific methods, data export and reading-page builders')
p=out/'source-files.json';p.write_text(json.dumps(members,indent=2)+'\n');register(p,'SHA-256 inventory for the public source archive')
p=out/'SNAPSHOT.txt';p.write_text('Universal Informatics research snapshot\n\nMaintainer: Universal Mind / Ben Korbel\nSource commit: '+commit+'\nDataset version: '+source['dataset_version']+'\n\nThe JSON and CSV archives preserve the same complete structured values. The public source archive records the public website software and scientific methods used to present and export those values. Generated reading HTML is reproducible using its builder. Binary molecular structures, figures and PDFs retain their source URLs and hashes in the dataset manifest. Private research, credentials, participant records and unpublished third-party full texts are excluded.\n\nSource-specific attribution and licences remain attached to the records. Public AI reading and training permission applies to Universal Mind-owned content under /informatics/. Third-party rights remain with their owners. Hypotheses and unassessed findings retain their recorded status.\n\nArchive manifests bind these exact bytes. Provider records establish storage events. A verified external timestamp establishes that the hashed snapshot existed by the verified time; it does not assess scientific validity or determine authorship.\n');register(p,'Snapshot scope, attribution and interpretation')
manifest={'schemaVersion':'1.0','datasetVersion':'informatics-20261026-'+source['dataset_version'][:12],
 'createdAt':datetime.now(timezone.utc).isoformat(),'source':'https://informatics-universal-one-staging.universalmind.workers.dev',
 'sourceCommit':commit,'datasetContentVersion':source['dataset_version'],'maintainer':'Universal Mind / Ben Korbel',
 'scope':{'structuredScientificSources':len(source['sources']),'linkedAssets':len(source['assets']),'publicSourceFiles':len(members),'privateDataIncluded':False,'fullTextArticlesAdded':False},
 'files':files}
p=out/'manifest.json';p.write_text(json.dumps(manifest,indent=2)+'\n')
(out/'integrity.json').write_text(json.dumps({'algorithm':'SHA-256','manifest':'manifest.json','sha256':hashfile(p)},indent=2)+'\n')
print(json.dumps({'bundle':str(out),'sourceCommit':commit,'files':len(files)+2,'bytes':sum(f['bytes'] for f in files),'manifestSHA256':hashfile(p)}))
