"""Build an allowlisted container context. Never copy private credentials."""
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(__file__).resolve().parents[4]
research=Path(sys.argv[1]);dest=Path(sys.argv[2])
if dest.exists():raise SystemExit('Use a new output directory')
dest.mkdir(parents=True)
copies=[]
def copy(source,target):
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
 copies.append({'path':str(target.relative_to(dest)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
service=root/'research/informatics/astra-service'
for name in ['runner.py','science.py','quantum_prepare.py','intent.json','review_flow.py']:copy(service/name,dest/'workspace/research/informatics/astra-service'/name)
for name in ['providers.py','swarm.py']:copy(root/'research/informatics/scientific-swarm'/name,dest/'workspace/research/informatics/scientific-swarm'/name)
for name in ['workflow.py','audit_tool.py']:copy(root/'research/informatics/agent-foundation'/name,dest/'workspace/research/informatics/agent-foundation'/name)
for name in ['oxtr-qrc-qualification.json','oxtr-qrc-study.json','oxtr-next-test-evidence.json']:copy(root/'Informatics/quantum/data'/name,dest/'workspace/Informatics/quantum/data'/name)
for name in ['research/quantum-lab/oxtr-qrc-01/source/ENCFF651DQL.bed.gz','research/quantum-lab/oxtr-qrc-01/results/wolfram-comparison.json','research/quantum-lab/oxtr-qrc-01/results/physical-model.json','research/oxtr-tfbs-v1/private-source-cache/ref-dna-expanded.json','research/oxtr-tfbs-v1/private-source-cache/jaspar2026-vertebrates.jaspar']:
 copy(research/name,dest/'workspace/research-inputs'/name)
for name in ['Dockerfile','runtime.py','destinations.py']:copy(service/'cloud'/name,dest/name)
copy(service/'requirements.txt',dest/'requirements.txt')
(dest/'manifest.json').write_text(json.dumps(copies,indent=2)+'\n')
print(json.dumps({'files':len(copies),'bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'context':str(dest)}))
