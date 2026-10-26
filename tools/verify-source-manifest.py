#!/usr/bin/env python3
"""Check imported source files against their recorded SHA-256 hashes."""
import hashlib,json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'docs/SOURCE_MANIFEST.json').read_text())
errors=[]
for item in manifest['files']:
    path=root/item['path']
    if not path.is_file():errors.append({'path':item['path'],'error':'missing'});continue
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        errors.append({'path':item['path'],'error':'hash mismatch'})
print(json.dumps({'source_commit':manifest['source_commit'],'files':len(manifest['files']),'errors':errors},indent=2))
raise SystemExit(bool(errors))
