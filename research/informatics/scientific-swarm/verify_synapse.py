"""Download the saved review archive and check every byte against local files."""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import synapseclient

proof = Path(__file__).resolve().parent / 'proof'
status = json.loads((proof / 'publication-status.json').read_text())
bundle = proof / 'publication' / status['revision']
receipt = json.loads((bundle / 'synapse-receipt.json').read_text())
token = Path(os.environ['SYNAPSE_AUTH_TOKEN_FILE']).read_text().strip()
with tempfile.TemporaryDirectory(prefix='astra-review-archive-') as temp:
    syn = synapseclient.Synapse(cache_root_dir=temp + '/cache')
    syn.login(authToken=token, silent=True)
    folder = syn.get(receipt['folder'], downloadFile=False)
    assert folder.parentId == receipt['project']
    checks = []
    for item in receipt['files']:
        record = syn.get(item['id'], version=item['version'], downloadLocation=temp)
        assert record.parentId == receipt['folder']
        actual = Path(record.path).read_bytes()
        expected = (bundle / item['name']).read_bytes()
        assert actual == expected, item['name']
        checks.append({'id': item['id'], 'version': item['version'], 'name': item['name'],
                       'sha256': hashlib.sha256(actual).hexdigest(), 'bytes': len(actual), 'matched': True})
    result = {'passed': True, 'verifiedAt': datetime.now(timezone.utc).isoformat(),
              'folder': receipt['folder'], 'project': receipt['project'],
              'visibility': 'existing_project_permissions', 'files': checks}
    (proof / 'synapse-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'files': len(checks), 'folder': receipt['folder']}))
