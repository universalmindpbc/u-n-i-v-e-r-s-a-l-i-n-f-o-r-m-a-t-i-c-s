"""Verify a research export, then optionally upload it to an existing Synapse project.

python publish-synapse.py BUNDLE_DIRECTORY               # verify only
SYNAPSE_AUTH_TOKEN=... python publish-synapse.py BUNDLE --project syn123 --upload
SYNAPSE_AUTH_TOKEN_FILE=/private/path/token is also supported.
Install synapseclient for upload. No account, project or public sharing is created.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import os
import tempfile
from datetime import datetime, timezone


def verify(folder):
    folder = Path(folder).resolve()
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    integrity = json.loads((folder / 'integrity.json').read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != integrity['sha256']:
        raise ValueError('Manifest hash mismatch.')
    paths = []
    names = set()
    seen = set()
    for entry in manifest['files']:
        raw = folder / entry['path']
        if any(part.is_symlink() for part in [raw, *raw.parents] if part != folder):
            raise ValueError('Symbolic links are not accepted.')
        p = raw.resolve()
        if not p.is_relative_to(folder) or p in seen or p.is_symlink():
            raise ValueError('Unsafe or duplicate file path.')
        name = str(p.relative_to(folder)).replace('/', '__')
        if name in names:
            raise ValueError('Upload names collide.')
        names.add(name)
        seen.add(p)
        data = p.read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('File verification failed: ' + entry['path'])
        paths.append(p)
    return manifest, paths + [manifest_path, folder / 'integrity.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle')
    parser.add_argument('--project')
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    manifest, files = verify(args.bundle)
    print(f'Verified {len(files)} files for {manifest["datasetVersion"]}.')
    if not args.upload:
        print('Verification only. No upload performed.')
        return
    if not re.fullmatch(r'syn\d+', args.project or ''):
        parser.error('--upload requires --project synID.')
    import synapseclient
    token = os.environ.get('SYNAPSE_AUTH_TOKEN')
    if not token and os.environ.get('SYNAPSE_AUTH_TOKEN_FILE'):
        token = Path(os.environ['SYNAPSE_AUTH_TOKEN_FILE']).read_text().strip()
    if not token:
        raise ValueError('Set SYNAPSE_AUTH_TOKEN_FILE or SYNAPSE_AUTH_TOKEN.')
    syn = synapseclient.Synapse(cache_root_dir=tempfile.mkdtemp(prefix='informatics-synapse-cache-'))
    syn.login(authToken=token, silent=True)
    if not syn.is_certified(syn.getUserProfile()['ownerId']):
        raise SystemExit('Synapse account certification is required before upload: https://docs.synapse.org/synapse-docs/synapse-user-account-types')
    project = syn.get(args.project)
    if project.concreteType != 'org.sagebionetworks.repo.model.Project':
        raise ValueError('Destination is not a Synapse project.')
    destination = Path(args.bundle) / 'synapse-receipt.json'
    manifest_hash = hashlib.sha256((Path(args.bundle) / 'manifest.json').read_bytes()).hexdigest()
    if destination.exists():
        receipt = json.loads(destination.read_text())
        if receipt['project'] != args.project or receipt['manifestSHA256'] != manifest_hash:
            raise ValueError('Receipt belongs to a different project or export.')
        folder = syn.get(receipt['folder'])
        if folder.parentId != args.project:
            raise ValueError('Recorded folder is outside the selected project.')
    else:
        folder = syn.store(synapseclient.Folder(name='Universal Informatics ' + manifest['datasetVersion'], parent=args.project))
        receipt = {'project': args.project, 'folder': folder.id, 'manifestSHA256': manifest_hash, 'status': 'uploading', 'files': []}
    def save_receipt():
        destination.write_text(json.dumps(receipt, indent=2) + '\n')
        destination.chmod(0o600)
    save_receipt()
    for p in files:
        # Flatten names safely; the manifest retains original relative paths.
        name = str(p.relative_to(Path(args.bundle).resolve())).replace('/', '__')
        if any(item['name'] == name for item in receipt['files']):
            continue
        entity = syn.store(synapseclient.File(path=str(p), name=name, parent=folder.id), used=manifest['source'] + '/api/v1/inventory')
        receipt['files'].append({'name': name, 'id': entity.id, 'version': entity.versionNumber})
        save_receipt()
    receipt['status'] = 'uploaded'
    receipt['completedAt'] = datetime.now(timezone.utc).isoformat()
    save_receipt()
    print('Uploaded under existing project permissions: ' + folder.id)


if __name__ == '__main__':
    main()
