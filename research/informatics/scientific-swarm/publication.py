"""Content-addressed JSON/CSV snapshots and a durable two-destination outbox.

Enqueue is local. A remote receipt is required before a destination is marked
delivered. An interrupted upload is reconciled before any retry.
"""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import os
from swarm import digest

DESTINATIONS = ('synapse', 'desci')


def write_csv(path, record):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f); writer.writerow(['field', 'type', 'value', 'escaped'])
        def walk(value, pointer):
            kind = ('null' if value is None else 'boolean' if isinstance(value, bool)
                    else 'object' if isinstance(value, dict) else 'array' if isinstance(value, list)
                    else 'number' if isinstance(value, (int, float)) else 'string')
            text = '' if kind in {'object', 'array', 'null'} else str(value).lower() if kind == 'boolean' else str(value)
            escaped = bool(text and (text[0] in '\t\r\n' or text.lstrip()[:1] in {'=', '+', '-', '@'}))
            writer.writerow([pointer, kind, "'" + text if escaped else text, int(escaped)])
            if kind in {'object', 'array'}:
                for key, child in (value.items() if kind == 'object' else enumerate(value)):
                    walk(child, pointer + '/' + str(key).replace('~', '~0').replace('/', '~1'))
        walk(record, '')


class Outbox:
    def __init__(self, root, sync=lambda: None):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'publication.sqlite')
        self.db.execute('CREATE TABLE IF NOT EXISTS delivery (revision TEXT, destination TEXT, status TEXT, receipt TEXT, PRIMARY KEY(revision,destination))')
        self.sync = sync

    def enqueue(self, record):
        revision = digest(record)
        target = self.root / revision
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=self.root) as temp:
                folder = Path(temp)
                (folder / (revision + '-study.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
                write_csv(folder / (revision + '-study.csv'), record)
                files = [{'path': p.name, 'bytes': p.stat().st_size,
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in sorted(folder.iterdir())]
                manifest = {'schema': 'universal-informatics/research-snapshot/v1',
                            'datasetVersion': revision, 'source': 'https://universalmind.tv/informatics',
                            'reviewStatus': record.get('review_status', 'preliminary'),
                            'files': files}
                raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
                (folder / 'manifest.json').write_bytes(raw)
                (folder / 'integrity.json').write_text(json.dumps({'algorithm': 'sha256', 'sha256': hashlib.sha256(raw).hexdigest()}) + '\n')
                os.rename(folder, target)
                # TemporaryDirectory cleanup tolerates an already moved folder.
        for destination in DESTINATIONS:
            self.db.execute('INSERT OR IGNORE INTO delivery VALUES (?,?,?,NULL)', (revision, destination, 'queued'))
        self.db.commit(); self.sync()
        return {'revision': revision, 'destinations': self.status(revision)}

    def status(self, revision):
        return {dest: state for dest, state in self.db.execute('SELECT destination,status FROM delivery WHERE revision=?', (revision,))}

    def deliver(self, revision, destination, adapter):
        if destination not in DESTINATIONS:
            raise ValueError('Unconfigured publication destination')
        self.db.execute('BEGIN IMMEDIATE')
        row = self.db.execute('SELECT status,receipt FROM delivery WHERE revision=? AND destination=?', (revision, destination)).fetchone()
        if row is None:
            self.db.rollback(); raise ValueError('Snapshot is not queued')
        if row[0] == 'delivered':
            self.db.commit(); return json.loads(row[1])
        previous_state = row[0]
        self.db.execute('UPDATE delivery SET status=? WHERE revision=? AND destination=?', ('in_flight', revision, destination))
        self.db.commit(); self.sync()
        receipt = adapter.reconcile(revision, self.root / revision)
        if not receipt:
            if previous_state != 'queued':
                raise RuntimeError('Publication outcome unresolved; reconcile before another upload.')
            receipt = adapter.publish(revision, self.root / revision)
        if receipt.get('revision') != revision or receipt.get('destination') != destination or not receipt.get('remote_id'):
            raise ValueError('Publication receipt does not match the queued snapshot')
        self.db.execute('UPDATE delivery SET status=?,receipt=? WHERE revision=? AND destination=?',
                        ('delivered', json.dumps(receipt), revision, destination))
        self.db.commit(); self.sync()
        return receipt


class ExistingConnector:
    """Runs the existing reviewed connector against owner-configured targets.

    No project creation, ACL change, blockchain signature or public release.
    A missing final connector receipt after interruption requires reconciliation.
    The connector owns file-level checks and receipt persistence.
    """
    def __init__(self, destination, command, target, credential_file, connector_root):
        if destination not in DESTINATIONS:
            raise ValueError('Unknown destination')
        self.destination, self.command, self.target = destination, command, target
        self.credential_file = str(Path(credential_file).resolve())
        self.root = Path(connector_root).resolve()

    def reconcile(self, revision, folder):
        name = 'synapse-receipt.json' if self.destination == 'synapse' else 'desci-draft-receipt.json'
        path = folder / name
        if not path.exists(): return None
        receipt = json.loads(path.read_text())
        expected = json.loads((folder / 'integrity.json').read_text())['sha256']
        if receipt.get('manifestSHA256') != expected:
            raise ValueError('Connector receipt hash mismatch')
        if self.destination == 'synapse':
            if receipt.get('project') != self.target: raise ValueError('Synapse destination mismatch')
            if receipt.get('status') != 'uploaded': return None
            remote_id = receipt.get('folder')
        else:
            if receipt.get('draftId', '').rstrip('.') != self.target.rstrip('.'): raise ValueError('DeSci destination mismatch')
            if receipt.get('status') != 'uploaded-private-draft': return None
            remote_id = receipt.get('draftId')
        return {'destination': self.destination, 'revision': revision, 'remote_id': remote_id,
                'visibility': 'existing_project_permissions' if self.destination == 'synapse' else 'private_draft',
                'connector_receipt': receipt}

    def publish(self, revision, folder):
        import subprocess
        env = os.environ.copy()
        if self.destination == 'synapse':
            env['SYNAPSE_AUTH_TOKEN_FILE'] = self.credential_file
            args = [self.command, str(self.root / 'publish-synapse.py'), str(folder), '--project', self.target, '--upload']
        else:
            env['DESCI_API_KEY_FILE'] = self.credential_file
            args = [self.command, str(self.root / 'draft-desci.mjs'), '--bundle', str(folder), '--draft', self.target, '--upload']
        result = subprocess.run(args, env=env, capture_output=True, timeout=180)
        receipt = self.reconcile(revision, folder)
        if not receipt:
            raise RuntimeError(f'{self.destination} delivery needs reconciliation; connector exit {result.returncode}.')
        return receipt
