"""Execute the approved, frozen OXTR qualification batch once; retain every receipt."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import uuid
import boto3
from botocore.config import Config
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
HW = OUT / 'hardware'
DEVICE = 'arn:aws:braket:us-east-1::device/qpu/quera/Aquila'
BUCKET = 'amazon-braket-quantum-lab-740168708062-us-east-1'
CAP = Decimal('22.10')
FEE = Decimal('1.30')

def save(path, data, exclusive=False):
    with path.open('x' if exclusive else 'w') as f:
        json.dump(data, f, indent=2, default=str)
        f.write('\n')

def active_limits(client):
    now = datetime.now(timezone.utc)
    rows = client.search_spending_limits(filters=[])['spendingLimits']
    rows = [r for r in rows if r['deviceArn'] == DEVICE and r['timePeriod']['startAt'] <= now <= r['timePeriod']['endAt']]
    assert rows, 'No active Aquila spending limit'
    assert all(Decimal(r['spendingLimit']) <= CAP for r in rows), 'Cap exceeds authorisation'
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inspect', 'submit', 'resume', 'status'])
    mode = parser.parse_args().mode
    HW.mkdir(exist_ok=True)
    manifest = json.loads((OUT / 'hardware-qualification-manifest.json').read_text())
    assert manifest['passed'] and manifest['tasks'] == 16 and manifest['shots_per_task'] == 100
    records = manifest['records']
    assert len(records) == 16 and len({r['id'] for r in records}) == 16
    for r in records:
        action = (OUT / 'programs' / r['file']).read_text()
        assert hashlib.sha256(action.encode()).hexdigest() == r['action_sha256']
    session = boto3.Session(profile_name='universal-mind-quantum', region_name='us-east-1')
    identity = session.client('sts').get_caller_identity()
    assert identity['Account'] == '740168708062'
    assert ':assumed-role/AWSReservedSSO_UniversalMindQuantumLab_' in identity['Arn']
    client = session.client('braket', config=Config(retries={'max_attempts': 0}, read_timeout=60))
    if mode == 'status':
        summaries = []
        ideal = {r['id']: r for r in json.loads((OUT/'qualification.json').read_text())['records']}
        for r in records:
            receipt = HW / (r['id'] + '-submission.json')
            if not receipt.exists():
                continue
            arn = json.loads(receipt.read_text())['quantumTaskArn']
            meta = client.get_quantum_task(quantumTaskArn=arn, additionalAttributeNames=['QueueInfo'])
            save(HW / (r['id'] + '-status.json'), meta)
            row = {'id': r['id'], 'task_arn': arn, 'status': meta['status'], 'shots': meta['shots'], 'queue': meta.get('queueInfo')}
            if meta['status'] == 'COMPLETED':
                raw = session.client('s3').get_object(Bucket=meta['outputS3Bucket'], Key=meta['outputS3Directory']+'/results.json')['Body'].read()
                (HW / (r['id'] + '-result.json')).write_bytes(raw)
                result = json.loads(raw)
                valid = []; failed = missing = 0
                for shot in result['measurements']:
                    if shot['shotMetadata']['shotStatus'] != 'Success':
                        failed += 1
                        continue
                    pre, post = shot['shotResult']['preSequence'], shot['shotResult']['postSequence']
                    assert len(pre) == len(post) == 6
                    if not all(pre):
                        missing += 1
                        continue
                    valid.append([1-int(v) for v in post])
                counts = Counter(''.join(map(str, reversed(v))) for v in valid)
                row.update(returned_shots=len(result['measurements']), failed_shots=failed,
                           initially_empty_shots=missing, fully_occupied_shots=len(valid), counts=dict(counts),
                           ended_at=result['taskMetadata'].get('endedAt'))
                if valid:
                    a = np.array(valid)
                    obs = np.column_stack([a]+[(a[:,i]*a[:,j])[:,None] for i in range(6) for j in range(i+1,6)]).mean(axis=0)
                    row.update(measured_observables=obs.tolist(), ideal_observables=ideal[r['id']]['ideal_observables'])
            summaries.append(row)
            print(r['id'], meta['status'], flush=True)
        save(OUT / 'hardware-qualification.json', {'at':datetime.now(timezone.utc).isoformat(), 'complete':len(summaries)==16 and all(r['status']=='COMPLETED' for r in summaries), 'records':summaries, 'purpose':'Measurement qualification; no occupancy prediction result. Hardware readout includes atom loss and noise.'})
        save(OUT / 'spending-limit-current.json', client.search_spending_limits(filters=[]))
        return
    gpu = json.loads((OUT / 'gpu-qualification.json').read_text())
    assert gpu['complete'] and len(gpu['cases']) == 17 and all(r['passed'] for r in gpu['cases'])
    assert gpu['cost_credits'] == 0
    wolfram = json.loads((OUT / 'wolfram-comparison.json').read_text())
    assert wolfram['passed']
    capability = client.get_device(deviceArn=DEVICE)
    frozen = json.loads((ROOT / 'source/aquila-device.json').read_text())
    assert capability['deviceStatus'] == 'ONLINE'
    nowcap = json.loads(capability['deviceCapabilities']); oldcap = json.loads(frozen['deviceCapabilities'])
    assert nowcap['paradigm'] == oldcap['paradigm'], 'Device physics changed; repeat model validation'
    assert nowcap['service']['shotsRange'][0] <= 100 <= nowcap['service']['shotsRange'][1]
    limits = active_limits(client)
    save(OUT / 'hardware-preflight.json', {'at':datetime.now(timezone.utc).isoformat(), 'identity':identity['Arn'], 'limits':limits, 'manifest_sha256':hashlib.sha256((OUT/'hardware-qualification-manifest.json').read_bytes()).hexdigest(), 'gpu_complete':True, 'physics_unchanged':True})
    print('Preflight: 16 checked programs, GPU complete, cap', [r['spendingLimit'] for r in limits], flush=True)
    if mode == 'inspect':
        return
    approval = json.loads((OUT / 'spend-approval.json').read_text())
    assert approval['approved_batch_usd'] == '20.80' and approval['approved_cumulative_cap_usd'] == '22.10'
    assert approval['tasks'] == 16 and approval['shots_per_task'] == 100
    pending = []
    for r in records:
        request_file = HW / (r['id']+'-request.json')
        receipt_file = HW / (r['id']+'-submission.json')
        if request_file.exists():
            assert mode == 'resume', 'Use explicit resume after inspecting previous receipts'
            assert receipt_file.exists(), 'Unknown request outcome: reconcile before any further submission'
            saved = json.loads(request_file.read_text())
            assert hashlib.sha256(saved['action'].encode()).hexdigest() == r['action_sha256']
            assert saved['shots'] == 100 and saved['deviceArn'] == DEVICE
            meta = client.get_quantum_task(quantumTaskArn=json.loads(receipt_file.read_text())['quantumTaskArn'])
            assert meta['deviceArn'] == DEVICE and meta['shots'] == 100
        else:
            assert not receipt_file.exists()
            pending.append(r)
    # The live API totalSpend includes reserved queued spend. Confirmed against
    # the console: total 11.70, queued 10.40, current 1.30, remaining 10.40.
    # Subtracting queuedSpend again double-counts the reservation.
    assert all(Decimal(r['totalSpend']) <= Decimal('1.30')+(16-len(pending))*FEE for r in limits), 'Unrelated spend detected'
    assert all(Decimal(r['spendingLimit'])-Decimal(r['totalSpend']) >= len(pending)*FEE for r in limits), 'Insufficient remaining approved budget'
    for r in pending:
        limits = active_limits(client)
        assert all(Decimal(x['spendingLimit'])-Decimal(x['totalSpend']) >= FEE for x in limits), 'Insufficient remaining budget'
        request = {'clientToken':str(uuid.uuid4()), 'deviceArn':DEVICE, 'shots':100,
                   'action':(OUT/'programs'/r['file']).read_text(), 'outputS3Bucket':BUCKET,
                   'outputS3KeyPrefix':'quantum-lab/oxtr-qrc-01/qualification-v1/'+r['id'],
                   'tags':{'Project':'QuantumLab','Experiment':'oxtr-qrc-01'}}
        save(HW / (r['id']+'-request.json'), request, exclusive=True)
        response = client.create_quantum_task(**request)
        save(HW / (r['id']+'-submission.json'), response, exclusive=True)
        print(r['id'], response['quantumTaskArn'], flush=True)
    save(OUT / 'hardware-batch-submitted.json', {'at':datetime.now(timezone.utc).isoformat(), 'tasks':16,'shots':1600,'quantum_fee_usd':'20.80','cumulative_cap_usd':'22.10'})

if __name__ == '__main__':
    main()
