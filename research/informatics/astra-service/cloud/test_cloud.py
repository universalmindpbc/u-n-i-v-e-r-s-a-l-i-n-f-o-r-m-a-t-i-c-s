import hashlib,io,json,os,sqlite3,sys,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runner
from workflow import Ledger
import runtime,dispatch

class Store:
 class exceptions:
  class NoSuchKey(Exception):pass
 def __init__(self):self.objects={}
 def get_object(self,**kw):
  if kw['Key'] not in self.objects:raise self.exceptions.NoSuchKey()
  raw,meta=self.objects[kw['Key']];return {'Body':io.BytesIO(raw),'Metadata':meta}
 def put_object(self,**kw):
  raw=kw['Body'];self.objects[kw['Key']]=(raw.encode() if isinstance(raw,str) else raw,kw.get('Metadata',{}))

class Tests(unittest.TestCase):
 def test_crash_preserves_model_reservation_before_request(self):
  with tempfile.TemporaryDirectory() as original,tempfile.TemporaryDirectory() as recovered:
   store=Store();archive=runtime.StateArchive('fixture','study',Path(original),store)
   ledger=Ledger(Path(original)/'usage.sqlite',sync=archive.save)
   def uncertain_call():
    self.assertIn(archive.key,store.objects)
    raise RuntimeError('connection lost after submission')
   with self.assertRaisesRegex(RuntimeError,'connection lost'):ledger.run('study','request',100,uncertain_call)
   runtime.StateArchive('fixture','study',Path(recovered),store).restore()
   resumed=Ledger(Path(recovered)/'usage.sqlite')
   with self.assertRaisesRegex(RuntimeError,'Uncertain model request'):resumed.run('study','request',100,lambda:self.fail('duplicate paid request'))

 def test_archive_restores_committed_result_and_rejects_tampering(self):
  with tempfile.TemporaryDirectory() as original,tempfile.TemporaryDirectory() as recovered:
   store=Store();archive=runtime.StateArchive('fixture','study',Path(original),store)
   ledger=Ledger(Path(original)/'usage.sqlite',sync=archive.save)
   ledger.run('study','request',100,lambda:{'tokens':5,'text':'saved'})
   target=runtime.StateArchive('fixture','study',Path(recovered),store);target.restore()
   self.assertEqual(Ledger(Path(recovered)/'usage.sqlite').run('study','request',100,lambda:1/0)['text'],'saved')
   raw,meta=store.objects[archive.key];store.objects[archive.key]=(raw+b'bad',meta)
   with self.assertRaisesRegex(RuntimeError,'checksum'):target.restore()
   data=io.BytesIO()
   with tarfile.open(fileobj=data,mode='w') as tar:
    item=tarfile.TarInfo('../outside');item.size=1;tar.addfile(item,io.BytesIO(b'x'))
   raw=data.getvalue();store.objects[archive.key]=(raw,{'sha256':hashlib.sha256(raw).hexdigest()})
   with self.assertRaisesRegex(RuntimeError,'Invalid checkpoint'):target.restore()

 def test_dispatch_reserves_before_call_and_stops_at_limit(self):
  store=Store();calls=[]
  class Secrets:
   def get_secret_value(self,**kw):return {'SecretString':'{"worker":"fixture"}'}
  class Invoke:
   def invoke(self,**kw):
    self_count=json.loads(next(iter(store.objects.values()))[0])['attempts']
    calls.append(self_count);assert json.loads(kw['Payload'])['environment'] in ['production','staging'];return {}
  def client(name):return {'s3':store,'secretsmanager':Secrets(),'lambda':Invoke()}[name]
  with patch.dict(os.environ,{'ASTRA_SECRET_ARN':'fixture','ASTRA_STATE_BUCKET':'fixture','ASTRA_WORKER_NAME':'fixture'}),patch.object(dispatch.boto3,'client',side_effect=client),patch.object(dispatch.urllib.request,'urlopen',side_effect=lambda *a,**k:io.BytesIO(b'{"pending":true}')):
   self.assertEqual(dispatch.handler({},None)['attempt'],1);self.assertEqual(calls,[1])
   key=next(iter(store.objects));store.put_object(Key=key,Body='{"attempts":120}')
   self.assertEqual(dispatch.handler({},None)['status'],'monthly_hosting_allowance_reached');self.assertEqual(calls,[1])
  store.objects.clear()
  with patch.dict(os.environ,{'ASTRA_SECRET_ARN':'fixture','ASTRA_STATE_BUCKET':'fixture'}),patch.object(dispatch.boto3,'client',side_effect=client),patch.object(dispatch.urllib.request,'urlopen',side_effect=lambda *a,**k:io.BytesIO(b'{"pending":false}')):
   self.assertEqual(dispatch.handler({},None)['status'],'idle')

 def test_environment_is_an_allowlist_not_an_arbitrary_endpoint(self):
  from destinations import destination
  self.assertEqual(destination('production')[1],'https://universalmind.tv')
  self.assertEqual(destination('staging')[1],'https://staging.universalmind.tv')
  with self.assertRaises(ValueError):runtime.handler({'environment':'https://untrusted.invalid'},None)

 def test_shared_review_allowance_never_resets_per_study(self):
  from providers import ReviewLedger
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);store=Store();seed=root/'seed.sqlite'
   ledger=ReviewLedger(seed);ledger.run('prior',100,lambda:{'usage':{'promptTokenCount':1,'totalTokenCount':2}})
   archive=runtime.SharedReviewBudget('fixture',seed,store);archive.save()
   second=runtime.SharedReviewBudget('fixture',root/'second.sqlite',store);second.restore()
   resumed=ReviewLedger(second.path,100,sync=second.save)
   self.assertEqual(resumed.db.execute('SELECT sum(reserve) FROM reviews').fetchone()[0],24)
   with self.assertRaises(RuntimeError):resumed.run('new',77,lambda:self.fail('Reset budget'))
   with self.assertRaisesRegex(RuntimeError,'not been seeded'):runtime.SharedReviewBudget('fixture',root/'empty',Store()).restore()

 def test_csv_retains_all_scientific_fields_and_escapes_formula(self):
  import csv
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'result.csv';runner.write_full_csv(path,{'question':'=fixture','measurements':[{'value':.25,'shots':100}],'memory':{'revision':'r1'},'empty':[],'none':None})
   with path.open(encoding='utf-8-sig') as stream:rows={r['field']:r for r in csv.DictReader(stream)}
   self.assertEqual(rows['/measurements/0/shots']['value'],'100')
   self.assertEqual(rows['/memory/revision']['value'],'r1')
   self.assertEqual(rows['/question']['escaped'],'1')
   self.assertEqual(rows['/empty']['type'],'array');self.assertEqual(rows['/none']['type'],'null')

if __name__=='__main__':unittest.main()
