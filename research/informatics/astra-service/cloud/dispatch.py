"""One shared hosting allowance for the production and staging brokers."""
import boto3,json,os,urllib.request
from datetime import datetime,timezone
from destinations import DESTINATIONS

def handler(event,context):
 secret=json.loads(boto3.client('secretsmanager').get_secret_value(SecretId=os.environ['ASTRA_SECRET_ARN'])['SecretString'])
 s3=boto3.client('s3');bucket=os.environ['ASTRA_STATE_BUCKET'];key='allowance/'+datetime.now(timezone.utc).strftime('%Y-%m')+'.json'
 try:record=json.loads(s3.get_object(Bucket=bucket,Key=key)['Body'].read())
 except s3.exceptions.NoSuchKey:record={'attempts':0}
 count=record['attempts']
 if count>=120:return {'status':'monthly_hosting_allowance_reached'}
 # Rotate priority; one worker and one allowance cover both sites.
 names=list(DESTINATIONS)
 if record.get('last_environment')==names[0]:names.reverse()
 selected=None
 for name in names:
  endpoint,origin=DESTINATIONS[name]
  request=urllib.request.Request(endpoint+'/agent/control',data=b'{"op":"pending"}',headers={'Content-Type':'application/json','Origin':origin,'Authorization':'Bearer '+secret['worker'],'User-Agent':'Universal-Informatics-Astra/1.0'})
  try:
   with urllib.request.urlopen(request,timeout=5) as response:pending=json.load(response)
  except Exception:continue
  if pending.get('pending'):selected=name;break
 if selected is None:return {'status':'idle'}
 # Reserved concurrency one serialises this ledger. Unknown outcomes retain reservations.
 s3.put_object(Bucket=bucket,Key=key,Body=json.dumps({'attempts':count+1,'limit':120,'last_environment':selected}),ServerSideEncryption='AES256')
 boto3.client('lambda').invoke(FunctionName=os.environ['ASTRA_WORKER_NAME'],InvocationType='Event',Payload=json.dumps({'environment':selected}).encode())
 return {'status':'dispatched','attempt':count+1,'environment':selected}
