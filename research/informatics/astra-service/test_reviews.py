import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from langgraph.checkpoint.sqlite import SqliteSaver
import runner
from review_flow import digest,ReviewLedger
from workflow import Ledger

class Reviewer:
 def __init__(self,decisions,events):self.decisions=iter(decisions);self.events=events
 def review(self,study,phase,material):
  self.events.append(phase);decision=next(self.decisions)
  return {'decision':decision,'phase':phase,'summary':'Fixture review','checks':[],'required_changes':['Clarify the scope'] if decision=='revise' else [],'evidence_sha256':digest(material),'provider_response_id':'fixture','model':'fixture','reviewed_at':'2026-10-26T00:00:00Z'}

class Tests(unittest.TestCase):
 def execute(self,decisions,supported=True):
  events=[]
  with tempfile.TemporaryDirectory() as temp,SqliteSaver.from_conn_string(':memory:') as cp:
   root=Path(temp);(root/'credentials.json').write_text('{"openai":"fixture"}')
   def planner(*args):events.append('plan');return {'plan':{'supported':supported,'objective':'OXTR','stop_reason':'' if supported else 'No recipe for this question'},'tokens':10}
   def writer(*args):events.append('report');return {'narrative':{'answer':'No clear improvement'},'tokens':10}
   def prepare(*args):events.append('compute');return {'rows':[],'sources':[]}
   with patch.object(runner,'SERVICE_ROOT',root),patch.object(runner,'SECRET',root/'credentials.json'),patch.object(runner.science,'prior_measurements',return_value={}),patch.object(runner.science,'prepare',side_effect=prepare),patch.object(runner.science,'analyse',return_value={'predictions':[]}),patch.object(runner.science,'load',return_value={'cases':[]}),patch('subprocess.run'):
    g=runner.graph(Ledger(root/'usage.sqlite',24000,4),cp,lambda *a:None,lambda q:{},Reviewer(decisions,events),planner=planner,writer=writer)
    result=g.invoke({'id':'fixture','question':'Fibonacci OXTR'},{'configurable':{'thread_id':'fixture'}})
    return events,result,(root/'results/fixture/result.json').exists()
 def test_reviews_surround_calculations(self):
  events,result,published=self.execute(['approve','approve'])
  self.assertEqual(events,['plan','design','compute','report','final']);self.assertTrue(published)
  self.assertEqual(result['report']['review']['status'],'model_review_complete')
  for record in result['report']['reviews']:self.assertEqual(record['evidence_sha256'],digest(result['report']['review_inputs'][record['evidence_sha256']]))
 def test_design_rejection_never_executes(self):
  events,result,published=self.execute(['human'])
  self.assertEqual(events,['plan','design']);self.assertFalse(published);self.assertIn('dependency',result)
 def test_revisions_are_bounded_and_final_rejection_never_publishes(self):
  events,result,published=self.execute(['revise','approve','revise','human'])
  self.assertEqual(events,['plan','design','plan','design','compute','report','final','report','final']);self.assertFalse(published)
 def test_unsupported_question_never_uses_oxtr_recipe(self):
  events,result,published=self.execute([],False)
  self.assertEqual(events,['plan']);self.assertFalse(published)
 def test_narrative_reservation_omits_arrays_but_retains_metrics(self):
  evidence={'cohort':{'sources':['fixture']},'analysis':{'predictions':[{'value':'x'*100000}],'difference_average_precision':0.002},'prior_quantum':{'rows':[{'value':'y'*100000}],'sha256':'recorded-sha'},'prepared_quantum':{'cases':[{'sample':'one','value':'z'*100000}],'scope':'Local program checks'},'research_memory':{'memories':[{'id':'source','source_sha256':'memory-sha','excerpt':'e'*100000}]}}
  request=runner.brief_input(evidence,'Does Fibonacci improve prediction?')
  data=json.loads(request['prompt'])['calculated_evidence']
  self.assertEqual(data['analysis']['difference_average_precision'],0.002)
  self.assertEqual(data['prior_quantum']['sha256'],'recorded-sha')
  self.assertEqual(data['prepared_quantum']['local_CUDA_Q_programs_completed'],1)
  self.assertEqual(data['research_memory_sources'][0]['source_sha256'],'memory-sha')
  self.assertLess(len(json.dumps(request).encode())+2524,16000)
  self.assertEqual(len(evidence['analysis']['predictions'][0]['value']),100000)
 def test_program_identity_survives_compact_review(self):
  first={'id':'a','sample':'sample','representation':'fibonacci','qasm_sha256':'q','ahs_sha256':'a','pulse':{'times':[0,1]},'reference_observables':[.2]*21,'cudaq_observables':[.2]*21,'max_export_observable_error':1e-5}
  second={**first,'id':'b','representation':'phi'}
  different={**first,'id':'c','representation':'alpha','ahs_sha256':'different'}
  cases=[first,second,different];before=json.dumps(cases)
  summary=runner.program_summary(cases)
  self.assertEqual(summary['identical_programs'][0]['representations'],['fibonacci','phi'])
  self.assertTrue(summary['identical_programs'][0]['pulse_equal'])
  self.assertEqual(len(summary['identical_programs']),1)
  self.assertEqual(summary['receipts'][0]['case_sha256'],digest(first))
  self.assertEqual(summary['receipts'][0]['max_export_observable_error'],1e-5)
  self.assertEqual(json.dumps(cases),before)
 def test_revision_omits_provider_metadata_and_passing_checks(self):
  evidence={'cohort':{},'analysis':{},'prior_quantum':{'rows':[]},'prepared_quantum':{'cases':[]}}
  feedback=[{'summary':'Explain the identical programs','required_changes':['Keep full sequence names'],'checks':[{'status':'pass','finding':'x'*100000},{'status':'fail','finding':'Two identical programs'}],'usage':{'padding':'y'*100000}}]
  request=runner.brief_input(evidence,'OXTR',feedback)
  revision=json.loads(request['prompt'])['Gemini_revision_request'][0]
  self.assertEqual(revision['required_changes'],['Keep full sequence names'])
  self.assertEqual(revision['failed_checks'],[{'status':'fail','finding':'Two identical programs'}])
  self.assertLess(len(json.dumps(request).encode())+2524,16000)
 def test_cost_settlement_keeps_unknown_requests(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'cost.sqlite';ledger=ReviewLedger(path,1000)
   with self.assertRaises(TimeoutError):ledger.run('unknown',300,lambda:(_ for _ in ()).throw(TimeoutError()))
   result=ledger.run('done',500,lambda:{'usage':{'promptTokenCount':10,'totalTokenCount':20}})
   self.assertEqual(result['accounted_micro_usd'],240)
   resumed=ReviewLedger(path,1000);resumed.settle_completed()
   self.assertEqual(resumed.db.execute('SELECT sum(reserve) FROM reviews').fetchone()[0],540)
   with self.assertRaises(RuntimeError):resumed.run('new',461,lambda:self.fail('Budget bypass'))
   with self.assertRaises(RuntimeError):resumed.run('unknown',300,lambda:self.fail('Duplicate call'))
if __name__=='__main__':unittest.main()
