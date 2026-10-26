import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from langgraph.checkpoint.sqlite import SqliteSaver
import runner
from test_reviews import Reviewer
from workflow import Ledger
class RunnerTests(unittest.TestCase):
 def test_resume_skips_completed_data_and_model(self):
  with tempfile.TemporaryDirectory() as d,SqliteSaver.from_conn_string(':memory:') as cp:
   cohort={'rows':[],'sources':[]};analysis={'predictions':[{'sample':'fixture','score':.1}]};prepared={'cases':[]};calls={'data':0,'analysis':0,'model':0}
   def data(progress):calls['data']+=1;return cohort
   def analyse(c):
    calls['analysis']+=1
    if calls['analysis']==1:raise RuntimeError('interruption fixture')
    return analysis
   def brief(e,q,feedback=None):calls['model']+=1;return {'tokens':10,'narrative':{'answer':'Fixture only'}}
   secret=Path(d)/'credentials.json';secret.write_text('{"openai":"fixture"}')
   with patch.object(runner,'SECRET',secret),patch.object(runner,'SERVICE_ROOT',Path(d)),patch.object(runner.science,'prior_measurements',return_value={}),patch.object(runner.science,'prepare',side_effect=data),patch.object(runner.science,'analyse',side_effect=analyse),patch.object(runner.science,'load',return_value=prepared),patch('subprocess.run'),patch.object(runner,'brief',side_effect=brief):
    g=runner.graph(Ledger(Path(d)/'ledger.sqlite'),cp,lambda *a:None,lambda q:{'schema':'fixture','memories':[]},reviewer=Reviewer(['approve','approve'],[]),planner=lambda *a:{'plan':{'supported':True,'stop_reason':''},'tokens':10});cfg={'configurable':{'thread_id':'fixture'}}
    with self.assertRaises(RuntimeError):g.invoke({'id':'fixture','question':'fixture question'},cfg)
    out=g.invoke(None,cfg);self.assertIn('report',out);self.assertEqual(calls,{'data':1,'analysis':2,'model':1})
    again=g.get_state(cfg);self.assertFalse(again.next);self.assertTrue((Path(d)/'results/fixture/result.json').exists());self.assertTrue((Path(d)/'results/fixture/result.csv').exists())
if __name__=='__main__':unittest.main()
