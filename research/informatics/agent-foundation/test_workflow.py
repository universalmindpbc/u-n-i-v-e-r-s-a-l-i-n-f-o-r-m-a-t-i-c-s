import json
import tempfile
import unittest
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from workflow import ALLOWED, Ledger, build, execute_tool

class FakeModel:
    def __init__(self):self.calls=0
    def respond(self,evidence,max_output):
        self.calls+=1
        return {'calls':[{'name':n,'arguments':{}} for n in sorted(ALLOWED)],'text':'','tokens':100} if not evidence else {'calls':[],'text':'Test report; no binding prediction evaluated.','tokens':100}
class Tests(unittest.TestCase):
    def test_checkpoint_review_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            model=FakeModel();ledger=Ledger(Path(d)/'ledger.db');cfg={'configurable':{'thread_id':'demo'}}
            with SqliteSaver.from_conn_string(str(Path(d)/'graph.db')) as saver:
                graph=build(model,ledger,saver);r=graph.invoke({'study_id':'demo','scope':'recorded-qualification-audit'},cfg)
                self.assertTrue(r['__interrupt__']);self.assertEqual(len(r['evidence']),3);self.assertEqual(model.calls,2)
            with SqliteSaver.from_conn_string(str(Path(d)/'graph.db')) as saver:
                graph=build(model,ledger,saver);r=graph.invoke(Command(resume={'decision':'approved','reviewer':'test fixture only'}),cfg)
                self.assertEqual(r['status'],'reviewed');self.assertEqual(model.calls,2)
    def test_unknown_scope_cannot_run(self):
        with SqliteSaver.from_conn_string(':memory:') as saver:
            model=FakeModel();g=build(model,Ledger(':memory:'),saver);cfg={'configurable':{'thread_id':'scope'}}
            r=g.invoke({'study_id':'scope','scope':'submit_aquila'},cfg);self.assertTrue(r['__interrupt__']);self.assertEqual(model.calls,0)
            r=g.invoke(Command(resume={'decision':'approved','reviewer':'fixture'}),cfg);self.assertEqual(r['status'],'unsupported_scope')
    def test_tools_reject_paid_and_arbitrary_arguments(self):
        with self.assertRaises(ValueError):execute_tool('submit_aquila',{})
        with self.assertRaises(ValueError):execute_tool('audit_recorded_measurements',{'path':'/etc/passwd'})
    def test_ledger_cache_allowance_and_unknown(self):
        l=Ledger(':memory:',token_limit=100,call_limit=2)
        self.assertEqual(l.run('s','a',80,lambda:{'tokens':10,'text':'yes'})['text'],'yes')
        self.assertEqual(l.run('s','a',80,lambda:1/0)['text'],'yes')
        with self.assertRaises(RuntimeError):l.run('s','b',95,lambda:{})
        with self.assertRaises(ZeroDivisionError):l.run('s','b',80,lambda:1/0)
        with self.assertRaises(RuntimeError):l.run('s','b',80,lambda:{})
        with self.assertRaises(RuntimeError):l.run('s','c',5,lambda:{})
    def test_measured_comparison(self):
        rows=execute_tool('compare_recorded_responses',{})['rows'];self.assertEqual(len(rows),16)
        self.assertTrue(all(r['calculated_change_percentage_points']==0 for r in rows if r['method']=='ordinary'))
        self.assertTrue(all(r['calculated_change_percentage_points']>0 for r in rows if r['method'] in ['gamma','epsilon']))
if __name__=='__main__':unittest.main()
