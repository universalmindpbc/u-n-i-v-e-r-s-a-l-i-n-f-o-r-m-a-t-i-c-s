import csv
import json
import tempfile
import unittest
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from swarm import build, digest
from providers import ReviewLedger
from publication import Outbox


class Scientist:
    def __init__(self, support=False):
        self.executions = 0; self.plans = 0; self.reports = 0; self.support = support
    def plan(self, question, evidence, feedback):
        self.plans += 1
        return {'goal': question, 'feedback_addressed': len(feedback)}
    def execute(self, plan, key, previous, advice):
        self.executions += 1
        if self.support and not advice:
            return {'status': 'support', 'question': 'Check this predefined uncertainty calculation.', 'tool_receipts': ['fixture-1']}
        return {'status': 'complete', 'result': 'null', 'tool_receipts': ['fixture-1']}
    def report(self, question, plan, evidence, feedback):
        self.reports += 1
        return {'answer': 'No established predictive improvement.', 'feedback_addressed': len(feedback)}


class Reviewer:
    def __init__(self, decisions): self.decisions = list(decisions); self.calls = []
    def review(self, study_id, phase, material):
        self.calls.append(phase)
        decision = self.decisions.pop(0)
        return {'decision': decision, 'summary': 'Fixture scientific review', 'checks': [],
                'required_changes': ['Revise the predefined controls'] if decision == 'revise' else [],
                'evidence_sha256': digest(material), 'provider_response_id': 'fixture-' + str(len(self.calls)),
                'model': 'fixture-only'}


class Tests(unittest.TestCase):
    def test_reviews_surround_execution_and_revisions_are_bounded(self):
        with tempfile.TemporaryDirectory() as temp, SqliteSaver.from_conn_string(':memory:') as cp:
            s, r = Scientist(), Reviewer(['revise', 'approve', 'revise', 'approve'])
            outbox = Outbox(temp)
            graph = build(s, r, cp, outbox.enqueue)
            result = graph.invoke({'id': 'test', 'question': 'Test Fibonacci', 'evidence': {}},
                                  {'configurable': {'thread_id': 'test'}})
            self.assertEqual(r.calls, ['design', 'design', 'final', 'final'])
            self.assertEqual((s.plans, s.executions, s.reports), (2, 1, 2))
            self.assertEqual(result['status'], 'review_complete')
            self.assertEqual(len(result['snapshots']), 5)
            self.assertEqual(result['publication']['destinations'], {'desci': 'queued', 'synapse': 'queued'})

    def test_rejection_stops_before_execution_and_resume_cannot_bypass_review(self):
        with SqliteSaver.from_conn_string(':memory:') as cp:
            s, r = Scientist(), Reviewer(['revise', 'human'])
            g = build(s, r, cp, lambda record: {'hash': digest(record)})
            cfg = {'configurable': {'thread_id': 'reject'}}
            result = g.invoke({'id': 'reject', 'question': 'Test', 'evidence': {}}, cfg)
            self.assertIn('__interrupt__', result); self.assertEqual(s.executions, 0)
            resumed = g.invoke(Command(resume={'decision': 'approve'}), cfg)
            self.assertEqual(s.executions, 0); self.assertEqual(resumed['status'], 'human_action_recorded')
            self.assertNotIn('publication', resumed)

    def test_mid_study_support_preserves_prior_tool_receipts(self):
        with SqliteSaver.from_conn_string(':memory:') as cp:
            s, r = Scientist(True), Reviewer(['approve', 'approve', 'approve'])
            g = build(s, r, cp, lambda record: {'hash': digest(record)})
            result = g.invoke({'id': 'support', 'question': 'Test', 'evidence': {}},
                              {'configurable': {'thread_id': 'support'}})
            self.assertEqual(r.calls, ['design', 'support', 'final'])
            self.assertEqual(result['support_requests'], 1)
            self.assertEqual(result['execution']['tool_receipts'], ['fixture-1'])

    def test_final_rejection_never_reaches_final_publication(self):
        with SqliteSaver.from_conn_string(':memory:') as cp:
            s, r = Scientist(), Reviewer(['approve', 'revise', 'revise'])
            published = []
            g = build(s, r, cp, lambda record: published.append(record))
            result = g.invoke({'id': 'final', 'question': 'Test', 'evidence': {}},
                              {'configurable': {'thread_id': 'final'}})
            self.assertIn('__interrupt__', result)
            self.assertTrue(all(x['review_status'] == 'preliminary' for x in published))

    def test_unknown_model_call_retains_reservation_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cost.sqlite'; ledger = ReviewLedger(path, 100)
            calls = []
            def fail(): calls.append(1); raise TimeoutError()
            with self.assertRaises(TimeoutError): ledger.run('same', 60, fail)
            resumed = ReviewLedger(path, 100)
            with self.assertRaises(RuntimeError): resumed.run('same', 60, fail)
            with self.assertRaises(RuntimeError): resumed.run('new', 60, fail)
            self.assertEqual(calls, [1])

    def test_completed_review_is_cached_across_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cost.sqlite'; ledger = ReviewLedger(path, 100)
            record = ledger.run('same', 60, lambda: {'ok': True, 'usage': {'promptTokenCount': 1, 'totalTokenCount': 2}})
            self.assertEqual(record['accounted_micro_usd'], 24)
            self.assertEqual(ReviewLedger(path, 100).run('same', 60, lambda: self.fail('Duplicate call')), record)

    def test_snapshot_is_idempotent_and_csv_retains_every_value(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Outbox(temp)
            record = {'x/y~': ['=SUM(1)', 0, False, None, {}, []], 'review_status': 'preliminary'}
            one = out.enqueue(record); two = out.enqueue(record); self.assertEqual(one, two)
            folder = Path(temp) / one['revision']
            self.assertEqual(json.loads(next(folder.glob('*-study.json')).read_text()), record)
            with next(folder.glob('*-study.csv')).open(encoding='utf-8-sig') as f: rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 9)
            formula = next(x for x in rows if x['field'] == '/x~1y~0/0')
            self.assertEqual((formula['escaped'], formula['value']), ('1', "'=SUM(1)"))
            self.assertEqual(next(x for x in rows if x['field'].endswith('/2'))['value'], 'false')

    def test_uncertain_upload_is_reconciled_without_a_second_send(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Outbox(temp); rev = out.enqueue({'result': 'null'})['revision']
            class Adapter:
                remote = None
                calls = 0
                def reconcile(self, revision, folder): return self.remote
                def publish(self, revision, folder):
                    self.calls += 1
                    self.remote = {'revision': revision, 'destination': 'synapse', 'remote_id': 'fixture'}
                    raise TimeoutError()
            adapter = Adapter()
            with self.assertRaises(TimeoutError): out.deliver(rev, 'synapse', adapter)
            receipt = Outbox(temp).deliver(rev, 'synapse', adapter)
            self.assertEqual(adapter.calls, 1); self.assertEqual(receipt['remote_id'], 'fixture')
            self.assertEqual(out.status(rev)['desci'], 'queued')

    def test_missing_remote_receipt_blocks_automatic_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Outbox(temp); rev = out.enqueue({'result': 'null'})['revision']
            class Adapter:
                calls = 0
                def reconcile(self, revision, folder): return None
                def publish(self, revision, folder): self.calls += 1; raise TimeoutError()
            adapter = Adapter()
            with self.assertRaises(TimeoutError): out.deliver(rev, 'desci', adapter)
            with self.assertRaises(RuntimeError): out.deliver(rev, 'desci', adapter)
            self.assertEqual(adapter.calls, 1)

if __name__ == '__main__': unittest.main()
