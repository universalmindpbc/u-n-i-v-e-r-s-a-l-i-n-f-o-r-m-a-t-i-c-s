"""Provider-neutral scientific oversight with durable LangGraph checkpoints.

Scientists and reviewers are adapters; neither owns budgets, execution rights,
publication destinations or the authoritative study record.
"""
from __future__ import annotations
import hashlib
import json
from typing import Protocol, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


class Scientist(Protocol):
    def plan(self, question: str, evidence: dict, feedback: list) -> dict: ...
    def execute(self, plan: dict, execution_key: str, previous: dict,
                advice: dict) -> dict: ...
    def report(self, question: str, plan: dict, evidence: dict,
               feedback: list) -> dict: ...


class Reviewer(Protocol):
    def review(self, study_id: str, phase: str, material: dict) -> dict: ...


class Study(TypedDict, total=False):
    id: str
    question: str
    evidence: dict
    plan: dict
    execution: dict
    report: dict
    reviews: list
    design_revisions: int
    report_revisions: int
    support_requests: int
    advice: dict
    review: dict
    status: str
    publication: dict
    resolution: dict
    snapshots: list


def checked_review(record, material):
    if record.get('decision') not in {'approve', 'revise', 'human'}:
        raise ValueError('Invalid scientific review decision')
    if record.get('evidence_sha256') != digest(material):
        raise ValueError('Scientific review refers to different evidence')
    if not record.get('provider_response_id') or not record.get('model'):
        raise ValueError('Scientific review is missing its provider receipt')
    for key in ['summary', 'checks', 'required_changes']:
        if key not in record:
            raise ValueError('Scientific review is incomplete')
    if not isinstance(record['summary'], str) or not isinstance(record['checks'], list) or not isinstance(record['required_changes'], list):
        raise ValueError('Scientific review field types are invalid')
    if any(not isinstance(x, str) for x in record['required_changes']):
        raise ValueError('Required changes must be explicit text')
    for check in record['checks']:
        if set(check) != {'claim', 'finding', 'evidence', 'status'} or check['status'] not in {'pass', 'fail', 'not_tested'}:
            raise ValueError('Invalid scientific check')
        if not isinstance(check['evidence'], list) or any(not isinstance(x, str) for x in check['evidence']):
            raise ValueError('Scientific checks require source identifiers')
    if record['decision'] == 'approve' and record['required_changes']:
        raise ValueError('Approval contains unresolved required changes')
    return record


def build(scientist, reviewer, checkpointer, publish, *, max_revisions=1,
          max_support=1, progress=lambda event: None):
    """An interruption resumes once, then exits for an explicit owner action.

    Resumption never transforms a rejected review into an approval. Model
    adapters reserve their own shared allowance before calls. Tool adapters
    reconcile stable execution keys before repeating an external action.
    """
    def snapshot(s, phase, data):
        receipt = publish({'schema': 'universal-informatics/scientific-progress/v1',
                           'study_id': s['id'], 'phase': phase,
                           'review_status': 'preliminary', 'data': data})
        return {'snapshots': [*s.get('snapshots', []), receipt]}

    def review(s, phase, material):
        record = checked_review(reviewer.review(s['id'], phase, material), material)
        record = {**record, 'phase': phase}
        progress({'stage': phase, 'decision': record['decision']})
        return {'review': record, 'reviews': [*s.get('reviews', []), record],
                **snapshot(s, phase, {'review': record, 'material': material})}

    def plan(s):
        feedback = [r for r in s.get('reviews', []) if r['phase'] == 'design']
        value = scientist.plan(s['question'], s['evidence'], feedback)
        if not isinstance(value, dict) or not value:
            raise ValueError('Astra did not return a study design')
        return {'plan': value, 'status': 'design_review'}

    def design(s):
        return review(s, 'design', {'question': s['question'], 'plan': s['plan'],
                                   'evidence': s['evidence']})

    def next_design(s):
        if s['review']['decision'] == 'approve':
            return 'execute'
        if s['review']['decision'] == 'revise' and s.get('design_revisions', 0) < max_revisions:
            return 'revise_design'
        return 'human'

    def execute(s):
        # Advice changes the phase key while previous tool receipts remain visible.
        key = s['id'] + ':' + digest({'plan': s['plan'], 'advice': s.get('advice', {})})
        result = scientist.execute(s['plan'], key, s.get('execution', {}), s.get('advice', {}))
        if result.get('status') not in {'complete', 'support', 'human'}:
            raise ValueError('Invalid scientific execution state')
        if result['status'] == 'support' and not result.get('question'):
            raise ValueError('A support request needs a specific question')
        return {'execution': result, 'status': result['status'],
                **snapshot(s, 'execution', result)}

    def next_execution(s):
        if s['execution']['status'] == 'complete':
            return 'report'
        if s['execution']['status'] == 'support' and s.get('support_requests', 0) < max_support:
            return 'support'
        return 'human'

    def support(s):
        result = review(s, 'support', {'plan': s['plan'], 'execution': s['execution'],
                                      'evidence': s['evidence']})
        return {**result, 'advice': result['review'],
                'support_requests': s.get('support_requests', 0) + 1}

    def report(s):
        feedback = [r for r in s.get('reviews', []) if r['phase'] == 'final']
        return {'report': scientist.report(s['question'], s['plan'], s['execution'], feedback),
                'status': 'final_review'}

    def final(s):
        # The reviewer receives actual calculations, source provenance and the
        # report, not just Astra's summary of its own work.
        return review(s, 'final', {'question': s['question'], 'plan': s['plan'],
                                  'execution': s['execution'], 'report': s['report']})

    def next_final(s):
        if s['review']['decision'] == 'approve':
            return 'publish'
        if s['review']['decision'] == 'revise' and s.get('report_revisions', 0) < max_revisions:
            return 'revise_report'
        return 'human'

    def publication(s):
        record = {'schema': 'universal-informatics/scientific-review/v1',
                  'study_id': s['id'], 'question': s['question'], 'plan': s['plan'],
                  'execution': s['execution'], 'report': s['report'], 'reviews': s['reviews'],
                  'review_status': 'model_review_complete',
                  'interpretation': 'Cross-model scientific review; not independent laboratory confirmation.'}
        return {'publication': publish(record), 'status': 'review_complete'}

    def human(s):
        value = interrupt({'kind': 'scientific_dependency', 'study_id': s['id'],
                           'review': s.get('review'), 'execution_status': s.get('execution', {}).get('status'),
                           'message': 'Progress is saved. A scope, access, budget or scientific decision needs a human.'})
        return {'status': 'human_action_recorded', 'resolution': value}

    g = StateGraph(Study)
    nodes = {'plan': plan, 'design': design, 'execute': execute, 'support': support,
             'report': report, 'final': final, 'publish': publication, 'human': human,
             'revise_design': lambda s: {'design_revisions': s.get('design_revisions', 0) + 1},
             'revise_report': lambda s: {'report_revisions': s.get('report_revisions', 0) + 1}}
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, 'plan'); g.add_edge('plan', 'design')
    g.add_conditional_edges('design', next_design)
    g.add_edge('revise_design', 'plan')
    g.add_conditional_edges('execute', next_execution)
    g.add_conditional_edges('support', lambda s: 'human' if s['review']['decision'] == 'human' else 'execute')
    g.add_edge('report', 'final'); g.add_conditional_edges('final', next_final)
    g.add_edge('revise_report', 'report')
    g.add_edge('publish', END); g.add_edge('human', END)
    return g.compile(checkpointer=checkpointer)
