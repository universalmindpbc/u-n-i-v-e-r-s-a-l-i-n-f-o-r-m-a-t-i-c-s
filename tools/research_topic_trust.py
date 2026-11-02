"""Chapter Trust Factor: three distinct qualifying studies, highest level met."""
from research_paper_trust import assess_paper


def assess_topic(records):
    groups = {'double-blind-rct': {}, 'rct': {}, 'single-blind-rct': {}, 'peer-reviewed': {}}
    for record in records:
        paper = record.get('paperTrustAssessment') or assess_paper(record)
        publication = str(record.get('doi') or record.get('pmid') or record['id']).lower()
        trial = (record.get('evidence') or {}).get('trialId') or publication
        if paper.get('peerReviewed') and paper.get('score'):
            groups['peer-reviewed'].setdefault(trial, record['id'])
        if paper.get('randomizedControlledTrial'):
            groups['rct'].setdefault(trial, record['id'])
            if paper.get('blinding') == 'double': groups['double-blind-rct'].setdefault(trial, record['id'])
            if paper.get('blinding') == 'single': groups['single-blind-rct'].setdefault(trial, record['id'])
    score = next((score for score, category in [(3, 'double-blind-rct'), (2, 'rct'), (1, 'peer-reviewed')]
                  if len(groups[category]) >= 3), None)
    return {'score': score, 'scope': 'Subchapter', 'minimumQualifyingStudies': 3,
            'status': 'assessed' if score else 'below-minimum',
            'qualifyingRecords': {key: list(values.values()) for key, values in groups.items()},
            'counts': {key: len(values) for key, values in groups.items()},
            'assessedAt': '2026-10-27',
            'basis': 'Highest threshold supported by the recorded paper assessments; repeated reports of an identified trial count once.'}
