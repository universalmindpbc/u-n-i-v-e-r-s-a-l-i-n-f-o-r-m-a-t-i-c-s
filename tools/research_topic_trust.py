"""Universal Mind topic scores, using the three-study minimum."""


def assess_topic(records):
    # Repeated publications and repeated reports of a registered trial count once.
    publications = {}
    for record in records:
        key = str(record.get('doi') or record.get('pmid') or record['id']).lower()
        publications.setdefault(key, record)
    groups = {'double-blind-rct': {}, 'single-blind-rct': {}, 'peer-reviewed': {}}
    for key, record in publications.items():
        assessment = record.get('studyDesignAssessment', {})
        if not assessment.get('peerReviewed'):
            continue
        groups['peer-reviewed'][key] = record['id']
        category = assessment.get('category')
        if category in ('double-blind-rct', 'single-blind-rct'):
            trial = record.get('evidence', {}).get('trialId') or key
            groups[category].setdefault(trial, record['id'])
    score = next((score for score, category in [(3, 'double-blind-rct'),
                 (2, 'single-blind-rct'), (1, 'peer-reviewed')]
                 if len(groups[category]) >= 3), None)
    return {
        'score': score,
        'scope': 'Subchapter',
        'minimumQualifyingStudies': 3,
        'status': 'assessed' if score else 'below-minimum',
        'qualifyingRecords': {key: list(values.values()) for key, values in groups.items()},
        'counts': {key: len(values) for key, values in groups.items()},
    }
