"""Paper-level Trust Factor from publication-specific evidence, never chapter grades."""
import re

POLICY = {
    '3': 'Confirmed double-blind randomized controlled trial',
    '2': 'Confirmed randomized controlled trial',
    '1': 'Peer-reviewed study',
    'unrated': 'Insufficient publication-specific evidence, protocol, guidance, preprint, retraction or unresolved identity',
}
RCT_CATEGORIES = {'double-blind-rct', 'single-blind-rct', 'rct', 'rct-masking-unclear', 'rct-open-label'}
EXCLUDED_CATEGORIES = {'protocol', 'guidance', 'context', 'preprint', 'technical-article'}

def normal(text):
    return re.sub(r'[‐‑–—-]', ' ', str(text or '')).lower()

def assess_paper(record, entries=()):
    assessment = record.get('studyDesignAssessment') or {}
    category = assessment.get('category')
    legacy_stars = record.get('legacyArticleStars', record.get('stars'))
    design = record.get('design') if assessment else None
    design = design or next((p.get('design') for p in entries if p.get('design')), None) or record.get('sourceDesign') or record.get('design') or ''
    types = record.get('publicationTypes') or []
    label = normal(' '.join([design, record.get('kind') or '', *types]))
    title = normal(record.get('title'))
    sources = list(dict.fromkeys(assessment.get('sources') or record.get('urls') or [u for p in entries for u in p.get('urls', [])]))
    if record.get('pmid'):
        sources = list(dict.fromkeys([f"https://pubmed.ncbi.nlm.nih.gov/{record['pmid']}/", *sources]))
    result = {'score': None, 'scope': 'Paper', 'status': 'unassessed', 'randomizedControlledTrial': None,
              'blinding': 'not-confirmed', 'peerReviewed': bool(assessment.get('peerReviewed')),
              'basis': 'Publication-specific design or publication status remains unconfirmed.',
              'sources': sources, 'reviewScope': 'Existing publication-specific assessments and bibliographic indexing; no blanket full-text appraisal.',
              'assessedAt': '2026-10-27'}
    excluded = category in EXCLUDED_CATEGORIES or record.get('retracted') or record.get('identityConflict') or re.search(r'\b(protocol|editorial|retraction|retracted|preprint)\b', label) or re.search(r'\b(?:study protocol|protocol for|protocol of)\b', title)
    if excluded:
        result.update(status='not-applicable', basis='This record is outside the paper-rating criteria or has unresolved publication identity.')
        return result
    peer = assessment.get('peerReviewed')
    if peer is None:
        peer = legacy_stars in (1, 2, 3) or 'Journal Article' in types or normal(record.get('kind')) in ('journal article', 'review / synthesis', 'review')
    result['peerReviewed'] = bool(peer)
    review = category == 'review' or bool(re.search(r'\b(review|meta analysis|synthesis)\b', label + ' ' + title))
    nonrandom = bool(re.search(r'\b(non\s*randomi[sz]ed|observational|cross sectional|case report|case series|mendelian)\b', normal(design) + ' ' + title))
    animal = bool(re.search(r'\b(mouse|mice|rats|animal|in vitro|cell culture)\b', normal(design)))
    indexed_rct = 'Randomized Controlled Trial' in types
    reported_rct = bool(re.search(r'\brandomi[sz](?:ed|ation)\b', normal(design) + ' ' + title))
    legacy_rct = not assessment and legacy_stars in (2, 3)
    explicit_rct = assessment.get('isRandomizedControlledTrial')
    trial = explicit_rct if explicit_rct is not None else category in RCT_CATEGORIES or ((indexed_rct or reported_rct or legacy_rct) and not (review or nonrandom or animal))
    blind = 'not-confirmed'
    if trial:
        if category == 'double-blind-rct': blind = 'double'
        elif category == 'single-blind-rct': blind = 'single'
        elif category == 'rct-open-label': blind = 'open-label'
        elif category == 'rct-masking-unclear': blind = 'not-confirmed'
        elif re.search(r'\bdouble\s+(?:blind|mask)', normal(design)): blind = 'double'
        elif re.search(r'\bsingle\s+(?:blind|mask)', normal(design)): blind = 'single'
        elif re.search(r'\bopen\s+label\b|\bunblinded\b', normal(design)): blind = 'open-label'
        elif not assessment and legacy_stars in (2, 3): blind = 'double' if legacy_stars == 3 else 'single'
    if trial:
        result.update(score=3 if blind == 'double' else 2, status='assessed', randomizedControlledTrial=True, blinding=blind,
                      basis=assessment.get('basis') or record.get('legacyRatingBasis', record.get('ratingBasis')) if legacy_rct else assessment.get('basis') or 'Randomized controlled design recorded in the publication-specific design or bibliographic indexing.')
    elif peer:
        result.update(score=1, status='assessed', randomizedControlledTrial=False if explicit_rct is False or review or nonrandom or animal else None,
                      blinding='not-applicable' if review or animal else 'not-confirmed',
                      basis=assessment.get('basis') or 'Indexed peer-reviewed research; a qualifying randomized controlled design is not established in the reviewed record.')
    return result
