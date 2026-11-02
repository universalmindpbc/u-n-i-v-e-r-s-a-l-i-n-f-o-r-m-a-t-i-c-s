"""Prepare Ben's approved youth summaries from the immutable review; no network."""
import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from research_paper_trust import POLICY
from research_topic_trust import assess_topic

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'Informatics'
SOURCE = SITE / 'review/youth-20261102/batch-f164c8325ea5f9c9.json'
SOURCE_SHA = 'f164c8325ea5f9c989c29524518b598913508117712826dc408f592b7f99a723'
OUT = SITE / 'library-data/youth-expansion.json'
DATE = '2026-11-02'
ALIASES = {'19798037': 'src-56e515aacddb', **{p: 'pmid-'+p for p in ['42324567', '40243164', '41340540', '29422189']}}
SINGLE_MASKED = set('19798037 42389090 22893815 36186313 34040886 29422189 24499130 35914112 36991537 40086808 36877740 34002511 37261464 21559213 41171275'.split())
# Source-supported broad age groups. The existing Youth filter includes adolescents;
# it is not an assertion that the paper isolates 16–24-year-olds.
CHILDREN = set('42324567 41340540 29066991 22893815 42337783 32232017 35820989 30345511 34527971 38566202 37393283 38693527 39813031 34294067 40761439 32572961 32527236 40504611 40521225 39666705 32504808 36342533'.split())
ADOLESCENTS = set('42324567 40243164 41340540 29066991 22893815 36186313 42337783 36407676 29144545 32232017 35820989 30345511 35914112 24901900 26220564 34527971 41454356 36991537 40086808 36620186 39720610 38566202 34983460 35980733 37393283 34002511 37261464 39260824 40180398 39813032 21192795 39813031 32442160 38693527 39004687 34294067 40761439 38584397 37390108 40789917 38939220 22799879 25537785 32572961 41262946 36373473 36964523 38090786 33725578 34021113 35544295 37213197 38362237 39665442 35976180 32683737 32527236 40504611 40521225 39666705 36424575 34927592 32504808 36342533 38192586'.split())
NO_ADULTS = set('42324567 40243164 22893815 36186313 42337783 30345511 24901900 26220564 34527971 41454356 36991537 40086808 38566202 38939220 32572961 41262946 32683737'.split())
OLDER = set('37205669 29144545 22799879'.split())
ARTS = {'36342533', '38192586'}
MUSIC_MAKING = {'41662046', '36407676', '37151974'}

TOPICS = [
 ('black-dog-youth', 'Black Dog Institute · Youth research', 'institution', 'Black Dog Institute'),
 ('orygen-youth', 'Orygen · Youth research', 'institution', 'Orygen'),
 ('universal-mind-mobile-lab', 'Universal Mind Mobile Lab · Direct response and inspired by', 'lab', None),
 ('emdr-youth', 'EMDR for youth', 'topic', 'EMDR'),
 ('music-youth', 'Music listening and therapy for youth', 'topic', 'Music / creative participation'),
 ('music-making-youth', 'Music making for youth', 'making', None),
 ('sound-entrainment-youth', 'Sound and brain entrainment for youth', 'topic', 'Sound / brain entrainment'),
 ('mindfulness-youth', 'Mindfulness for youth', 'topic', 'Mindfulness'),
 ('yoga-youth', 'Yoga for youth', 'topic', 'Yoga'),
 ('mantra-youth', 'Mantra for youth · Evidence gap', 'topic', 'Mantra'),
 ('transcendental-meditation-youth', 'Transcendental meditation for youth', 'topic', 'Transcendental meditation'),
 ('youth-access-connection', 'Youth access, digital support and connection', 'topic', 'Youth needs / digital / access'),
]
LAB_COPY = [
 'The Universal Mind Mobile Lab is planned as an education and creative-skills initiative: electronic music, fashion and mindfulness experiences, stigma reduction and referral pathways. Reduced short-term stress and greater felt connection are exploratory outcomes to measure. No included paper tests the combined lab model.',
 'Direct response means a response to a documented need or access barrier. Inspired by means a design or evaluation idea informed by relevant research. Neither label establishes that the proposed lab will reproduce a study outcome. Null findings and limits are included alongside supportive evidence.',
 'Ben has identified Central Coast, Bondi / Bondi Junction and Northern Rivers near Byron Bay as the first three destinations. Permanent sites depend on community and council agreement. Public self-nomination, touring and smaller pop-ups are planned access routes. Headspace, Black Dog Institute and Orygen are prospective partners; no partnership or endorsement is asserted.',
 'Programme planning includes youth co-design, community authority, mentoring for 16–17-year-olds, separate clinical referral and nonclinical learning pathways, and independent choices about research and filming. Continuity after a visit, referral follow-through, affordability, transport, accessibility and culturally governed participation remain practical design and evaluation questions.',
]
LIMITS = ('AI assessed; human approved for publication. This rapid review inspected selected methods and retained source evidence; it is not an independent systematic risk-of-bias or GRADE assessment. '
          'Age ranges often extend below 16 or above 24. Source access, age fit, masking, attrition and transfer limits remain attached to each paper. '
          'Article crosses describe study design, not benefit, certainty or human clinical review. Reviews do not inherit the crosses of their included trials.')
PATTERNS = {'anxiety': r'\banxiety\b', 'ptsd': r'\b(?:c?ptsd|post[ -]?traumatic stress)\b',
 'trauma': r'\b(?:trauma|traumatic|posttraumatic|post-traumatic)\b', 'depression': r'\bdepress(?:ion|ive)\b',
 'psychosis': r'\bpsychosis|psychotic\b', 'schizophrenia': r'\bschizophren', 'bipolar': r'\bbipolar\b',
 'adhd': r'\badhd\b|attention.deficit', 'stress': r'(?<!traumatic )\bstress\b', 'sleep': r'\b(?:sleep|insomnia)\b',
 'substance-use': r'\b(?:substance use|substance-use|addiction|drug use|alcohol)\b', 'autism': r'\bautis',
 'eating-disorders': r'\b(?:eating disorder|anorexia|bulimia)'}

def normal(value):
    return re.sub(r'\W+', '', str(value).lower())

def render():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA, 'Review revision changed'
    batch = json.loads(SOURCE.read_text())
    selected = [r for r in batch['records'] if r['reviewReady'] and r['quality']['tier'] in ['qualified', 'priority'] and not r.get('duplicateOf')]
    assert len(selected) == 105
    # Recheck all original corpus sources, including exact-title legacy matches.
    base = json.loads((ROOT/'research/informatics/youth-20261102/baseline-corpus.json').read_text())
    current = {}
    for file, digest in base['sourceSHA256'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == digest, 'Re-deduplicate changed corpus: '+file
        for row in json.loads((ROOT/file).read_text())['records']:
            current[row['id']] = {**current.get(row['id'], {}), **row}
    historical_citations = {}
    for collection in ['sound', 'vision', 'language']:
        for chapter in json.loads((SITE/f'library-data/{collection}-edition.json').read_text())['chapters']:
            for topic in chapter['topics']:
                for paper in topic['papers']:
                    if isinstance(paper.get('citationCount'), int):
                        ident = paper['recordId']
                        historical_citations[ident] = max(historical_citations.get(ident, 0), paper['citationCount'])
    records, ages, communities, classifications = [], [], [], []
    access, conditions = {}, {}
    for r in selected:
        pmid = r['pmid']; rid = ALIASES.get(pmid, r['id'])
        matches = [v['id'] for v in current.values() if (v.get('pmid') and str(v['pmid']) == pmid) or (r.get('doi') and normal(v.get('doi', '')) == normal(r['doi'])) or normal(v['title']) == normal(r['title'])]
        assert (rid in matches) if pmid in ALIASES else not matches, (pmid, matches)
        topics = r['topics']; collections = []
        if 'EMDR' in topics: collections.append('vision')
        if any(t in topics for t in ['Music / creative participation', 'Sound / brain entrainment']) or pmid in ARTS: collections.append('sound')
        if not collections or pmid in ARTS: collections.append('language')
        collections = list(dict.fromkeys([*current.get(rid, {}).get('collections', []), *collections]))
        groups = (['children'] if pmid in CHILDREN else []) + (['adolescents'] if pmid in ADOLESCENTS else []) + ['youth'] + ([] if pmid in NO_ADULTS else ['adults']) + (['older-adults'] if pmid in OLDER else [])
        fit = r['population']['ageFit']
        age = {'recordId': rid, 'collections': collections, 'groups': groups, 'reportedAges': r['population']['age'],
          'youthScope': 'focus' if fit in ['direct', 'target-range', 'target-centred', 'younger-overlap', 'broader-institution-youth'] else 'included',
          'youthSourceNote': 'The existing youth filter includes adolescents and source-defined young people. It does not establish a separate 16–24 estimate.',
          'ageFit': fit, 'sourceURL': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/', 'basisType': 'selected-methods-ai-editorial-assessment',
          'classificationNote': 'Broad groups describe source-reported populations or review scope. Eligibility and study mean ages are not observed participant ranges; read the reported-age wording.'}
        ages.append(age)
        community = r['population']['community']; community = community if isinstance(community, list) else [community]
        cm = [{'categoryId': c, 'relation': 'direct'} for c in community if c in ['refugees', 'housing', 'defence', 'indigenous', 'domestic-violence', 'war-trauma']]
        communities.append({'recordId': rid, 'matches': cm})
        fields = {'title': r['title'], 'findings': r['findings']}
        tags = []
        for cid, pattern in PATTERNS.items():
            hits = [{'field': k, 'text': v, 'matchedText': m.group()} for k, v in fields.items() if (m := re.search(pattern, v, re.I))]
            if hits: tags.append({'id': cid, 'evidence': hits, 'scope': 'Studied topic or reported outcome, including null findings; no diagnosis, benefit or subgroup membership is inferred.'})
        conditions[rid] = tags
        urls = list(dict.fromkeys(([f'https://doi.org/{r["doi"]}'] if r.get('doi') else []) + [s['url'] for s in r['sources']]))
        access[rid] = {'doi': r.get('doi'), 'pmid': pmid, 'urls': urls, 'journalLabel': r['journal']}
        randomized = r['design']['category'] == 'randomized-report'
        blind = 'double' if r['design']['doubleBlindingVerified'] else 'single' if pmid in SINGLE_MASKED else 'not-confirmed' if randomized else 'not-applicable'
        category = ('double-blind-rct' if blind == 'double' else 'single-blind-rct' if blind == 'single' else 'rct-masking-unclear') if randomized else r['design']['category']
        score = 3 if blind == 'double' else 2 if randomized else 1
        assert score == r['design']['proposedCrosses']
        design = {'randomized-report': 'Randomized controlled trial report', 'review': 'Review / synthesis', 'qualitative': 'Qualitative study', 'observational': 'Observational study'}[r['design']['category']]
        if randomized: design += ' · '+{'double': 'verified double blind', 'single': 'single masked role verified', 'not-confirmed': 'double blinding not verified'}[blind]
        elif 'meta-analy' in r['title'].lower(): design = 'Systematic review / meta-analysis'
        sources = [s['url'] for s in r['sources']]
        source_access = {**r['sourceAccess'], 'level': r['sourceAccess']['scope'], 'sourceURLs': sources, 'retrievedAt': DATE}
        trust = {'score': score, 'displayCrosses': score, 'scope': 'Paper', 'status': 'assessed', 'ratingScale': 'library-trust-factor/1-3',
          'randomizedControlledTrial': randomized, 'peerReviewed': True, 'blinding': blind, 'maskingRoles': r['design']['masking'],
          'basis': r['design']['description'], 'sources': sources, 'assessedAt': DATE, 'reviewScope': 'selected-methods',
          'humanApproval': False, 'reviewLimit': 'AI source-based design assessment; human content sign-off is recorded separately.'}
        assessment = {'category': category, 'isRandomizedControlledTrial': randomized, 'peerReviewed': True, 'basis': r['design']['description'], 'maskingRoles': r['design']['masking'], 'sources': sources, 'reviewScope': 'selected-methods'}
        evidence = {'evidenceType': design, 'intervention': r['intervention']['descriptionAndDose'], 'provider': r['intervention']['provider'],
          'comparator': r['intervention']['comparator'], 'sample': r['population']['sample'], 'population': r['population'],
          'measurement': 'Specific outcomes are reported in the finding and retained appraisal.', 'trialId': r['randomizedStudyUnitId'] or r['studyFamilyId'],
          'trialRegistries': r['trialRegistries'], 'registryScope': r['registryScope'], 'relationshipNote': r['labRelevance']['implication']+' '+r['labRelevance']['transferLimit'],
          'measuredGenes': [], 'pathways': [], 'quality': r['quality'], 'maskingRoles': r['design']['masking'], 'harmsAndAttrition': r['harmsAndAttrition'], 'sourceAccess': source_access}
        record = {'id': rid, 'aliases': [r['id']] if rid != r['id'] else [], 'title': r['title'], 'authors': r['authors'], 'year': r['year'], 'journal': r['journal'], 'doi': r.get('doi'), 'pmid': pmid, 'urls': urls,
          'collections': collections, 'categories': ['Methods & context'] if r['design']['category'] in ['review', 'qualitative', 'observational'] else ['Clinical & self-report'],
          'targets': [], 'neurotransmitters': [], 'conditions': [t['id'] for t in tags], 'conditionEvidence': tags, 'ageGroups': groups, 'agePopulation': age, 'communityMatches': cm,
          'kind': 'Review' if r['design']['category'] == 'review' else 'Journal article', 'design': design, 'origin': 'approved-youth-20261102', 'priority': 1, 'quotes': [],
          'finding': r['findings'], 'summary': r['findings'], 'limitation': r['limitations'], 'evidence': evidence, 'studyDesignAssessment': assessment, 'paperTrustAssessment': trust,
          'jadad': {'status': 'unassessed' if randomized else 'not-applicable', 'score': None},
          'citationCount': current.get(rid, {}).get('citationCount') if isinstance(current.get(rid, {}).get('citationCount'), int) else historical_citations.get(rid),
          'citationSnapshot': current.get(rid, {}).get('citationSnapshot') or {'count': None, 'source': None, 'retrievedAt': None, 'status': 'not-assessed'},
          'publicationApproval': {'status': 'approved-for-publication', 'approvedBy': 'Ben Korbel', 'approvedAt': DATE, 'assessment': 'AI assessed; human approved for publication',
            'scope': 'Approved research summaries for Informatics staging. No clinical peer review, treatment recommendation, partner endorsement or production release is claimed.', 'sourceRevision': r['revision'], 'sourceContentHash': r['contentHash']},
          'sourceAccess': source_access, 'studyFamilyId': r['studyFamilyId'], 'randomizedStudyUnitId': r['randomizedStudyUnitId'], 'institutions': r['institutions'], 'institutionEvidence': r['institutionEvidence'],
          'researchTopics': topics, 'labRelevance': r['labRelevance'], 'publicationStatus': r['publicationStatus'], 'noticeSummary': r['noticeSummary'], 'approaches': topics, 'assessmentRecord': copy.deepcopy(r)}
        records.append(record)
        classifications.append({'recordId': rid, 'sourceContentHash': r['contentHash'], 'collections': collections, 'age': age, 'conditions': tags, 'communityMatches': cm, 'maskingClass': blind,
          'targetBasis': 'No directly measured gene or molecular target established in the retained assessment; no target inferred from an intervention or mechanistic claim.'})
    counts = {'publications': len(records), 'new': len(records)-len(ALIASES), 'existingMatches': len(ALIASES), 'designs': dict(Counter(r['design']['category'] for r in selected)),
      'paperCrosses': dict(Counter(str(r['paperTrustAssessment']['score']) for r in records)), 'studyFamilies': len({r['studyFamilyId'] for r in records}),
      'randomizedStudyUnits': len({r['randomizedStudyUnitId'] for r in records if r['paperTrustAssessment']['randomizedControlledTrial']}),
      'institutions': dict(Counter(i for r in records for i in r['institutions'])), 'collections': {c: sum(c in r['collections'] for r in records) for c in ['sound', 'vision', 'language']},
      'sourceAccess': dict(Counter(r['sourceAccess']['level'] for r in records)), 'ageFit': dict(Counter(r['agePopulation']['ageFit'] for r in records)), 'labRelationship': dict(Counter(r['labRelevance']['relationship'] for r in records))}
    overview = '105 approved publications: 47 randomized trial reports, 42 reviews or syntheses, eight qualitative studies and eight observational studies. They comprise 100 additions and five existing-library matches. The 47 randomized reports represent 42 conservatively linked trial units. Cross-listed papers and institutional groups overlap; publication counts are not independent participants or trials.'
    chapters = {}
    dv = json.loads((SITE/'library-data/domestic-violence-expansion.json').read_text())
    for c in ['sound', 'vision', 'language']:
        rows = [r for r in records if c in r['collections']]; topics = []
        for tid, title, kind, key in TOPICS:
            members = [r for r in rows if kind == 'lab' or kind == 'institution' and key in r['institutions'] or kind == 'topic' and key in r['researchTopics'] or kind == 'making' and r['pmid'] in MUSIC_MAKING]
            if not members and not (tid == 'mantra-youth' and c == 'language'): continue
            assessment = assess_topic([{**r, 'evidence': {**r['evidence'], 'trialId': r['randomizedStudyUnitId'] or r['studyFamilyId']}} for r in members])
            assessment.update(assessedAt=DATE, basis='Existing minimum-three-study design rubric. Known linked reports count once by trial unit or study family. Overlap between reviews remains a limitation; no pooled efficacy or certainty grade is assigned.')
            paragraphs = ['Chapter Trust Factor counts distinct qualifying study families. Paper crosses identify each publication’s design. Neither score establishes effectiveness; related reports and overlapping reviews are not independent replications.']
            if kind == 'lab': paragraphs += LAB_COPY
            if kind == 'institution': paragraphs += [f'Institutional association with {key} is retained from source evidence. These papers span youth needs and intervention research; association does not mean that every study tested the named lab activities, or that the institution endorses Universal Mind. Partnerships remain prospective.']
            if tid == 'music-making-youth': paragraphs += ['These papers examine active or mixed music therapy, including facilitated group participation. They do not establish benefits of electronic music production, DJ training or fashion education. Listening-only studies are listed separately.']
            if tid == 'sound-entrainment-youth': paragraphs += ['Evidence for neural entrainment is inconsistent. Sound exposure, musical engagement, measured brain activity and clinical benefit are distinct outcomes.']
            if tid == 'mantra-youth': paragraphs += ['No new mantra paper met this batch’s selected-methods qualification threshold. Abstract-only candidates remain outside the approved library addition. Transcendental meditation has its own distinct section; absence from this batch is not proof of ineffectiveness.']
            if tid == 'emdr-youth': paragraphs += ['The studies concern specified clinical EMDR protocols and populations. Audiovisual stimulation, bilateral experiences and self-guided lab activities are not equivalent to clinical EMDR.']
            if tid == 'transcendental-meditation-youth': paragraphs += ['The two reports in this batch come from the same university cohort and count as one trial unit. Their mixed-age sample does not isolate ages 16–24.']
            topics.append({'id': tid, 'title': title, 'description': f'{len(members)} publications in {c.title()}; cross-listing is not additional evidence. Reported ages and source limits remain visible in each record.', 'paragraphs': paragraphs,
              'trustScore': assessment['score'], 'trustAssessment': assessment, 'papers': [{**{k: copy.deepcopy(v) for k, v in r.items() if k != 'assessmentRecord'}, 'recordId': r['id'], 'id': c+'-'+tid+'-'+r['id'], 'sourcePages': []} for r in members]})
        chapters[c] = {'number': dv['chapters'][c]['number']+1, 'title': 'AI Guided Youth Research', 'overview': [overview, LIMITS], 'topics': topics}
    result = {'schema': 'approved-youth/v1', 'approvedAt': DATE, 'sourceBatch': SOURCE.name, 'sourceSHA256': SOURCE_SHA, 'paperTrustFactorPolicy': POLICY,
      'counts': counts, 'overview': overview, 'limitations': LIMITS, 'labFraming': LAB_COPY, 'records': records, 'chapters': chapters, 'publicationAccess': access,
      'conditions': conditions, 'ages': ages, 'communities': communities, 'classificationAudit': classifications}
    return json.dumps(result, ensure_ascii=False, indent=2)+'\n'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); args = parser.parse_args(); text = render()
    if args.check: assert OUT.read_text() == text, 'Youth integration differs from immutable source and classification rules'
    else: OUT.write_text(text)
    print(json.dumps(json.loads(text)['counts']))
