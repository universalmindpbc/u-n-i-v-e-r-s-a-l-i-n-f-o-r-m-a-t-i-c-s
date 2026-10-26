"""Import the reviewed BrainTF subset and OXTR enhancer paper.

Run before normalize_oxtr_sites.py, annotate_oxtr_binding_outcomes.py and
package_oxtr_catalogue.py. Inputs are archived locus extracts, not raw reads.
"""
import csv
import json
from pathlib import Path
import shutil
from csv_records import write_record, read_record

ROOT = Path(__file__).resolve().parents[3]
REVIEW = ROOT / 'research/informatics/oxtr-study-review-20261021'
SUP = ROOT / 'Informatics/pilot-v1/oxtr-tfbs-v1'
OUT = SUP / 'study-additions-v1'
OUT.mkdir(exist_ok=True)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def retained(r):
    return (r['assay_class'] == 'TF_or_chromatin_protein'
            and r['replication_qc_statuses'] == ['pass']
            and len(r['qc_rows']) == 1
            and r['minimum_FRiP'] is not None and r['minimum_FRiP'] >= .01
            and r['author_HOT_statuses'] == ['NOT']
            and not r['source_metadata_conflict'])


sources = {r['file']: r for r in read(REVIEW / 'retrieval.json') if r.get('status') == 'retrieved'}
locus = read(REVIEW / 'all-braintf-locus-annotated.json')
qc_rows = read(REVIEW / 'braintf-qc-rows.json')
qc_by_row = {r['source_row']: r for r in qc_rows}
selected = sorted(filter(retained, locus), key=lambda r: (r['start'], r['end'], r['target_canonical'], r['context']))
assert len(selected) == 126
assert len({(r['source_file'], r['source_row_1based']) for r in selected}) == 126
peaks = []
for r in selected:
    source = sources[r['source_file']]
    qc = qc_by_row[r['qc_rows'][0]]
    assert qc['Rep_test'] == 'pass'
    assert min(qc['FRiP_rep1'], qc['FRiP_rep2']) == r['minimum_FRiP']
    p = dict(
        id=f"BRAINTF_{r['context']}_{r['source_row_1based']}",
        study_id='braintf', pmid='38831039', species='Homo sapiens',
        gene='OXTR', assembly='GRCh38', chromosome=r['chrom'],
        start_0based=r['start'], end_exclusive=r['end'], strand=None,
        factor=r['target_canonical'], target_reported=r['name'],
        label='BrainTF · ' + r['context'], context=r['context'],
        cell_types=[r['context']], evidence_kind='occupancy_region',
        finding='Protein association detected across this region in the reported human brain ChIP-seq experiment.',
        methods='Author-processed ChIP-seq peak calls; replication and signal-quality review. Raw reads were not reanalysed.',
        limitation='Regional occupancy does not identify an exact binding sequence or establish an effect on native OXTR expression. Brain regions share four donors; contexts are not independent population replications.',
        source_url=source['url'], source_file=r['source_file'],
        source_sha256=source['sha256'], source_row_1based=r['source_row_1based'],
        score=r['score'], signal_value=r['signalValue'],
        source_p_value=r['pValue'], source_q_value=r['qValue'],
        source_peak_offset=r['peak'],
        score_semantics='Original narrowPeak-style fields retained verbatim; -1 denotes an unavailable source field. These values are not cross-platform confidence scores.',
        qc_sheet='Supplementary Table 2 / ChIP-seq',
        qc_source_url='https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41593-024-01658-8/MediaObjects/41593_2024_1658_MOESM4_ESM.xlsx',
        qc_rows=r['qc_rows'], replication_qc=r['replication_qc_statuses'][0],
        minimum_FRiP=r['minimum_FRiP'], author_HOT_status='NOT',
        promoter_overlap=r['proximal_promoter'],
        existing_ENCODE_overlaps=r['same_target_existing_ENCODE'],
        brain_cCRE_overlaps=r['brain_cCRE_overlaps'],
        normalized_location={'mapping_status':'Source GRCh38 region, 0-based start and exclusive end; no motif or contact base inferred.'})
    peaks.append(p)
write(OUT / 'braintf-peaks.json', peaks)
table(OUT / 'braintf-peaks.csv', peaks)
for r in locus:
    r['included_in_study_browser'] = retained(r)
write(OUT / 'braintf-locus-review.json', locus)
table(OUT / 'braintf-locus-review.csv', locus)
write(OUT / 'braintf-qc-rows.json', qc_rows)
table(OUT / 'braintf-qc-rows.csv', qc_rows)
enhancers = read(REVIEW / 'oxtr-enhancer-review.json')
for r in enhancers:
    r['coordinate_interpretation'] = 'Reported candidate-region coordinates retained verbatim; tested-construct boundaries are not interchangeable with the candidate interval.'
    r['TF_binding_result'] = 'No named-factor binding assay established by this reporter result.'
    if r['name'] == 'OCE7':
        r['construct_note'] = 'Table 1 luciferase-primer suffixes match GRCh38 [8757365,8758831), shorter than the reported candidate region. The mouse experiment uses separate Table 3 primers. No precise binding-site or geometry overlay is assigned.'
write(OUT / 'enhancer-regions.json', enhancers)
table(OUT / 'enhancer-regions.csv', enhancers)

metadata = dict(
    version='oxtr-study-additions-2026-10-21-v1', reviewed_on='2026-10-21',
    window=dict(assembly='GRCh38', chromosome='chr3', start_0based=8740380, end_exclusive=8779628),
    braintf=dict(
        records=len(peaks), targets=len({r['factor'] for r in peaks}),
        contexts=len({r['context'] for r in peaks}),
        promoter_records=sum(r['promoter_overlap'] for r in peaks),
        selection='One matching QC row; author replication test passes; both reported FRiP values at least 1%; overlapping author union regions labelled NOT-HOT; no identified target-label conflict.',
        FRiP_interpretation='The 1% cutoff is an explicit Atlas review filter, not a universal quality threshold or the authors’ replication-pass definition.',
        HOT_interpretation='Uses the authors’ union-region annotation; high-occupancy, conflicting and unassessed classifications are excluded.',
        count_unit='Peak observations across assay contexts, not unique sites or independent donors.',
        coordinate_conversion='Author GRanges starts were converted from 1-based to 0-based by subtracting one; exclusive ends equal the converted start plus GRanges width.',
        donor_limit='The experiments share four postmortem donors across regions. Repeated contexts do not establish population replication.',
        assay_limit='Assayed targets include chromatin-associated proteins and cofactors as well as transcription factors. ChIP-seq measures regional occupancy; it does not locate a contact nucleotide.',
        contextual_annotations='Overlaps with existing ENCODE and PsychSCREEN brain candidate regulatory regions are spatial context, not independent validation of OXTR regulation.',
        review_scope='13 author-packaged peak datasets and 226 additional public portal peak files inspected. The browser retains the conservative subset; all 390 locus observations, including histone marks, remain in the review download.',
        excluded_result='No nominal allele-specific result in the separately reviewed BrainTF table passed BH-adjusted P < 0.05 at this locus; none is presented as confirmed allele-specific binding.',
        sources=['https://doi.org/10.1038/s41593-024-01658-8',
                 'https://github.com/aanderson54/Loupe_BrainTF',
                 'https://chip-portal.hudsonalpha.org/']),
    enhancer=dict(
        pmid='39957428', doi='10.1093/hmg/ddaf022',
        result='OCE1, OCE5 and OCE7 enhanced reporter activity in mouse hypothalamic cells. The tested DNA was human. OCE7 also drove reporter activity in mouse olfactory bulb at P28 and P56; adult hypothalamic activity was not detected.',
        atlas_scope='OCE2, OCE6 and OCE7 reported regions intersect the current window. OCE2 and OCE6 were negative in the reporter assay. The window is unchanged.',
        boundary='Reporter activity does not establish a named transcription-factor binding site or causal regulation of native human OXTR.',
        coordinates='The reported OCE7 candidate interval and the Table 1 luciferase-primer span differ. Reported intervals and primer matches are retained separately, without a precise binding-site overlay.',
        source='https://pmc.ncbi.nlm.nih.gov/articles/PMC12056307/'),
    geometry_scope='The Studies comparison and data browser both include 128 observations: US-1, US-2 and 126 screened BrainTF occupancy regions. Region centres are plotting representatives, not precise binding sites. The enhancer paper remains functional literature evidence.',
    files=dict(peaks='study-additions-v1/braintf-peaks.json',
               peak_csv='study-additions-v1/braintf-peaks.csv',
               review='study-additions-v1/braintf-locus-review.json',
               enhancers='study-additions-v1/enhancer-regions.json'))
write(OUT / 'review.json', metadata)
write_record(OUT / 'review-fields.csv', metadata)
assert read_record(OUT / 'review-fields.csv') == metadata
hashes = read(REVIEW / 'source-file-hashes.json')
write(OUT / 'source-file-hashes.json', hashes)
table(OUT / 'source-file-hashes.csv', hashes)
retrieval = list(sources.values()) + read(REVIEW / 'braintf-portal-retrieval.json')
write(OUT / 'sources.json', retrieval)
table(OUT / 'sources.csv', retrieval)
shutil.copy2(REVIEW / 'BrainTF-LICENSE.txt', OUT / 'BrainTF-LICENSE.txt')
(OUT / 'README.txt').write_text(
    'OXTR study additions — 21 October 2026\n\n'
    'BrainTF: 126 screened ChIP-seq peak observations, 43 protein targets and 13 brain contexts. '
    'GRCh38, 0-based starts and exclusive ends. Records can overlap and share donors. '
    'These are regional occupancy measurements, not 126 distinct binding sites. '
    'Read review.json or the equivalent review-fields.csv for exact inclusion criteria and limitations. '
    'braintf-locus-review retains all 390 reviewed locus observations and the selection flag. '
    'Histone marks and rejected observations do not enter the study browser.\n\n'
    'The 2025 enhancer study contributes functional reporter evidence. Its reported candidate intervals '
    'and primer mappings remain separate in enhancer-regions. No precise TF binding position is inferred.\n\n'
    'The Studies comparison includes both sequence elements and all 126 screened BrainTF regions. The DNA window, ENCODE and JASPAR imports remain unchanged. '
    'All study data is included in the complete JSON/CSV archives. Source URLs, row numbers and SHA-256 hashes '
    'identify the reviewed public files. Raw sequencing reads and controlled individual-level data are not included.\n')

catalogue = read(SUP / 'catalogue.json')
publications = [p for p in catalogue['publications'] if p['pmid'] not in ('38831039', '39957428')]
for i, pmid in enumerate(('38831039', '39957428'), start=25):
    p = read(REVIEW / f'publication-{pmid}.json')
    publications.append(dict(
        id=f'TFP{i:02}', pmid=pmid, title=p['title'], authors=p['authorString'],
        doi=p['doi'], year=int(p['pubYear']), first_publication_date=p['firstPublicationDate'],
        journal=p['journalInfo']['journal']['title'], url=f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
        doi_url='https://doi.org/' + p['doi'], pmcid=p['pmcid'],
        source_review='Public processed data, publication, supplementary tables and source metadata reviewed',
        existing_register_id=None, trust_score=None,
        trust_status='Not assigned; evidence type and review depth reported',
        jadad='Not applicable: not a randomized clinical trial'))
literature = [r for r in read(SUP / 'literature-evidence.json') if r['id'] not in ('TFBS044', 'TFBS045')]
for rid, pmid, label, kind, finding, methods, position, limitation in [
    ('TFBS044', '38831039', 'BrainTF human brain occupancy', 'occupancy_region',
     '126 screened protein-occupancy observations overlap the OXTR window, representing 43 targets across 13 contexts. 114 overlap the defined proximal promoter.',
     'Human postmortem brain ChIP-seq; review of author replication tests, FRiP and HOT-region annotations.',
     'Individual GRCh38 peak intervals are in study-additions-v1/braintf-peaks.json.',
     metadata['braintf']['count_unit'] + ' ' + metadata['braintf']['donor_limit'] + ' ' + metadata['braintf']['assay_limit']),
    ('TFBS045', '39957428', 'OCE7 enhancer reporter evidence', 'functional_region',
     metadata['enhancer']['result'],
     'Human candidate enhancer DNA in mouse hypothalamic-cell luciferase assays and mouse reporter experiments.',
     'OCE7 reported hg38 chr3:8757366–8759184; candidate interval and tested construct boundaries retained separately.',
     metadata['enhancer']['boundary'] + ' ' + metadata['enhancer']['coordinates'])]:
    literature.append(dict(id=rid, pmid=pmid, species='Homo sapiens',
        factor='43 assayed protein targets' if rid == 'TFBS044' else 'Factor not resolved',
        label=label, evidence_kind=kind, methods=methods, finding=finding,
        reported_position=position, reported_sequence=None,
        reported_count=None, genome_build='GRCh38', chromosome=None,
        start_0based=None, end_exclusive=None, motif_strand=None, reference_transcript=None,
        coordinate_status='Individual regional observations retained separately; no single precise binding-site coordinate assigned',
        limitation=limitation, related_evidence=None,
        source_url=f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
        publication_id='TFP25' if rid == 'TFBS044' else 'TFP26'))
write(SUP / 'literature-evidence.json', literature)
table(SUP / 'literature-evidence.csv', literature)
write(SUP / 'publications.json', publications)
table(SUP / 'publications.csv', publications)
old = {r['id']: r for r in catalogue['evidence']}
catalogue['evidence'] = [old.get(r['id'], r) for r in literature]
catalogue['publications'] = publications
catalogue['summary'].update(publications=len(publications), literature_evidence_records=len(literature),
                            braintf_screened_observations=126, braintf_protein_targets=43)
catalogue['study_additions'] = metadata
catalogue['files']['study_peaks'] = 'study-additions-v1/braintf-peaks.json'
catalogue['files']['study_review'] = 'study-additions-v1/review.json'
write(SUP / 'catalogue.json', catalogue)
readme = SUP / 'README.txt'
readme.write_text(readme.read_text().split('\nStudy additions —')[0] +
                 '\nStudy additions — 21 October 2026\n\n' +
                 (OUT / 'README.txt').read_text().split('\n\n', 1)[1] +
                 '\nRebuild with add_oxtr_brain_studies.py, normalize_oxtr_sites.py, '
                 'annotate_oxtr_binding_outcomes.py and package_oxtr_catalogue.py, in that order. '
                 'The archived review inputs are in research/informatics/oxtr-study-review-20261021.\n')
print(json.dumps({'publications':len(publications), 'literature_records':len(literature), 'BrainTF_records':len(peaks)}))
