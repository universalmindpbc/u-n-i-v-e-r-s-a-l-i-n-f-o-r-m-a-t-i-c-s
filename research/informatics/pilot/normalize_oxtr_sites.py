"""Reproducible harmonisation of the saved OXTR imports; preserves source records.

Exact reference intervals group locations, never independent biological evidence.
Only published, recovered sequences are matched; no new motif scan is performed.
"""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUP = ROOT / 'Informatics/pilot-v1/oxtr-tfbs-v1'
VERSION = 'oxtr-human-sites-2026-10-19-v1'
HUMAN = 'Homo sapiens'


def read(name):
    b = (SUP / name).read_bytes()
    return json.loads(gzip.decompress(b) if name.endswith('.gz') else b)


def write(name, value):
    b = (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    (SUP / name).write_bytes(gzip.compress(b, mtime=0) if name.endswith('.gz') else b)


def table(name, rows, fields=None):
    with (SUP / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False, separators=(',', ':'))
                        if isinstance(v, (list, dict)) else v for k, v in r.items()})


def exact_matches(sequence, dna):
    """All exact forward/reverse matches in this window, including overlaps."""
    reverse = sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
    matches = []
    for strand, query in [('+', sequence), ('-', reverse)]:
        offset = 0
        while True:
            i = dna.find(query, offset)
            if i < 0:
                break
            matches.append((i, i + len(query), strand))
            offset = i + 1
    return matches


# Editorial appraisal of the recorded observation, not a ranking of labels.
# Laboratory tested includes negative and mixed outcomes and grouped reports.
LAB = set(range(3, 17)) | set(range(21, 42)) | {44, 45}
BINDING = {3, 4, 5, 7, 8, 10, 12, 13, 15, 16, 28, 30, 31, 33, 34, 39, 40, 41, 44}
FUNCTIONAL = {6, 7, 10, 11, 12, 14, 15, 16, 21, 22, 24, 26, 37, 38, 39, 40, 41, 45}
ANNOTATION = {1, 2, 17, 18, 19, 20, 42, 43}
CONTEXT = {9, 14, 27, 34, 35, 36}
CATEGORIES = [
    dict(id='laboratory_tested', label='Laboratory tested', definition='The record reports a laboratory assay. Includes positive, mixed and negative results; a group-level test does not resolve every member.'),
    dict(id='binding_evidence', label='Binding evidence', definition='Reported DNA binding or regional occupancy, at the resolution and in the system stated by the source.'),
    dict(id='functional_evidence', label='Functional evidence', definition='An element, region or element-based reporter has an experimental regulatory response. This does not imply every nucleotide or predicted motif was independently validated.'),
    dict(id='annotation_prediction', label='Annotation or prediction', definition='Sequence or database annotation; direct binding is not established by the annotation alone.'),
    dict(id='regulatory_context', label='Regulatory context', definition='Evidence concerning gene regulation without an independently resolved binding site in this record.')]


def categories(r):
    n = int(r['id'][4:])
    return [k for k, ids in [('laboratory_tested', LAB), ('binding_evidence', BINDING),
                            ('functional_evidence', FUNCTIONAL), ('annotation_prediction', ANNOTATION),
                            ('regulatory_context', CONTEXT)] if n in ids]


def overlap(a, b):
    if any(a[k] is None or b[k] is None for k in ['assembly', 'chromosome', 'start_0based', 'end_exclusive']):
        return 0
    if (a['species'], a['assembly'], a['chromosome']) != (b['species'], b['assembly'], b['chromosome']):
        return 0
    return max(0, min(a['end_exclusive'], b['end_exclusive']) - max(a['start_0based'], b['start_0based']))


def build():
    literature = read('literature-evidence.json')
    reference = read('reference-dna.json')
    dna = reference['dna_plus_strand']
    raw_files = ['literature-evidence.json', 'jaspar-predictions.json.gz', 'encode-clusters.json', 'oreganno-regions.json', 'reference-dna.json']
    if (SUP / 'study-additions-v1/braintf-peaks.json').exists():
        raw_files.append('study-additions-v1/braintf-peaks.json')
    raw_hashes = {f: hashlib.sha256((SUP / f).read_bytes()).hexdigest() for f in raw_files}
    records, mapping = [], []

    def base(rid, source, species=HUMAN):
        return dict(observation_id=source+':'+rid, source_collection=source, source_record_id=rid,
                    species=species, gene='OXTR' if species == HUMAN else 'Oxtr',
                    feature_type=None, factor_reported=None, assembly=None, chromosome=None,
                    start_0based=None, end_exclusive=None, strand=None, sequence_reported=None,
                    evidence_categories=[], assay_methods=None, context_reference=None,
                    original_position=None, mapping_status='Unresolved', source_url=None,
                    pmid=None, matrix_id=None, original_score=None, score_semantics=None,
                    default_display_threshold_pass=None, outcome=None, limitations=None,
                    reference_interval_id=None)

    for r in literature:
        o = base(r['id'], 'literature', r['species'])
        o.update(feature_type=r['evidence_kind'], factor_reported=r['factor'],
                 evidence_categories=categories(r), assay_methods=r['methods'],
                 original_position=r['reported_position'], sequence_reported=r['reported_sequence'],
                 pmid=r['pmid'], source_url=r['source_url'], outcome=r['finding'], limitations=r['limitation'],
                 context_reference='literature-evidence.json#'+r['id'])
        if r['start_0based'] is not None:
            o.update(assembly='GRCh38', chromosome=r['chromosome'], start_0based=r['start_0based'],
                     end_exclusive=r['end_exclusive'], mapping_status='Source regional annotation')
        if r['species'] == HUMAN and r['reported_sequence']:
            matches = exact_matches(r['reported_sequence'], dna)
            m = dict(record_id=r['id'], pmid=r['pmid'], sequence=r['reported_sequence'],
                     reference_file='reference-dna.json', reference_sha256=raw_hashes['reference-dna.json'],
                     method='Exact sequence and reverse-complement search within the saved OXTR window',
                     window_start_0based=reference['start_0based'], window_end_exclusive=reference['end_exclusive'],
                     matches_in_window=len(matches), assembly='GRCh38', chromosome='chr3',
                     start_0based=None, end_exclusive=None, strand=None,
                     limitation='Uniqueness is local to this reference window, not genome-wide. Mapping a published element does not establish a new binding experiment.')
            if len(matches) == 1:
                a, b, strand = matches[0]
                mapped = dict(start_0based=reference['start_0based']+a,
                              end_exclusive=reference['start_0based']+b, strand=strand)
                m.update(mapped)
                o.update(**mapped, assembly='GRCh38', chromosome='chr3', mapping_status='Exact published-sequence match; unique within saved window')
            mapping.append(m)
        records.append(o)

    for source, filename in [('encode', 'encode-clusters.json'), ('oreganno', 'oreganno-regions.json'), ('jaspar', 'jaspar-predictions.json.gz')]:
        for r in read(filename):
            o = base(r['id'], source)
            o.update(assembly='GRCh38', chromosome=r['chromosome'], start_0based=r['start_0based'],
                     end_exclusive=r['end_exclusive'], factor_reported=r.get('target', r.get('factor')),
                     strand=r.get('strand'), mapping_status='Source GRCh38 interval',
                     context_reference=filename+'#'+r['id'])
            if source == 'jaspar':
                o.update(feature_type='predicted_motif', evidence_categories=['annotation_prediction'],
                         sequence_reported=r['sequence_oriented'], matrix_id=r['matrix_id'],
                         original_score=r['score'], score_semantics='UCSC score: -100 log10(p), capped at 1000',
                         default_display_threshold_pass=r['default_display_threshold_pass'],
                         source_url='https://jaspar.elixir.no/matrix/'+r['matrix_id']+'/',
                         outcome='Sequence match; binding and functional activity untested by this prediction')
            elif source == 'encode':
                o.update(feature_type='occupancy_region', evidence_categories=['laboratory_tested', 'binding_evidence'],
                         assay_methods='ChIP-seq cluster', context_reference='encode-provenance.json#'+r['id'],
                         original_score=r['score'], score_semantics='Source UCSC ENCODE cluster score; not comparable to JASPAR score',
                         source_url='https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=encRegTfbsClustered',
                         outcome='Protein association with a region; factor and cell contexts retained in provenance')
            else:
                o.update(feature_type='annotated_region', evidence_categories=['annotation_prediction'],
                         source_url='https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=oreganno',
                         outcome='Reused regional annotation; experimental interpretation requires the original source',
                         limitations='Annotation strand is retained in the original import, not interpreted as motif orientation')
            records.append(o)

    if 'study-additions-v1/braintf-peaks.json' in raw_files:
        for r in read('study-additions-v1/braintf-peaks.json'):
            o = base(r['id'], 'braintf')
            o.update(feature_type='occupancy_region', factor_reported=r['factor'],
                     assembly='GRCh38', chromosome=r['chromosome'],
                     start_0based=r['start_0based'], end_exclusive=r['end_exclusive'],
                     evidence_categories=['laboratory_tested', 'binding_evidence'],
                     assay_methods=r['methods'], pmid=r['pmid'],
                     context_reference='study-additions-v1/braintf-peaks.json#'+r['id'],
                     mapping_status='Source GRCh38 peak interval',
                     source_url=r['source_url'], original_score=r['score'],
                     score_semantics=r['score_semantics'], outcome=r['finding'],
                     limitations=r['limitation'])
            records.append(o)

    # One spatial interval, multiple observations. Factor, motif model, strand,
    # experiment and evidence type remain on the observation; no observations lost.
    intervals = {}
    for o in records:
        if o['start_0based'] is None:
            continue
        key = (o['species'], o['assembly'], o['chromosome'], o['start_0based'], o['end_exclusive'])
        if key not in intervals:
            digest = hashlib.sha256('|'.join(map(str, key)).encode()).hexdigest()[:16]
            intervals[key] = dict(reference_interval_id='OXI_'+digest, species=key[0], assembly=key[1],
                                  chromosome=key[2], start_0based=key[3], end_exclusive=key[4],
                                  observation_count=0, source_collections=[])
        locus = intervals[key]
        locus['observation_count'] += 1
        if o['source_collection'] not in locus['source_collections']:
            locus['source_collections'].append(o['source_collection'])
        o['reference_interval_id'] = locus['reference_interval_id']
    assert len({r['observation_id'] for r in records}) == len(records)
    literature_normal = [r for r in records if r['source_collection'] == 'literature']
    overlaps = []
    for a in literature_normal:
        if a['species'] != HUMAN or a['start_0based'] is None:
            continue
        for b in records:
            if b['source_collection'] == 'literature':
                continue
            bp = overlap(a, b)
            if bp:
                overlaps.append(dict(literature_observation_id=a['observation_id'], other_observation_id=b['observation_id'],
                                     overlap_bp=bp, relationship='same_reference_interval' if a['reference_interval_id'] == b['reference_interval_id'] else 'spatial_overlap',
                                     other_source=b['source_collection'], default_display_threshold_pass=b['default_display_threshold_pass'],
                                     interpretation='Spatial relationship only; not independent validation or proof of the same binding event'))
    relationships = [
        dict(a='literature:TFBS014', b='literature:TFBS013', relationship='same_reported_element', basis='Both reports investigate US-2; the later reporter record retains its own assay context.'),
        dict(a='literature:TFBS007', b='literature:TFBS006', relationship='reported_subregion', basis='The later -712 to -692 region is inside the earlier -851 to -656 region under the reported promoter numbering; genomic anchor remains unresolved.'),
        dict(a='literature:TFBS011', b='literature:TFBS010', relationship='distinct_elements_in_shared_region', basis='Two cooperating elements in the -85 to -65 regulatory region; preserve both, not duplicates.'),
        dict(a='literature:TFBS043', b='oreganno:OREG1492306', relationship='annotation_reused', basis='The paper cites this existing annotation; not a new independent experiment.')]
    interval_rows = sorted(intervals.values(), key=lambda r: (r['species'], r['assembly'], r['chromosome'], r['start_0based'], r['end_exclusive']))
    write('normalized-observations.json.gz', records)
    table('normalized-observations.csv', records)
    write('reference-intervals.json.gz', interval_rows)
    table('reference-intervals.csv', interval_rows)
    write('sequence-mapping.json', mapping)
    table('sequence-mapping.csv', mapping)
    write('literature-overlaps.json', overlaps)
    table('literature-overlaps.csv', overlaps)
    write('evidence-relationships.json', relationships)
    table('evidence-relationships.csv', relationships)
    write('normalized-literature.json', literature_normal)
    table('normalized-literature.csv', literature_normal)
    with (SUP / 'mapped-literature.bed').open('w') as bed:
        for r in literature_normal:
            if r['start_0based'] is not None:
                bed.write('\t'.join(map(str, [r['chromosome'], r['start_0based'], r['end_exclusive'], r['source_record_id'], 0, r['strand'] or '.']))+'\n')
    human = [r for r in literature_normal if r['species'] == HUMAN]
    summary = dict(total_observations=len(records), literature_records=len(literature), human_literature_records=len(human),
                   distinct_reference_intervals=len(intervals), colocated_interval_groups=sum(r['observation_count'] > 1 for r in interval_rows),
                   retained_observations=len(records), discarded_observations=0,
                   mapped_human_literature_records=sum(r['start_0based'] is not None for r in human),
                   exact_published_sequence_mappings=sum(m['matches_in_window'] == 1 for m in mapping),
                   literature_database_overlap_pairs=len(overlaps),
                   human_evidence_categories=dict(Counter(k for r in human for k in r['evidence_categories'])),
                   unique_confirmed_human_site_total=None)
    project = dict(version=VERSION, date='2026-10-19', notion_url='https://www.notion.so/3fd7b9c99c56815a8b84e6839dc1d3ed', status='First harmonisation pass complete; legacy site recovery in progress',
                   question='Which human OXTR binding elements and regional evidence are reported, where are they located, and which records overlap or reuse evidence?',
                   summary=summary, categories=CATEGORIES, input_sha256=raw_hashes,
                   coordinate_system='GRCh38, 0-based start and exclusive end. Historical TSS/ATG/CpG positions retained verbatim until anchored.',
                   normalisation='Common columns and stable identifiers. Original source records remain unchanged. Source scores retain their own scales; missing values remain null.',
                   deduplication='Group exact intervals by species, assembly, chromosome, start and end. Preserve all observations, strands, factors and experiments. Interval identity is not biological-site identity.',
                   overlap_scope='Mapped human literature records versus imported ENCODE, ORegAnno, JASPAR and screened BrainTF observations. Not an all-pairs motif network. Prediction display-threshold flags are retained.',
                   species_policy='Human and animal observations remain distinguishable; no cross-species deduplication or coordinate projection.',
                   atlas_status='No DNA-to-M9 mapping or statistical comparison performed. Native positions remain the basis for future defined comparisons.',
                   original_fifteen=dict(pmid='16569740', species=HUMAN, reported_count=15,
                                         groups={'AP1':3,'C/EBP':9,'NF-kappaB':3}, individual_coordinates_recovered=0,
                                         status='Full-text tables and figures still required',
                                         evidence='Human promoter study reports computer analysis, EMSA and mutation assays; effects differ by factor and combination.',
                                         source='https://academic.oup.com/jcem/article/91/6/2317/2843585'),
                   priorities=[
                       'Recover the 2006 paper and any supplementary sequence or oligonucleotide tables through library access, publisher purchase or an author-provided copy.',
                       'Resolve the 2006 and 2011 promoter anchors and exact tested sequences before mapping their sites to GRCh38.',
                       'Recover GABP/CRE-like and ERE-like motif extents from human primary studies; preserve assay-specific outcomes.',
                       'Inspect GSE197160 processed tracks for human myometrial HIF1A; bigWig signal is not an independently recovered peak interval.',
                       'Expand human primary-source screening and deduplicate with sequence, factor, experiment and cell-context checks.',
                       'Develop the original DNA line, full-position circle and M9 view once mapping rules are defined.'],
                   access_routes=['University or public-library document delivery', 'Publisher purchase after checking included tables and supplements', 'Author manuscript or original site-data request; no request sent'],
                   sources=[{'url':'https://pubmed.ncbi.nlm.nih.gov/10221779/','use':'Published US-1 sequence and laboratory findings'},
                            {'url':'https://pubmed.ncbi.nlm.nih.gov/10527846/','use':'Published US-2 sequence and MAFF binding'},
                            {'url':'https://pubmed.ncbi.nlm.nih.gov/21734268/','use':'Shared 20-base human response region'},
                            {'url':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE197160','use':'Human myometrial HIF1A source data located; no peak file imported'}])
    write('human-site-project.json', project)
    table('human-evidence-summary.csv', [dict(category=c['label'], records=summary['human_evidence_categories'].get(c['id'], 0), unit='literature evidence records; categories overlap') for c in CATEGORIES])
    catalogue = read('catalogue.json')
    catalogue['harmonisation'] = dict(version=VERSION, summary=summary, categories=CATEGORIES,
        project='human-site-project.json', observations='normalized-observations.json.gz',
        intervals='reference-intervals.json.gz', overlaps='literature-overlaps.json',
        literature='normalized-literature.json', bed='mapped-literature.bed')
    lookup = {r['source_record_id']:r for r in literature_normal}
    for r in catalogue['evidence']:
        n = lookup[r['id']]
        r['evidence_categories'] = n['evidence_categories']
        r['normalized_location'] = {k:n[k] for k in ['assembly','chromosome','start_0based','end_exclusive','strand','mapping_status','reference_interval_id']}
    write('catalogue.json', catalogue)
    existing_readme = (SUP/'README.txt').read_text()
    later_sections = existing_readme[existing_readme.index('\nCoordinate experiment —'):] if '\nCoordinate experiment —' in existing_readme else ''
    readme = existing_readme.split('\nHuman site harmonisation')[0]
    (SUP/'README.txt').write_text(readme+'\nHuman site harmonisation — 19 October 2026\n\n'+
        'Read human-site-project.json (or human-site-project-fields.csv) for scope and open recovery tasks. normalized-observations combines every imported record under one column schema, while the original tables remain intact. Evidence categories are multi-valued: laboratory tested includes positive, negative and mixed results; functional evidence includes regions and element-based reporters. The category counts are evidence records, not unique confirmed sites.\n\n'+
        'reference-intervals groups only identical genomic intervals; every observation remains linked to its source. Shared coordinates do not establish the same factor, experiment, or biological function. literature-overlaps contains spatial links from the mapped human literature records to all imported database records. It is not an all-pairs network or independent corroboration. evidence-relationships records known reuse and distinct cooperating elements.\n\n'+
        'US-1 and US-2 each match once in the saved human OXTR window on the reverse strand. These are computed reference placements of published sequences, not new experiments or a claim of genome-wide uniqueness. mapped-literature.bed contains two sequence elements and one broad annotation; use normalized-literature to distinguish them.\n\n'+
        'The original human 15-site report remains an aggregate of 3 AP1, 9 C/EBP and 3 NF-kappaB sites. Its individual sequences and historical coordinate anchors remain unresolved. No total of unique confirmed human OXTR sites or DNA-to-Atlas comparison is asserted.\n' + later_sections)
    print(json.dumps(summary, indent=2))
    return project


if __name__ == '__main__':
    build()
