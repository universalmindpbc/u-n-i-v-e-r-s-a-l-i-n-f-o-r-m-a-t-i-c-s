"""Record source-supported binding outcomes without altering measured coordinates.

This is a scoped interpretation layer: two mapped literature elements, ENCODE
peak clusters and JASPAR motif predictions. It is not a new biological analysis.
"""
import csv
import gzip
import json
from pathlib import Path
from csv_records import write_record, read_record

ROOT = Path(__file__).resolve().parents[3]
SUP = ROOT / 'Informatics/pilot-v1/oxtr-tfbs-v1'

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')

def build():
    unknown = dict(status='not_classified', label='Binding outcome not yet classified',
                   resolution='unresolved', result='The outcome has not been classified in this interpretation layer; consult the original finding.',
                   assay='See source record', context='See source record',
                   expression_effect='Not assessed in this interpretation layer',
                   review_depth='No outcome classification assigned', source_url=None)
    guide = dict(
        version='oxtr-binding-outcomes-2026-10-21-v2', reviewed_on='2026-10-21',
        scope='Source-supported interpretation of the current mapped observations; source coordinates and measurements are unchanged.',
        priority=[
            dict(label='Binding detected at a defined sequence', description='A positive binding result at a specified DNA sequence is the most direct anchor for a location comparison. Check the protein, assay and biological context.'),
            dict(label='Protein association detected across a region', description='ENCODE and BrainTF supply experimental support over a stretch of DNA. Resolve the site within it and inspect the contributing cell types and experiments.'),
            dict(label='Sequence match predicted', description='JASPAR matches are candidates for exploration and experimental follow-up. A model match alone does not establish binding at that location.')],
        priority_note='This is an Atlas research-priority guide, not a platform trust score. A published paper can contain any evidence type. Relevant context, assay quality and replication matter. Binding and effects on gene activity are separate outcomes.',
        geometry_note='Biological support does not establish statistical significance or predictive value for an Atlas geometric match. That requires a defined comparison and appropriate controls.',
        negative_controls='No negative binding controls have been curated for this coordinate experiment. An absent record or an unclassified outcome is not a negative experiment.',
        outcome_vocabulary={
            'binding_detected':'Positive binding result reported for the named DNA element under the stated assay conditions.',
            'regional_association_detected':'Positive regional protein-association signal; direct sequence-specific binding and an exact motif are not established by this cluster alone.',
            'binding_not_detected':'No binding detected under specified assay conditions; assign only from an explicit source result.',
            'mixed_or_unclear':'Results differ or are inconclusive; retain each assay and condition.',
            'prediction_only':'The record reports a computational match; no location-specific binding result is established by this record.',
            'not_classified':'Outcome not classified in this layer; not equivalent to untested or negative.'},
        defaults={
            'studies':unknown,
            'oreganno':unknown,
            'encode':dict(status='regional_association_detected', label='Regional association detected', resolution='region',
                result='A ChIP-seq peak cluster records protein association with this DNA region. Binding can be direct or mediated by other proteins; the plotted centre is not an identified binding site.',
                assay='ChIP-seq peak clusters', context='Contributing experiments and cell types are retained in the record.',
                expression_effect='Effect on OXTR expression not established by this cluster',
                review_depth='UCSC ENCODE 3 track methods and archived cluster provenance; individual raw experiments not reanalysed',
                source_url='https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=encRegTfbsClustered'),
            'jaspar':dict(status='prediction_only', label='Prediction only', resolution='motif_match',
                result='A DNA sequence matches a JASPAR motif model. This record does not establish binding at this location in a cell.',
                assay='Computational motif scan', context='Model-derived sequence match; no cell-specific occupancy measurement in this record.',
                expression_effect='Effect on OXTR expression not established by this prediction',
                review_depth='JASPAR documentation and archived UCSC prediction',
                source_url='https://jaspar.elixir.no/docs/')},
        records={
            'TFBS012':dict(status='binding_detected', label='Binding detected', resolution='reported_sequence',
                result='US-1 showed protein binding in the reported DNA-binding assays. The specific binding protein remains unresolved.',
                assay='Differential electrophoretic mobility shift assay (EMSA)',
                context='Nuclear extracts from human term-pregnant and non-pregnant myometrium (uterine muscle).',
                expression_effect='Enhancer activity reported in an artificial reporter construct in cultured SKN cells; not a direct measurement of the element’s effect at its native OXTR locus.',
                review_depth='Published abstract reviewed; full-text methods and controls not independently reassessed',
                source_url='https://pubmed.ncbi.nlm.nih.gov/10221779/'),
            'TFBS013':dict(status='binding_detected', label='Binding detected', resolution='reported_sequence',
                result='US-2 showed specific binding by human MafF in the reported experiments.',
                assay='Yeast one-hybrid screening followed by EMSA',
                context='Human term-myometrium cDNA library and follow-up protein–DNA binding assays.',
                expression_effect='A role in OXTR upregulation was proposed; a causal effect on native OXTR expression is not established by the reviewed abstract.',
                review_depth='Published abstract reviewed; full-text methods and controls not independently reassessed',
                source_url='https://pubmed.ncbi.nlm.nih.gov/10527846/')} )
    guide['defaults']['braintf'] = dict(
        status='regional_association_detected', label='Regional association detected',
        resolution='region', result='A screened BrainTF ChIP-seq peak records protein association across this region; an exact binding sequence is unresolved.',
        assay='ChIP-seq peak calls', context='Human postmortem brain; the context is retained on each observation.',
        expression_effect='Effect on native OXTR expression not established by this occupancy measurement.',
        review_depth='Public processed peaks, author replication tests, FRiP and HOT annotations reviewed; raw reads not reanalysed.',
        source_url='https://doi.org/10.1038/s41593-024-01658-8')
    guide['records']['TFBS044'] = guide['defaults']['braintf']
    guide['records']['TFBS045'] = {**unknown,
        'label':'Functional reporter evidence; binding unresolved',
        'expression_effect':'Human OCE7 DNA enhanced reporter activity in mouse cells and drove mouse olfactory-bulb reporter activity. Native human OXTR regulation remains unresolved.',
        'source_url':'https://doi.org/10.1093/hmg/ddaf022'}
    write(SUP/'binding-outcomes.json', guide)
    write_record(SUP/'binding-outcomes-fields.csv', guide)
    assert read_record(SUP/'binding-outcomes-fields.csv') == guide
    catalogue = json.loads((SUP/'catalogue.json').read_text())
    catalogue['binding_interpretation'] = guide
    write(SUP/'catalogue.json', catalogue)
    # One explicit outcome row per plotted observation, joined by its stable ID.
    manifest = json.loads((SUP/'experiment-v1/manifest.json').read_text())
    rows = []
    for collection, entry in manifest['files'].items():
        p = SUP/'experiment-v1'/entry['file']
        data = json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
        for r in data:
            outcome = guide['records'].get(r['id'], guide['defaults']['braintf' if r.get('study_id') == 'braintf' else collection])
            rows.append(dict(observation_id=r['id'], collection=collection,
                             interpretation_version=guide['version'], **outcome))
    write(SUP/'mapped-binding-outcomes.json', rows)
    with (SUP/'mapped-binding-outcomes.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert len(rows)==9547 and len({r['observation_id'] for r in rows})==9547
    print(json.dumps({'mapped_outcomes':len(rows),'positive_sequence_elements':2,'regional_associations':381,'predictions':9164,'negative_controls_curated':0}))

if __name__=='__main__':
    build()
