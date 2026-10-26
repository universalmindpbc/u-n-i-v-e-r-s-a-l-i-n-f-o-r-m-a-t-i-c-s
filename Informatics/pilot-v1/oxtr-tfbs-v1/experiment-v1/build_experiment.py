"""Build OXTR coordinate experiment v1 from the archived source files.

Run: python build_experiment.py --source /path/to/oxtr-tfbs-v1 --atlas /path/to/m9.json
Requires the pinned requirements.txt. No network calls, motif scans or significance tests.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
from collections import Counter
from pathlib import Path

import Bio
from Bio import SeqIO, motifs
from Bio.Seq import Seq

VERSION = 'oxtr-coordinate-experiment-2026-10-20-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def write(path, value):
    raw = (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == '.gz' else raw)


def csv_table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v, ensure_ascii=False, separators=(',', ':'))
                             if isinstance(v, (dict,list)) else v for k,v in row.items()})


def project(start, end, lo, hi):
    assert isinstance(start, int) and isinstance(end, int) and start < end
    a, b = max(start, lo), min(end, hi)
    assert a < b, 'The source interval must overlap the fixed window'
    begin, stop = hi-b, hi-a
    offset = (begin+stop-1)//2
    residue = offset % 9
    fraction = offset/(hi-lo)
    theta = 2*math.pi*fraction
    return dict(visible_start_0based=a, visible_end_exclusive=b,
                clipped_to_window=(a != start or b != end),
                offset_start=begin, offset_end_exclusive=stop,
                anchor_genomic_0based=hi-1-offset, anchor_offset_bp=offset,
                full_circle_turns=fraction, full_circle_degrees=360*fraction,
                full_circle_xy=[math.sin(theta), -math.cos(theta)],
                m9_residue=residue, m9_display=residue or 9,
                m9_degrees=40*residue,
                m9_xy=[math.sin(2*math.pi*residue/9), -math.cos(2*math.pi*residue/9)])


def build(source, atlas, output):
    output.mkdir(parents=True, exist_ok=True)
    cat, reference, m9 = read(source/'catalogue.json'), read(source/'reference-dna.json'), read(atlas)
    lo, hi = reference['start_0based'], reference['end_exclusive']
    dna = Seq(reference['dna_plus_strand'])
    fasta = SeqIO.read(source/'reference-dna.fa', 'fasta')
    assert str(fasta.seq).upper() == str(dna).upper()
    assert len(dna) == hi-lo == 39248
    with (source/'jaspar2026-vertebrates.jaspar').open() as handle:
        templates = {m.matrix_id: m for m in motifs.parse(handle, 'jaspar')}
    mappings = {r['record_id']: r for r in read(source/'sequence-mapping.json')}
    provenance = {}
    for r in read(source/'encode-provenance.json'):
        provenance.setdefault(r['cluster_id'], []).append(r)
    published = []
    for r in cat['evidence']:
        if r['id'] not in mappings or mappings[r['id']]['start_0based'] is None:
            continue
        m = mappings[r['id']]
        assert r['species'] == 'Homo sapiens'
        sequence = Seq(m['sequence'])
        matches = []
        for strand, query in [('+', sequence), ('-', sequence.reverse_complement())]:
            offset = dna.find(query)
            while offset >= 0:
                matches.append((lo+offset, lo+offset+len(query), strand))
                offset = dna.find(query, offset+1)
        assert matches == [(m['start_0based'], m['end_exclusive'], m['strand'])]
        paper = next(p for p in cat['publications'] if p['pmid'] == r['pmid'])
        published.append(dict(r, **{k:m[k] for k in ['start_0based','end_exclusive','strand','chromosome']},
                              sequence_oriented=str(sequence), source_url=paper['doi_url'], publication=paper))
    published.extend(read(source/'study-additions-v1/braintf-peaks.json'))
    predictions = [r for r in read(source/'jaspar-predictions.json.gz') if r['score'] >= 400]
    inputs = {'studies':published, 'encode':read(source/'encode-clusters.json'), 'jaspar':predictions}
    unit = {'studies':'observations', 'encode':'regions', 'jaspar':'matches'}
    sequence_checks = Counter()
    summaries, files = {}, {}
    for collection, raw in inputs.items():
        records = []
        for r in raw:
            row = dict(r, collection=collection, count_unit=unit[collection], assembly='GRCh38')
            row['factor'] = r.get('factor') or r.get('target') or r.get('label') or 'Unspecified'
            if collection == 'encode':
                row['experiments'] = provenance.get(r['id'], [])
                assert len(row['experiments']) == r['source_count']
                row['cell_types'] = sorted({p['cell_type'] for p in row['experiments'] if p['cell_type']})
                row['source_url'] = 'https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=encRegTfbsClustered'
            if collection == 'jaspar':
                row['source_url'] = 'https://jaspar.elixir.no/matrix/'+r['matrix_id']+'/'
                assert r['matrix_id'] in templates
            row['mapping'] = project(r['start_0based'], r['end_exclusive'], lo, hi)
            # Preserve broad ENCODE regions. A centre is a plotting representative, not a recovered binding nucleotide.
            if r.get('sequence_oriented'):
                if lo <= r['start_0based'] < r['end_exclusive'] <= hi:
                    seq = dna[r['start_0based']-lo:r['end_exclusive']-lo]
                    if r['strand'] == '-':
                        seq = seq.reverse_complement()
                    assert str(seq).upper() == r['sequence_oriented'].upper(), r['id']
                    sequence_checks[collection] += 1
                    row['sequence_validation'] = 'Biopython reference extraction and strand check passed'
                else:
                    row['sequence_validation'] = 'Source interval extends beyond saved reference; full sequence check unavailable'
            records.append(row)
        records.sort(key=lambda r:(r['mapping']['anchor_offset_bp'], r['start_0based'], r['id']))
        counts = [sum(r['mapping']['m9_residue'] == i for r in records) for i in range(9)]
        assert sum(counts) == len(raw)
        summaries[collection] = dict(count=len(records), unit=unit[collection], residue_counts_0_to_8=counts,
            clipped_intervals=sum(r['mapping']['clipped_to_window'] for r in records),
            anchor_offset_sum=sum(r['mapping']['anchor_offset_bp'] for r in records))
        name = collection+'.json'+('.gz' if collection == 'jaspar' else '')
        write(output/name, records)
        csv_table(output/(collection+'.csv'), [dict(r, **r['mapping'], experiment_version=VERSION,
            window_start_0based=lo, window_end_exclusive=hi, origin_genomic_base_0based=hi-1,
            direction='OXTR minus strand', coordinate_convention='0-based, end-exclusive') for r in records])
        files[collection] = dict(file=name, sha256=digest(output/name))
    # Recompute the recurrence, primary nodes and interval sums independently of the saved drawing data.
    alpha, a, b = [], 0, 1
    while True:
        alpha.append(a)
        a, b = b, (a+b) % 9
        if (a,b) == (0,1):
            break
    gamma = [i for i,v in enumerate(alpha) if v % 3 == 0]
    sums = [sum(alpha[i+1:j]) for i,j in zip(gamma, gamma[1:]+[len(alpha)])]
    beta = [s % 9 for s in sums]
    assert alpha == m9['alpha']['residues']
    assert gamma == m9['gamma']['alpha_indices']
    assert sums == m9['beta']['interval_sums']
    assert beta == m9['beta']['interval_residues']
    ep = m9['epsilon']['alpha_indices']
    assert [alpha[i] or 9 for i in ep] == m9['epsilon']['traversal_values']
    assert ep == [0,8,20,12,4,16,0]
    source_names = ['catalogue.json','study-additions-v1/braintf-peaks.json','reference-dna.json','reference-dna.fa','sequence-mapping.json',
                    'encode-clusters.json','encode-provenance.json','jaspar-predictions.json.gz','jaspar2026-vertebrates.jaspar']
    manifest = dict(version=VERSION, status='Exploratory coordinate comparison; biological association untested',
        reference=dict(cat['region'], origin_genomic_base_0based=hi-1, direction='OXTR minus strand; descending genomic coordinates',
                       clockwise_from='north', coordinate_convention='0-based, end-exclusive'),
        mapping=dict(interval='Use the interval intersection with the fixed reference window; retain the original interval',
                     anchor='Central base of visible interval; for even widths choose the first central base in OXTR direction',
                     formula='d0=window_end-min(end,window_end); d1=window_end-max(start,window_start); d=floor((d0+d1-1)/2)',
                     full_circle='turns=d/window_length; x=sin(2*pi*turns), y=-cos(2*pi*turns)',
                     modulo9='r=d mod 9; display 0 as 9; x=sin(2*pi*r/9), y=-cos(2*pi*r/9)',
                     origin_note='This window-edge origin is a fixed plotting convention, not the OXTR transcription start site',
                     inference='No rotation optimisation, fit statistic, p-value, enrichment test or biological validation'),
        jaspar_minimum_score=400, summaries=summaries, files=files,
        atlas=dict(id=m9['id'], sha256=digest(atlas), alpha=m9['alpha'], beta=m9['beta'], gamma=m9['gamma'], epsilon=m9['epsilon'],
                   orientation=m9['conventions']['orientation'], grids='Alpha/Gamma/Epsilon: 24 period positions; Beta: 9 residue positions'),
        source_sha256={name:digest(source/name) for name in source_names},
        verification_files=['python-validation.json','wolfram-validation.json','verify.wl'],
        reproducibility_files=['build_experiment.py','requirements.txt','methods.md'])
    write(output/'manifest.json', manifest)
    validation = dict(version=VERSION, status='passed', python=platform.python_version(), biopython=Bio.__version__,
        reference_length=len(dna), fasta_json_match=True, published_exact_matches=2,
        sequence_checks=dict(sequence_checks), parsed_jaspar_templates=len(templates),
        records_projected=sum(x['count'] for x in summaries.values()), summaries=summaries,
        atlas_checks='Fibonacci recurrence, Gamma indices, Beta interval sums, canonical Epsilon traversal',
        scope='Precomputed coordinate and sequence checks; not biological significance or live server execution')
    if (output/'wolfram-validation.json').exists():
        wolfram = read(output/'wolfram-validation.json')
        table = ''.join(','.join(map(str, [d,d%9,d%9 or 9,
            round(1e9*math.sin(2*math.pi*d/len(dna))), round(-1e9*math.cos(2*math.pi*d/len(dna))),
            round(1e9*math.sin(2*math.pi*(d%9)/9)), round(-1e9*math.cos(2*math.pi*(d%9)/9))]))+'\n'
            for d in range(len(dna)))
        table_hash = hashlib.sha256(table.encode()).hexdigest()
        assert table_hash == wolfram['coordinate_table_sha256']
        for s,e,g,d,r,z in wolfram['interval_cases']:
            m = project(s,e,lo,hi)
            assert [m[k] for k in ['anchor_genomic_0based','anchor_offset_bp','m9_residue','m9_display']] == [g,d,r,z]
        assert alpha == wolfram['alpha_residues'] and gamma == wolfram['gamma_indices']
        assert beta == wolfram['beta_residues'] and ep == wolfram['epsilon_indices']
        validation['wolfram_coordinate_table_sha256'] = table_hash
        validation['wolfram_parity'] = 'passed: all offsets, nine edge cases, reference sequences'
    write(output/'python-validation.json', validation)
    for filename, value in [('manifest-fields.csv',manifest),('python-validation-fields.csv',validation)]:
        # Metadata is losslessly retained as typed JSON values inside ordinary CSV cells.
        csv_table(output/filename, [dict(field=k, value_json=json.dumps(v, ensure_ascii=False, separators=(',', ':'))) for k,v in value.items()])
    if (output/'wolfram-validation.json').exists():
        csv_table(output/'wolfram-validation-fields.csv', [dict(field=k, value_json=json.dumps(v, ensure_ascii=False, separators=(',', ':'))) for k,v in wolfram.items()])
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--atlas', type=Path, default=Path(__file__).resolve().parents[2]/'m9.json')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    build(args.source, args.atlas, args.output)
