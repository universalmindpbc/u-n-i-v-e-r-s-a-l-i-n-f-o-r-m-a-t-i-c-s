"""Build the frozen OXTR evidence supplement from retrieved public sources.

Usage: python build_oxtr_catalogue.py /path/to/oxtr-sources
No motif scans, coordinate liftover, Atlas comparison or probability calculation.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Informatics/pilot-v1/oxtr-tfbs-v1'
SRC = Path(sys.argv[1])
OUT.mkdir(exist_ok=True)
DATE = '2026-10-19'
VERSION = 'oxtr-tfbs-2026-10-19-v1'


def write_json(name, value, compressed=False):
    data = (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    (OUT / name).write_bytes(gzip.compress(data, mtime=0) if compressed else data)


def write_csv(name, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            assert all(not isinstance(v, (list, dict)) for v in row.values()), name
            w.writerow(row)


def read(name):
    return json.loads((SRC / name).read_text())


# These are curated evidence statements, not a list of independent binding sites.
# Group counts remain groups; unknown motif extents and genomic mappings stay null.
evidence = []


def add(pmid, species, factor, label, kind, methods, finding, position=None,
        sequence=None, count=None, limitation=None, related=None):
    evidence.append(dict(id=f'TFBS{len(evidence)+1:03}', pmid=str(pmid), species=species,
        factor=factor, label=label, evidence_kind=kind, methods=methods,
        finding=finding, reported_position=position, reported_sequence=sequence,
        reported_count=count, genome_build=None, chromosome=None,
        start_0based=None, end_exclusive=None, motif_strand=None,
        reference_transcript=None, coordinate_status='Original report; genomic mapping unresolved',
        limitation=limitation or 'A motif-level genomic interval was not recovered in this review.',
        related_evidence=related, source_url=f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'))


H, M = 'Homo sapiens', 'Mus musculus'
add(7798245,H,'SP1; AP1; AP2; GATA1; MYB; NFIL6; APRE','Historical promoter annotation','reported_motif_group','Gene sequencing / motif annotation','The original gene report annotates several regulatory motifs.','SP1 approximately 65 bp upstream of the reported TSS; two TSSs reported',limitation='Legacy numbering and multiple TSSs need reconciliation. These annotations are not individual experimentally validated sites.')
add(7798245,H,'Estrogen receptor family','ERE half-sites','reported_motif_group','Gene sequencing / motif annotation','Three half-palindromic EREs were reported.',count=3,limitation='The report did not identify a classical complete ERE. Later ERE-like annotations use different regions and methods.')
for factor, count in [('AP1',3),('C/EBP',9),('NF-kappaB',3)]:
    add(16569740,H,factor,'The original 15-site report','reported_motif_group','Computer analysis and EMSA; promoter assays',f'{count} {factor} sites form part of the paper’s reported set of 15.',count=count,limitation='Individual site sequences and coordinates were not recovered. The abstract does not establish the same functional contribution for every site; these groups cannot yet be deduplicated against later papers.')
add(16569740,H,'C/EBPbeta; NF-kappaB','Synergistic response region','functional_region','Deletion / reporter assays','A promoter region participates in the combined response.','−851 to −656 relative to the paper’s promoter origin',limitation='Overlaps the more focused region investigated in PMID 21734268; do not count the two studies as independent sites.',related='21734268')
add(21734268,H,'CEBPB; RELA','Shared 20-base region','functional_region','Deletion / DNA-binding / reporter assays','A 20-base region is required for synergy and binds both factors.','−712 to −692 in the paper’s promoter numbering',limitation='One shared region, not two independent sites. Historical endpoint convention retained; no genomic conversion applied.',related='16569740')
add(16888077,H,'NF-kappaB','IL-1 recruitment region','occupancy_region','ChIP / RNA polymerase II assays','IL-1 recruits NF-kappaB while reducing promoter-associated RNA polymerase II.','Approximately −955 from TSS',limitation='A reported position is not a motif extent. Regulatory direction depends on cellular and stimulus context.')
add(15494465,H,'C/EBP; AP1','Mechanical stretch response','regulatory_context','Primary myocytes / EMSA / reporter assays','Stretch increases factor binding; C/EBP overexpression activates the tested OXTR promoter.',limitation='Regulatory context; no additional individually mapped site is counted.')
add(10218980,H,'GABPA; GABPB','GABP element','functional_region','EMSA / immunoshift / mutation / reporter assays','A GABP element supports OXTR promoter activation in Hs578T breast cells.','−85 to −65 from TSS',limitation='Reported regulatory interval contains more than one element. GABP alpha/beta names are unrelated to Atlas sequence labels.')
add(10218980,H,'CRE/AP1-like element','Adjacent cooperating element','functional_region','Mutation / reporter assays','Mutation of the CRE/AP1-like element also removes promoter activation.','Within −85 to −65 from TSS',limitation='Exact motif endpoints and the binding factor at this element remain unresolved in the material reviewed.',related='TFBS010')
add(10221779,H,'Unresolved','US-1 enhancer','sequence_supported','Differential EMSA / enhancer reporter','A sequence supports differential protein binding and enhancer activity.','Within a reported 20-base region','TCTGCCTTCATCCAGCC',limitation='The reported sequence has 17 bases; retain it separately from the 20-base region description. Factor identity and genomic mapping remain open.')
add(10527846,H,'MAFF','US-2 element','sequence_supported','Yeast one-hybrid / EMSA','Human MafF binds the reported US-2 sequence.',sequence='GGAATGATTACTCAGCTAGA',limitation='Sequence retained as reported; genomic placement has not been independently aligned.')
add(16549056,H,'MAFF; MIP','US-2 co-activation','regulatory_context','Tandem-element reporter / co-expression','MIP supports MafF-mediated activation in the US-2 reporter system.',limitation='Reuses US-2; not an additional binding site. MIP is a co-activator.',related='TFBS013')
for pos in [-1005,-944]:
    add(33633538,H,'ESR2 / ERbeta','ERE-like element','functional_region','Promoter mutation / reporter / ChIP','ERE-like elements participate in the reported OXTR response.',f'{pos} from TSS',limitation='Position labels, not recovered motif extents. ChIP supports regional occupancy; it does not resolve each motif separately.')
    evidence[-1]['reference_transcript']='ENST00000316793.7'
for factor,count in [('GATA1',1),('SP1',1),('YY1',1),('CEBPA',2)]:
    add(33633538,H,factor,'Other motifs in the assayed region','reported_motif_group','Motif annotation / regional ChIP','Candidate motif annotations in the −1100 to −900 interval.','−1100 to −900 from TSS',count=count,limitation='Prediction is not proof of individual occupancy or regulatory function. The two ERE-like elements are recorded separately.')
for factor,pos in [('Half ERE',-932),('Putative SP1',-733)]:
    add(23441222,M,factor,'CpG-associated candidate element','candidate_region','Methylation / promoter mutation / reporter assays','The study tests methylation-sensitive Oxtr promoter regulation.',f'CpG position {pos}; not the motif start',limitation='Coordinates label CpGs inside candidate elements. Direct occupancy by the named factor was not established by these assays.')
for label in ['E1','E2','E3','E4']:
    supported=label in ['E2','E4']
    add(41109427,M,'BMAL1 / clock complex',label+" E′-box",'functional_candidate' if supported else 'tested_candidate', 'NIH3T3 promoter mutation / reporter assays', 'Mutation abolishes the tested BMAL1-induced reporter response.' if supported else 'Mutation does not abolish the tested BMAL1-induced reporter response.', 'Within −1000 to +200 of mouse Oxtr promoter',limitation='The four candidate sites belong to the mouse promoter. Published occupancy context includes male mouse liver; these are not four confirmed human or uterine ChIP sites.')
add(21079000,H,'ZEB1; ZEB2','Myometrial repression','regulatory_context','Overexpression / expression assays','ZEB1 and ZEB2 overexpression represses OXTR in human myometrial cells.',limitation='Abstract-level review; no human motif-level genomic interval recovered. Animal experiments in the same paper are not merged into this record.')
for label in ['BS1','BS2','BS3','BS4','BS5']:
    supported=label in ['BS1','BS3','BS4']
    add(30106994,M,'PEG3',label,'binding_supported' if supported else 'tested_candidate','Competitive EMSA'+(' / ChIP' if label=='BS4' else ''),'Competitive EMSA supports binding.'+(' BS4 also lies in the ChIP-tested region.' if label=='BS4' else '') if supported else 'The reported competitive EMSA did not support binding.', 'Third-exon region' if label=='BS4' else None,limitation='Site sequences require figure-level recovery. The paper describes a 225-bp ChIP region but also lists mm10 chr6:112488870–112489904, an inconsistent span; genomic normalization is withheld.')
add(36094838,H,'HIF1A','Hypoxia-associated coding-region occupancy','occupancy_region','ChIP-seq / hypoxia / knockdown','HIF1A occupancy is reported in the OXTR coding region with expression responses to hypoxia.',position='OXTR coding region',limitation='A ChIP region is not a resolved short binding motif. Exact peak coordinates were not recovered from the article.')
add(38897566,M,'PGR; CNOT1','Progesterone receptor complex','regulatory_context','Myometrial ChIP / re-ChIP','The mouse Oxtr region shows PR binding; CNOT1 re-ChIP enrichment at Oxtr was not significant.',limitation='CNOT1 is a coregulator, not an established sequence-specific DNA-binding factor. Human cell experiments in the paper are additional context, not merged site counts.')
add(28049625,H,'NF-kappaB','Inflammatory macrophage response','regulatory_context','Primary macrophages / THP-1 / pathway inhibition','Inflammatory induction of OXTR depends on NF-kappaB signalling in the studied cells.',limitation='Regulatory evidence without an additional mapped binding site.')
add(22030357,H,'SP1','Neural cell expression response','regulatory_context','Human tissue / SK-N-SH silencing / inhibitor','SP1 perturbation changes OXTR expression in the studied neural cells.',limitation='Expression dependence does not locate a binding site. Other genes in the study are not OXTR sites.')
for pos in [-104,-64]:
    add(16254027,'Ovis aries','SP1','GC-rich promoter element','functional_region','Promoter mutation / reporter assays','SP1 elements support basal and estrogen-mediated promoter activity.',f'{pos} in the paper’s promoter numbering',limitation='Ovine evidence; estrogen-mediated action at these elements is not a human ERE site.')
add(9048622,'Rattus norvegicus','Estrogen receptor','Upstream palindromic ERE','functional_region','EMSA / mutation / reporter assays','An upstream ERE supports estrogen-responsive transcription.','Approximately 4 kb upstream of translation start (ATG)',limitation='Translation-start anchor, not TSS; approximate position only.')
add(12604656,'Bos taurus','IRF1; IRF2','Interferon-response element','functional_region','EMSA / reporter assays','An upstream element binds IRF1 and IRF2.','Approximately −2400 from TSS',limitation='One shared bovine element, not two independent sites.')
add(19439725,'Oryctolagus cuniculus','CREB','cAMP-responsive region','occupancy_region','DNase hypersensitivity / ChIP / reporter assays','A CREB-associated region participates in the amnion cAMP response.','A 200-bp region approximately 4.7 kb upstream of TSS',limitation='Broad region in rabbit amnion; exact motif endpoints were not recovered.')
add(27108164,H,'ESR1; SP1','Methylation-associated motif annotations','candidate_region','Motif prediction / published genomic tracks','The paper relates methylation measurements to candidate regulatory motifs.','hg19 CpGs chr3:8810078 and chr3:8810069',limitation='These coordinates are CpG anchors, not motif coordinates. No new binding-site count is inferred.')
add(37479681,H,'EGR1','OREG1492306','annotated_region','Published annotation / methylation study','The paper references an EGR1-associated ORegAnno region.','GRCh38 chr3:8767502–8768543 (BED)',limitation='Annotation reused by the paper, not a new EGR1 binding experiment.',related='OREG1492306')
evidence[-1].update(genome_build='GRCh38',chromosome='chr3',start_0based=8767502,end_exclusive=8768543,coordinate_status='ORegAnno interval recovered; regional annotation')

search_files=['pubmed-oxtr.json','pubmed-oxtr-expanded.json','tf-publications.json','inoue.json']
papers_by_id={r['id']:r for n in search_files for r in read(n)['resultList']['result']}
fulltext={'33633538':'diabetes.xml','23441222':'sp1.xml','41109427':'clock.xml','30106994':'peg3.xml','36094838':'hif.xml','38897566':'cnot.xml','37479681':'landscape.xml'}
if '38897566' not in papers_by_id:
    x=ET.parse(SRC/'cnot.xml').getroot()
    papers_by_id['38897566']={'title':''.join(x.find('.//article-title').itertext()),'doi':'10.1016/j.jbc.2024.107484','pubYear':'2024','firstPublicationDate':'2024-06-19','pmcid':'PMC11301068','authorString':'; '.join(' '.join(a.itertext()) for a in x.findall('.//article-meta/contrib-group/contrib/name')),'journalInfo':{'journal':{'title':'Journal of Biological Chemistry'}}}
publications=[]
for n,pmid in enumerate(dict.fromkeys(r['pmid'] for r in evidence),1):
    p=papers_by_id[pmid]
    publications.append(dict(id=f'TFP{n:02}',pmid=pmid,title=p['title'],authors=p.get('authorString'),doi=p.get('doi'),year=int(p.get('pubYear') or p['firstPublicationDate'][:4]),first_publication_date=p.get('firstPublicationDate'),journal=p.get('journalInfo',{}).get('journal',{}).get('title'),url=f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',doi_url='https://doi.org/'+p['doi'],pmcid=p.get('pmcid'),source_review='Selected full-text methods, results and captions' if pmid in fulltext else ('Abstract and figure descriptions' if pmid=='27108164' else 'Abstract and bibliographic metadata'),existing_register_id='E03' if pmid=='41109427' else None,trust_score=None,trust_status='Not assigned; evidence type and review depth reported',jadad='Not applicable: not a randomized clinical trial'))
pub_index={p['pmid']:p for p in publications}
for r in evidence:r['publication_id']=pub_index[r['pmid']]['id']
write_json('literature-evidence.json',evidence)
write_json('publications.json',publications)
write_csv('publications.csv',publications)

ens=read('ensembl-oxtr.json')
assert (ens['start'],ens['end'],ens['strand'])==(8750381,8769628,-1)
region=dict(species=H,assembly='GRCh38 / hg38',chromosome='chr3',start_0based=8740380,end_exclusive=8779628,length_bp=39248,gene_id=ens['id'],canonical_transcript=ens['canonical_transcript'],gene_start_1based=ens['start'],gene_end_1based=ens['end'],gene_strand=-1,selection='Ensembl OXTR gene body plus 10,000 bases on each genomic side',scope='A defined locus window; distal regulatory elements outside it are not captured. Overlapping records may extend beyond its boundaries.')
dna=read('oxtr-dna.json')['dna'].upper()
expanded=read('ref-dna-expanded.json')['dna'].upper()
assert len(dna)==39248 and set(dna)<=set('ACGTN') and len(expanded)==60000
write_json('reference-dna.json',dict(**region,dna_plus_strand=dna))
(OUT/'reference-dna.fa').write_text('>hg38_chr3_8740380_8779628 BED_0based_halfopen plus_strand\n'+'\n'.join(dna[i:i+80] for i in range(0,len(dna),80))+'\n')
write_csv('reference-dna.csv',(dict(assembly='hg38',chromosome='chr3',position_0based=8740380+i,base_plus_strand=b) for i,b in enumerate(dna)))
complement=str.maketrans('ACGTN','TGCAN')
def sequence_at(row):
    seq=expanded[row['chromStart']-8730000:row['chromEnd']-8730000]
    assert len(seq)==row['chromEnd']-row['chromStart']
    return seq.translate(complement)[::-1] if row['strand']=='-' else seq

jaspar=read('jaspar-oxtr.json')
hits=jaspar['jaspar2026']
assert len(hits)==217857 and jaspar['itemsReturned']==len(hits)
hits_flat=[dict(id=f'J26_{i+1:06}',assembly='hg38',chromosome=r['chrom'],start_0based=r['chromStart'],end_exclusive=r['chromEnd'],matrix_id=r['name'],factor=r['TFName'],score=r['score'],strand=r['strand'],default_display_threshold_pass=r['score']>=400,sequence_oriented=sequence_at(r),source_track='jaspar2026') for i,r in enumerate(hits)]
write_json('jaspar-predictions.json.gz',hits_flat,True)
write_csv('jaspar-predictions.csv',hits_flat)
default_hits=[r for r in hits_flat if r['default_display_threshold_pass']]
assert len(default_hits)==9164
write_csv('jaspar-default-threshold.csv',default_hits)
shutil.copyfile(SRC/'jaspar2026-vertebrates.jaspar',OUT/'jaspar2026-vertebrates.jaspar')
pfms=[]; current=None
for line in (SRC/'jaspar2026-vertebrates.jaspar').read_text().splitlines():
    if line.startswith('>'):
        mid,name=line[1:].split(None,1); current=dict(matrix_id=mid,factor=name,counts={});pfms.append(current)
    elif line.strip():
        base,values=line.split('[',1);current['counts'][base.strip()]=[int(float(n)) for n in values.rstrip(' ]').split()]
assert len(pfms)==1019
assert {r['matrix_id'] for r in hits_flat}<={p['matrix_id'] for p in pfms}
write_json('jaspar-pfm-templates.json',pfms)
write_csv('jaspar-pfm-templates.csv',(dict(matrix_id=p['matrix_id'],factor=p['factor'],base=b,position_1based=i+1,count=v) for p in pfms for b,vs in p['counts'].items() for i,v in enumerate(vs)))

enc=read('encode-oxtr.json')['encRegTfbsClustered']
source_lines=list(csv.reader(gzip.open(SRC/'encode-sources.txt.gz','rt'),delimiter='\t'))
input_lines=list(csv.reader(gzip.open(SRC/'encode-inputs.txt.gz','rt'),delimiter='\t'))
srcs={int(r[0]):r for r in source_lines}
inputs={r[1]:r for r in input_lines}
clusters=[]; provenance=[]
for i,r in enumerate(enc):
    cid=f'ENC3_{i+1:04}'
    ids=[int(n) for n in r['sourceIds'].strip(',').split(',')]
    scores=[int(n) for n in r['sourceScores'].strip(',').split(',')]
    assert len(ids)==len(scores)==r['sourceCount']
    clusters.append(dict(id=cid,assembly='hg38',chromosome=r['chrom'],start_0based=r['chromStart'],end_exclusive=r['chromEnd'],target=r['name'],score=r['score'],source_count=r['sourceCount'],strand=None,source_track='encRegTfbsClustered'))
    for sid,score in zip(ids,scores):
        source=srcs[sid]; inp=inputs[source[2]]
        assert inp[2]==r['name'],(sid,inp,r)
        provenance.append(dict(cluster_id=cid,source_id=sid,source_score=score,source_name=source[2],target=inp[2],antibody=inp[3],cell_type=inp[4],experiment=inp[5],lab=inp[6],file_accession=inp[0].removeprefix('encTfChipPk'),experiment_url='https://www.encodeproject.org/experiments/'+inp[5]+'/'))
assert len(clusters)==255 and len({r['target'] for r in clusters})==133
write_json('encode-clusters.json',clusters);write_json('encode-provenance.json',provenance)
write_csv('encode-clusters.csv',clusters);write_csv('encode-provenance.csv',provenance)
oreg=[dict(id=r['id'],assembly='hg38',chromosome=r['chrom'],start_0based=r['chromStart'],end_exclusive=r['chromEnd'],annotation_strand=r['strand'],annotation_name=r['name'],factor='EGR1 (literature annotation)' if r['id']=='OREG1492306' else None,source_track='oreganno') for r in read('oreganno-oxtr.json')['oreganno']]
assert len(oreg)==37
write_json('oreganno-regions.json',oreg);write_csv('oreganno-regions.csv',oreg)

retrieval=read('retrieval.json')
source_urls={r['file']:r['url'] for r in retrieval}
source_files=['ensembl-oxtr.json','oxtr-dna.json','ref-dna-expanded.json','jaspar-oxtr.json','jaspar2026-vertebrates.jaspar','jaspar-vertebrates.json','jaspar-vertebrates-page2.json','encode-oxtr.json','encode-inputs.txt.gz','encode-sources.txt.gz','oreganno-oxtr.json','jaspar-docs.html','jaspar-track-doc.html','encode-track-doc.html','oreganno-doc.html']
sources=[]
raw=OUT/'source-snapshots';raw.mkdir(exist_ok=True)
for name in source_files:
    data=(SRC/name).read_bytes()
    sources.append(dict(file=name,url=source_urls[name],retrieved_on=DATE,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),snapshot='source-snapshots/'+name+'.gz',reuse_note='Public JASPAR data / UCSC distributed genome tracks and track documentation; retain source attribution. No TRANSFAC licensed data included.'))
    (raw/(name+'.gz')).write_bytes(gzip.compress(data,mtime=0))
write_csv('sources.csv',sources)
queries=[dict(file=n,url=source_urls[n],retrieved_on=DATE,returned=read(n)['hitCount']) for n in search_files]
ledger=[]
all_results={r['id']:r for n in search_files for r in read(n)['resultList']['result']}
for pid,p in all_results.items():
    ledger.append(dict(pmid=pid,title=p['title'],doi=p.get('doi'),year=p.get('pubYear'),status='Included in evidence supplement' if pid in pub_index else 'Search return; not included in this first extraction',reason='Relevant OXTR transcriptional-regulation evidence reviewed' if pid in pub_index else 'No inclusion or exclusion judgement implied; full systematic screening remains open',url=f'https://europepmc.org/article/{p.get("source","MED")}/{pid}'))
write_json('search-ledger.json',ledger);write_csv('search-ledger.csv',ledger)

resources=[
 dict(name='JASPAR 2026',status='Collected',url='https://jaspar.elixir.no/docs/',evidence='Published motif predictions and PFM templates',count=len(hits),unit='predicted motif matches',note='9,164 meet UCSC’s default score >=400 display threshold. All 217,857 returned predictions are retained. Several models and strands can overlap one location. Score is a motif-match statistic, not an OXTR regulatory probability.'),
 dict(name='ENCODE 3 / UCSC',status='Collected',url='https://www.encodeproject.org/chip-seq/transcription_factor/',evidence='ChIP-seq clusters, release November 2018',count=len(clusters),unit='regional occupancy clusters',note='133 assayed targets across aggregated experimental contexts. Includes coregulators; not 133 confirmed OXTR-regulating TFs. Provenance joins each cluster to its experiment and cell type.'),
 dict(name='ORegAnno / UCSC',status='Collected',url='https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=oreganno',evidence='Regulatory annotations, UCSC snapshot October 2015',count=len(oreg),unit='overlapping regional annotations',note='Annotation class and factor are unresolved for most returned rows. Regional overlap alone does not establish OXTR regulation.'),
 dict(name='TRANSFAC',status='Licensed access pending',url='https://genexplain.com/transfac-database/',evidence='Curated binding sites / regulatory knowledge',count=None,unit=None,note='Public product documentation reviewed. No licensed export or redistribution permission available; no protected entries copied.'),
 dict(name='TFBSdb',status='Legacy endpoint unavailable',url='https://tfbsdb.systemsbiology.net/viewgene/5021',evidence='Historical hg19 motif / footprint resource',count=None,unit=None,note='Indexed OXTR page returns HTTP 404. Search snippets are not imported as a verified database export.'),
 dict(name='ReMap',status='Resource identified; locus export pending',url='https://remap.univ-amu.fr/download_page',evidence='Independent ChIP-seq compendium',count=None,unit=None,note='No additional ReMap records counted in this release.'),
 dict(name='ChIP-Atlas',status='Resource identified; locus export pending',url='https://chip-atlas.org/target_genes',evidence='ChIP-seq and target-gene context',count=None,unit=None,note='Gene-target associations and precise binding intervals are distinct; no extra sites counted.'),
 dict(name='Wolfram',status='Planned analysis resource',url='https://reference.wolfram.com/language/ref/BioSequence.html',evidence='Sequence handling and symbolic analysis',count=None,unit=None,note='A future analysis tool, not an experimental binding-site source.')]
catalogue=dict(schema_version='1.0',version=VERSION,collected_on=DATE,title='OXTR binding-site catalogue',target_gene='OXTR',status='Initial literature and database collection; comparisons not yet performed',region=region,
    summary=dict(publications=len(publications),literature_evidence_records=len(evidence),jaspar_all_predictions=len(hits),jaspar_default_threshold=400,jaspar_default_threshold_predictions=len(default_hits),jaspar_distinct_matrices=len({r['matrix_id'] for r in hits_flat}),jaspar_available_pfm_templates=len(pfms),encode_clusters=len(clusters),encode_targets=133,oreganno_regions=len(oreg),unique_confirmed_oxtr_binding_site_total=None),
    interpretation='The original 15 sites are one human promoter study’s reported motif set. The catalogue expands the evidence base, but overlapping reports, broad regions, predictions and species are not added into one confirmed-site total.',
    coordinate_convention='Database intervals are GRCh38/hg38, 0-based start and exclusive end. Literature coordinates retain their original TSS/ATG/CpG anchors and genome builds. Unrecovered values are null; CSV blanks represent null unless the typed schema says otherwise.',
    planned_views=['Original DNA line with biological coordinates','Full-position circle preserving relative spacing and a fixed origin','Modulus 9 view alongside the uncompressed positions'],
    analysis_status='No DNA-to-Atlas mapping, M9 reduction, enrichment test, new motif scan or biological significance score has been applied. Motif-oriented sequence strings are reference-genome extractions, not observed individual genomes.',
    resources=resources,publications=publications,evidence=evidence,search_queries=queries,
    coverage='Broad web and bibliographic discovery plus selected primary-source review. This is a documented first collection, not a completed systematic review or a census of every OXTR regulator. Genome-wide distal elements, all cell states and all databases remain outside its scope.',
    outstanding=['Recover individual sequences / coordinates for the 2006 15-site paper and other legacy articles','Resolve historical TSS numbering and the Peg3 span inconsistency before coordinate conversion','Obtain authorized TRANSFAC access and check current ReMap / ChIP-Atlas locus data','Deduplicate overlapping evidence and distinguish factor identity from motif-family ambiguity','Choose tissue, regulatory window and comparison rules before any Atlas geometry test'],
    files=dict(evidence='literature-evidence.json',publications='publications.json',jaspar_predictions='jaspar-predictions.json.gz',jaspar_templates='jaspar-pfm-templates.json',encode='encode-clusters.json',encode_provenance='encode-provenance.json',oreganno='oreganno-regions.json',reference_dna='reference-dna.json',sources='sources.csv',search_ledger='search-ledger.json',json_bundle='oxtr-tfbs-json.zip',csv_bundle='oxtr-tfbs-csv.zip'),sources=sources)
write_json('catalogue.json',catalogue)
from csv_records import write_record, read_record
write_record(OUT/'catalogue-fields.csv',catalogue)
assert read_record(OUT/'catalogue-fields.csv')==catalogue
(OUT/'README.txt').write_text('''OXTR binding-site catalogue — 19 October 2026

This supplement collects published OXTR regulatory evidence and existing database records. It does not perform a new motif scan or compare the data with Atlas geometry.

Read catalogue.json or the lossless typed catalogue-fields.csv for scope, evidence, publications, source URLs, methods, caveats and unresolved work. literature-evidence.csv is the accessible row table; reported_count is an aggregate reported by a paper, not a new unique-site total. Null coordinates remain blank in flat CSV files. Literature positions remain verbatim; never treat a TSS-relative label or CpG coordinate as a precise motif interval.

Database region: GRCh38 chr3:8740380–8779628, 0-based half-open, Ensembl OXTR gene body plus 10 kb on each side. OXTR is on the minus strand. The reference DNA file is the genomic plus strand. Motif sequence_oriented follows the motif strand. Overlap queries retain records extending beyond the window; those sequences use the separately archived 60 kb reference extraction. No liftover or transcript realignment was performed.

JASPAR: 217,857 published predictions in the returned window; 9,164 meet UCSC’s score >=400 default display filter. The track’s underlying prediction method uses a relative PWM threshold of 0.8 and p<0.05; UCSC score is based on -100 log10(p), capped at 1000. The display filter corresponds approximately to p<=1e-4, without a locus-level multiple-testing correction. Prediction is not observed binding or proof of OXTR regulation. The complete returned set is retained, alongside the filtered CSV. 1,019 vertebrate nonredundant PFM templates are included; 1,003 different models occur among the returned matches. A PFM is a model, not an OXTR-specific site. JASPAR 2026 publication: https://doi.org/10.1093/nar/gkaf1209 .

ENCODE: 255 regional clusters for 133 target names, from the UCSC ENCODE 3 November 2018 release. Each cluster links to all its source experiments and cell types in encode-provenance.csv/json. Clusters represent broad occupancy and include coregulators. They are not 255 precisely located, functionally confirmed OXTR motifs.

ORegAnno: 37 overlapping regional annotations from the UCSC 2015 snapshot. Most factor/class metadata were not resolved. OREG1492306 is cross-referenced by the 2023 OXTR methylation paper. It is counted once in this table, not again as a new experiment.

Publication findings are factual summaries with source links. Full article text, figures, and licensed TRANSFAC data are not redistributed. The existing 14-study Atlas research register remains separate; its E03 circadian study is cross-referenced here, not a new independent study. Review depth and evidence type are recorded; numerical trust and WSS scores are unassigned for this supplement.

CSV archive: all tabular evidence, metadata, DNA bases, matrix counts, predictions, clusters and provenance are available without a JSON parser. catalogue-fields.csv uses the Atlas typed path convention (field,type,value,resolved_url); source/csv_records.py restores its nested structure. Flat tables use one row per declared record and retain explicit identifiers. Neither repeated motif names nor repeated database hits establish independent observations.

The search ledger records all returned bibliographic items; items outside the selected extraction have not all been systematically screened. TRANSFAC licensing, the unavailable legacy TFBSdb endpoint, individual old-paper coordinates and additional database exports remain open. No exhaustive completeness claim is made. The three future views preserve original DNA positions, full-circle spacing and a separately labelled M9 reduction.
''',encoding='utf-8')
print(json.dumps(dict(version=VERSION,summary=catalogue['summary'],encode_provenance_rows=len(provenance),search_returns=len(ledger),output=str(OUT)),indent=2))

# Keep the shared harmonisation layer when rebuilding source exports.
from normalize_oxtr_sites import build as normalize_sites
normalize_sites()
