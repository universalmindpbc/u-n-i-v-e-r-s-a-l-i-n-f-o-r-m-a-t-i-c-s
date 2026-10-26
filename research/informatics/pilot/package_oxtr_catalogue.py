"""Attach the shared OXTR supplement to both complete Atlas export formats."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from csv_records import write_record, read_record

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'Informatics/pilot-v1'
SUP=DATA/'oxtr-tfbs-v1'


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')


def replace_zip(path, replacements):
    """Preserve unrelated members, avoiding duplicate entries during refresh."""
    temp=path.with_suffix('.working.zip')
    with zipfile.ZipFile(path) as old, zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as new:
        for item in old.infolist():
            if item.filename not in replacements and '/oxtr-tfbs-v1/' not in item.filename and not item.filename.startswith('oxtr-tfbs-v1/'):
                with old.open(item) as src, new.open(item,'w') as dst:shutil.copyfileobj(src,dst)
        for name, file in replacements.items():new.write(file,name)
    temp.replace(path)


def publish():
    if not (SUP/'catalogue.json').exists():return
    catalogue=json.loads((SUP/'catalogue.json').read_text())
    if (SUP/'experiment-v1/manifest.json').exists():
        catalogue['coordinate_experiment']={
            'version':'oxtr-coordinate-experiment-2026-10-20-v1',
            'manifest':'experiment-v1/manifest.json',
            'methods':'experiment-v1/methods.md',
            'shared_graph_rules':'experiment-v1/overlay-v1.json',
            'shared_graph_rules_csv':'experiment-v1/overlay-v1-fields.csv',
            'overlay_version':json.loads((SUP/'experiment-v1/overlay-v1.json').read_text())['version'],
            'collections':['studies','encode','jaspar'],
            'views':['line','circle','m9'],
            'atlas_sequences':['alpha','beta','gamma','epsilon'],
            'counts':{'studies':128,'encode':255,'jaspar_score_400':9164},
            'status':'Exploratory coordinate mapping with Python, Biopython and Wolfram checks; biological association untested'}
        catalogue['status']='Source catalogue with a reproducible exploratory coordinate experiment'
        catalogue['analysis_status']='Source observations remain unchanged. experiment-v1 supplies fixed DNA-line, full-circle and M9 mappings with computational checks. No enrichment test, new motif scan or biological significance score has been applied.'
        overlay=json.loads((SUP/'experiment-v1/overlay-v1.json').read_text())
        write_record(SUP/'experiment-v1/overlay-v1-fields.csv',overlay)
        assert read_record(SUP/'experiment-v1/overlay-v1-fields.csv')==overlay
    notion_file=ROOT/'research/informatics/oxtr-notion-map.json'
    if notion_file.exists():
        notion=json.loads(notion_file.read_text())
        catalogue['notion']={'page_url':notion['page_url'],'database_id':notion['database_id'],'data_source_id':notion['data_source_id']}
        by_id={r['record_id']:r['url'] for r in notion['records']}
        for r in catalogue['evidence']:
            if r['id'] in by_id:r['notion_url']=by_id[r['id']]
        # The canonical catalogue retains the Notion map; the flat literature
        # table is authored independently and keeps its stable scientific fields.
    write(SUP/'catalogue.json',catalogue)
    write_record(SUP/'catalogue-fields.csv',catalogue)
    assert read_record(SUP/'catalogue-fields.csv')==catalogue
    if (SUP/'human-site-project.json').exists():
        project=json.loads((SUP/'human-site-project.json').read_text())
        write_record(SUP/'human-site-project-fields.csv',project)
        assert read_record(SUP/'human-site-project-fields.csv')==project
    supplement=dict(id=catalogue['version'],title=catalogue['title'],relationship_to_atlas='Shared source data; no sequence-to-biology correspondence assessed',catalogue='pilot-v1/oxtr-tfbs-v1/catalogue.json',json_bundle='pilot-v1/oxtr-tfbs-v1/oxtr-tfbs-json.zip',csv_bundle='pilot-v1/oxtr-tfbs-v1/oxtr-tfbs-csv.zip',publication_count=len(catalogue['publications']),literature_evidence_records=len(catalogue['evidence']))
    if 'harmonisation' in catalogue:
        supplement['harmonisation']=catalogue['harmonisation']
    if 'coordinate_experiment' in catalogue:
        supplement['coordinate_experiment']=catalogue['coordinate_experiment']
    for name in ['dataset','index','m1-m1001']:
        file=DATA/(name+'.json');d=json.loads(file.read_text());d.setdefault('research_supplements',{})['oxtr_tfbs']=supplement;write(file,d)
        write_record(DATA/'csv-shared'/(name+'.csv'),d)
        assert read_record(DATA/'csv-shared'/(name+'.csv'))==d
    with (DATA/'m1-m1001.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    key='shared_oxtr_tfbs_catalogue'
    if key not in fields:fields.append(key)
    for row in rows:row[key]='oxtr-tfbs-v1/catalogue.json'
    with (DATA/'m1-m1001.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    for name in ['M1-M1001-README.txt','M1-M1001-CSV-README.txt']:
        p=DATA/name;text=p.read_text()
        heading='\nOXTR transcriptional-regulation supplement'
        text=text.split(heading)[0]
        p.write_text(text+heading+' — 19 October 2026\n\noxtr-tfbs-v1/ contains a separate source-linked catalogue, literature records, reference DNA, JASPAR motif templates and existing predictions, ENCODE occupancy clusters and experimental provenance, and ORegAnno annotations. Read its README.txt and catalogue metadata for the bounded reference window and interpretation. CSV users have complete tables and typed metadata without needing JSON. The supplement is shared across the Atlas; no modulus-specific biological comparison has been performed. The original 14-study register is unchanged.\n')
        if 'coordinate_experiment' in catalogue:
            with p.open('a') as f:f.write('\nCoordinate experiment — 20 October 2026\n\noxtr-tfbs-v1/experiment-v1/ contains fixed projections for 128 study observations (2 sequence elements and 126 BrainTF occupancy regions), 255 ENCODE regions and 9,164 JASPAR predictions (score >=400), compared visually with all four M9 P24 Atlas sequences. It includes original intervals, source provenance, mappings, methods, Python/Biopython code and Wolfram checks. JSON and CSV exports carry the same experiment information. This is visual exploration with computational verification; biological association is untested.\n')
    # Integrity manifest excludes itself and archives to avoid circular hashes.
    files=[p for p in SUP.rglob('*') if p.is_file() and p.suffix not in ['.zip','.pyc'] and '__pycache__' not in p.parts and p.name!='file-manifest.csv']
    with (SUP/'file-manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['file','bytes','sha256'])
        for p in sorted(files):w.writerow([str(p.relative_to(SUP)),p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()])
    files.append(SUP/'file-manifest.csv')
    csv_files=[p for p in files if p.suffix in ['.csv','.bed','.fa'] or p.name=='README.txt' or ('experiment-v1' in p.parts and p.suffix in ['.py','.wl','.md','.txt']) or ('study-additions-v1' in p.parts and p.suffix=='.txt')]
    json_files=[p for p in files if p.suffix!='.csv' or p.name in ['sources.csv','file-manifest.csv']]
    for name,selected in [('oxtr-tfbs-json.zip',json_files),('oxtr-tfbs-csv.zip',csv_files)]:
        with zipfile.ZipFile(SUP/name,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(selected):z.write(p,str(p.relative_to(SUP)))
            z.write(Path(__file__).with_name('csv_records.py'),'source/csv_records.py')
    common=['dataset.json','index.json','m1-m1001.json','m1-m1001.csv','M1-M1001-README.txt','M1-M1001-CSV-README.txt']
    replacements={f'pilot-v1/{n}':DATA/n for n in common}
    replacements.update({f'pilot-v1/oxtr-tfbs-v1/{p.relative_to(SUP)}':p for p in json_files})
    for p in Path(__file__).parent.glob('*.py'):replacements['source/'+p.name]=p
    replace_zip(DATA/'m1-m1001-data.zip',replacements)
    replacements={n:DATA/n for n in ['m1-m1001.csv','M1-M1001-CSV-README.txt']}
    replacements.update({f'csv-shared/{n}.csv':DATA/'csv-shared'/(n+'.csv') for n in ['dataset','index','m1-m1001']})
    replacements.update({f'oxtr-tfbs-v1/{p.relative_to(SUP)}':p for p in csv_files})
    for p in Path(__file__).parent.glob('*.py'):replacements['source/'+p.name]=p
    replace_zip(DATA/'m1-m1001-csv.zip',replacements)
    print(json.dumps({'supplement_attached':True,'overview_records':len(rows),'supplement_files':len(files),'csv_tables':sum(p.suffix=='.csv' for p in csv_files),'archive_bytes':{n:(SUP/n).stat().st_size for n in ['oxtr-tfbs-json.zip','oxtr-tfbs-csv.zip']}}))


if __name__=='__main__':publish()
