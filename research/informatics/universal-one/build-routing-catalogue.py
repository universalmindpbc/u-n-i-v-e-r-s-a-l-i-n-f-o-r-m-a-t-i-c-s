"""Regenerate specialty routing from approved metadata, without changing it."""
import json, pathlib, re, unicodedata, sys
root=pathlib.Path(sys.argv[1]).resolve() if len(sys.argv)>1 else pathlib.Path(__file__).resolve().parents[3]
data=root/'Informatics/library-data'
terms=set()
def add(value):
    value=unicodedata.normalize('NFKC',str(value)).lower()
    value=re.sub(r'[–—−-]',' ',value)
    value=re.sub(r'[^\w.+]+',' ',value).strip()
    value=re.sub(r'\s+',' ',value)
    if len(value)>2: terms.add(value)
aliases='''gene expression|genes|transcription|transcriptomics|transcriptomic|epigenetics|epigenetic|genomics|genomic|dna methylation|microRNA|microRNAs|EMDR|eye movement desensitization and reprocessing|eye movement desensitisation and reprocessing|bilateral stimulation|bilateral tapping|saccadic|mantra|mantras|transcendental meditation|TM|mindfulness|mindfulness based stress reduction|MBSR|MBCT|meditation|meditating|yoga|molecular hydrogen|hydrogen rich water|H2 water|binaural|binaural beats|isochronic|brain entrainment|neural entrainment|transcranial magnetic stimulation|TMS|rTMS|Schumann|Schumann resonance|human resonances|human resonance|5 to 8 hertz|5 to 8 hz|5 8 hertz|5 8 hz|five to eight hertz|7.83 hertz|7.83 hz|seven point eight three hertz|528 hz|528 hertz|theta|alpha waves|music|music making|music listening|sound therapy|sound healing|sound frequencies|oxytocin|serotonin|dopamine|cortisol|interleukin 6|IL 6|NMDA receptor|Fibonacci|Pisano|Phi ratio|golden ratio|M9 family|nested sequences|molecular docking|pi stacking|mitosis|circadian|baroreflex|Mayer waves|HRV|heart rate variability|psychotherapy|psychology|talking therapies|nonviolent communication|non violent communication|emotional intelligence|digital health|population statistics|near infrared|photobiomodulation|virtual reality|art therapy|quantum computing'''
for value in aliases.split('|'): add(value)
# Short, unambiguous gene symbols are retained even when only two characters.
terms.add('tm')
records=json.loads((data/'library.json').read_text())['records']
for record in records:
    for key in ('targets','categories','conditions'):
        for value in record.get(key,[]): add(value)
    for value in re.split(r'[,;&/]| and ',record.get('strand','')): add(value)
# Include the Molecular Atlas symbols and receptor names as well as library tags.
for record in json.loads((root/'Informatics/molecular/data/catalogue.json').read_text())['records']:
    for value in [record['id'],record.get('name',''),record.get('pdb',''),*record.get('genes',[])]: add(value)
for name in ('sound','vision','language','general'):
    edition=json.loads((data/(name+'-edition.json')).read_text())
    for chapter in edition['chapters']:
        for item in [chapter,*chapter.get('topics',[])]:
            add(item['title'])
            for value in re.split(r'[,;:&/]| and ',item['title']): add(value)
terms.difference_update({'who','context','methods','general','research','outcomes','health','biology','trust','learning','memory','language','vision','sound','human','clinical','self report'})
result={'version':'research-specialties-20261024','terms':sorted(terms)}
target=root/'cloudflare/informatics-research-topics.json'
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(f'{len(terms)} specialty terms generated from library metadata and explicit aliases.')
