"""Apply the explicitly reviewed notes to the existing public library sources."""
import csv, hashlib, json, re, unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / 'Informatics/library-data'
CACHE = Path('/private/tmp/um-daytab-20261026')
DATE = '2026-10-26'
notes = list(csv.DictReader((HERE/'curated-notes.tsv').open(), delimiter='\t'))
selected = json.loads((HERE/'selected-pmids.json').read_text())
assert [n['pmid'] for n in notes] == selected and len(set(selected)) == 70
metadata = json.loads((CACHE/'primary-metadata.json').read_text())
verification = {r['pmid']:r for r in json.loads((HERE/'selected-source-verification.json').read_text())}
daytab = {}
for group in json.loads((HERE/'daytab-source-map.json').read_text()).values():
    for item in group:
        daytab.setdefault(item['pmid'], []).append('https://daytab.com/study/'+item['daytabId'])

def load(name): return json.loads((DATA/(name+'.json')).read_text())
def save(name, data): (DATA/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def norm(text): return re.sub(r'[^a-z0-9]','',str(text or '').lower())
def journal_key(text):
    text=''.join(c for c in unicodedata.normalize('NFKD',text) if not unicodedata.combining(c)).lower().replace('&','and')
    text=re.sub(r'^the\s+','',text)
    return norm(re.sub(r'\([^)]*\)|\s*[:=].*$','',text))

baseline = json.loads((ROOT/'research/informatics/release-20261026/library-download.json').read_text())['records']
baseline_dois = {str(r.get('doi') or '').lower() for r in baseline} - {''}
baseline_titles = {norm(r['title']) for r in baseline}
baseline_pmids = set()
for record in baseline:
    if record.get('pmid'): baseline_pmids.add(str(record['pmid']))
    for value in [record['id'],*record.get('aliases',[]),*record.get('urls',[])]:
        baseline_pmids.update(re.findall(r'(?:pmid-|pubmed\.ncbi\.nlm\.nih\.gov/)(\d+)',value))

library, access, conditions = load('library'), load('publication-access'), load('clinical-conditions')
books = {name:load(name+'-edition') for name in ['sound','vision','language']}
new_ids = {'pmid-'+n['pmid'] for n in notes}
# Idempotence: replace this batch only. Earlier archive records are retained.
library['records'] = [r for r in library['records'] if r['id'] not in new_ids]
for book in books.values():
    for chapter in book['chapters']:
        for topic in chapter['topics']:
            topic['papers'] = [p for p in topic['papers'] if p['recordId'] not in new_ids]

sound_interventions = {
 '41450809':'Binaural beats around periodontal surgery','41911215':'1 Hz binaural beats before anaesthesia',
 '40795897':'Binaural-beat music before surgery','40563070':'1 Hz binaural beats, 432/431 Hz carriers',
 '42671585':'5.5–8.5 Hz binaural beats with pink noise after cycling','42307797':'4 Hz binaural beats after a stress challenge',
 '39088370':'Binaural-beat music during endoscopy','32653082':'Combined auditory and visual entrainment during surgery',
 '36011825':'Binaural-beat audio during cataract surgery','38319068':'Binaural beats embedded in familiar music',
 '32564499':'5 Hz binaural beats for chronic pain','33107329':'Binaural beats during cystoscopy or stent removal',
 '32412916':'4 Hz binaural beats, 256/260 Hz carriers','40269359':'Binaural beats or 432 Hz music',
 '36790246':'Alpha binaural-beat music after orthodontic appliance placement','35713444':'Binaural-beat music around bronchoscopy',
 '38982009':'Device-based audiovisual brainwave entrainment','42116620':'BrainTap headset-based brainwave entrainment',
 '40621706':'Binaural auditory stimulation during stroke rehabilitation','35655482':'10, 16 and 40 Hz binaural stimulation during a cognitive task'}
comparators = {
 '41450809':'No auditory intervention','41911215':'No binaural intervention','40795897':'Control without binaural stimulation',
 '40563070':'Silent headphones','42671585':'Sham stimulation','42307797':'Matched-frequency sham stimulation',
 '39088370':'No auditory intervention','32653082':'Standard surgical protocol without audiovisual entrainment',
 '36011825':'Silent headphones','38319068':'The same music without binaural beats','32564499':'Sham stimulation',
 '33107329':'Classical music and silent headphones','32412916':'256 Hz tones in both ears','40269359':'White noise and the other active audio condition',
 '36790246':'Music without binaural beats and no music','35713444':'Plain music and no music','38982009':'Standard behaviour management',
 '42116620':'Music therapy','40621706':'Music during rehabilitation','35655482':'Pure tone and silence',
 '40769315':'Two telephone contacts','40716437':'Waitlist','38227442':'Routine-care waitlist','39758524':'Narrative exposure therapy',
 '42518933':'Treatment as usual','41655996':'Waitlist','37882058':'Treatment as usual','30711853':'Trauma-focused CBT',
 '24880581':'Treatment as usual','32892758':'Imagery rescripting','22322458':'Brief eclectic psychotherapy','39916551':'Imagery rescripting',
 '12883106':'Waitlist','11020090':'Waitlist','19235193':'Usual care','25686304':'Sleep-hygiene education',
 '23541163':'Stress-management education','20876468':'Usual care','22430268':'Standard care','21723049':'Freedom from Smoking programme',
 '29422189':'Usual student support','24647726':'Relapse prevention and usual aftercare','14756612':'Treatment as usual','26586819':'Placebo conditioning, sham mindfulness and book listening',
 '29921143':'Present-centred therapy','16847590':'Attention control','18356284':'Delayed treatment','25397817':'Treatment as usual',
 '31632617':'Nonspecific psychotherapy','39427264':'Waitlist','7591024':'Progressive muscle relaxation and health education',
 '30279666':'Responders versus nonresponders to the same MBSR programme','34907015':'Within-person pre/post comparison; no randomised control',
 '29900257':'Health education','33393484':'Glaucoma medication alone','30256277':'Waitlist','39413359':'Education/support',
 '37546047':'Experienced practitioners not attending the retreat','35818436':'Experienced practitioners remaining in the community',
 '40414454':'Yoga and education arms of the same parent trial','33734493':'Treatment as usual','27723444':'Usual care',
 '34729544':'Health education','38078959':'Delayed instruction','41642087':'Waitlist','42640862':'Comparison group described as active waitlist',
 '29936408':'Waitlist','36006668':'Moodzone active digital control','39808431':'Waitlist','42599846':'Services as usual'}
clinical_interventions = {
 '12883106':'Eight-week workplace mindfulness training','11020090':'Seven-week mindfulness stress reduction',
 '19235193':'Six-week MBSR adapted for breast-cancer survivors','25686304':'Six-week mindful-awareness practices',
 '23541163':'Eight-week MBSR','20876468':'Eight-week mindfulness training','22430268':'Eight-week MBSR',
 '21723049':'Four-week mindfulness training for smoking cessation','29422189':'Mindfulness Skills for Students',
 '24647726':'Mindfulness-based relapse prevention','14756612':'Mindfulness-based cognitive therapy',
 '26586819':'Four days of mindfulness meditation training','36006668':'Unguided Headspace mindfulness app',
 '39808431':'Digital meditation, ten minutes daily for eight weeks','42599846':'Eight-week hybrid digital mindfulness programme'}
molecular = {
 '30279666':(['Epigenetics'],['FKBP5','SLC6A4'],'Blood DNA methylation','MBSR','22 veterans'),
 '34907015':(['Gene expression'],[],'Blood transcriptomics','Eight-day intensive meditation retreat',None),
 '29900257':(['Gene expression'],[],'Affymetrix blood-expression microarray dataset','Mindful-awareness practice','60 older adults with mild cognitive impairment'),
 '33393484':(['Gene expression'],['NOS1','NOS3'],'Trabecular-meshwork gene expression and intraocular pressure','Three-week meditation adjunct','60 glaucoma patients'),
 '30256277':(['Gene expression','Molecular biomarkers'],[],'Gene-expression and stress-biomarker measurements with intraocular pressure','Mindfulness meditation','90 glaucoma patients'),
 '39413359':(['Genetics & genomics'],['DRD2','COMT','ANKK1'],'Genotype moderation of cognitive response; not gene expression','MBSR','128 breast-cancer survivors'),
 '37546047':(['Gene expression'],['IL6','IL1B','TNF'],'Inflammatory and antioxidant messenger RNA','Sudarshan Kriya Yoga retreat','97 experienced practitioners'),
 '35818436':(['Gene expression'],['TNF'],'Blood gene expression, including inflammatory and chromatin-related signals','Three-week meditation retreat','28 retreat participants and 34 community controls'),
 '40414454':(['Gene expression','Brain & nervous system'],[],'Brain connectivity and stress-related transcriptional regulation','Eight-week meditation, yoga or education','94 participants in a parent-trial subset')}
nonrandom = {'30279666','34907015','37546047','35818436'}
secondary = {'39916551':'IREM; parent publication PMID 32892758','31632617':'Parent publication PMID 29921143',
 '25397817':'Secondary publication from the 146-veteran mantram trial','30279666':'NCT01548742; responder subsample',
 '40414454':'NCT01308970; parent-trial overlap with PMID 41020112','39413359':'NCT02786797; exploratory genetic moderator analysis',
 '29900257':'Data publication from the mindful-awareness versus health-education trial'}
conditions_by_pmid = {}
for key, ids in {
 'anxiety':'41450809 40795897 39088370 36011825 40269359 35713444 38982009 19235193 23541163',
 'ptsd':'40769315 40716437 38227442 42518933 32892758 22322458 39916551 29921143 18356284 25397817 31632617 30279666 33734493',
 'trauma':'24880581 37882058 39758524 27723444','depression':'30711853 19235193 20876468 14756612 40621706',
 'psychosis':'38227442 37882058','bipolar':'24880581','eating-disorders':'41655996',
 'stress':'42307797 11020090 12883106 29422189 39427264 41642087 42640862 29936408 36006668 39808431 42599846 38078959',
 'sleep':'38319068 25686304','substance-use':'21723049 24647726'}.items():
    for pmid in ids.split(): conditions_by_pmid.setdefault(pmid,[]).append(key)

additions=[]
for note in notes:
    pmid=note['pmid']; raw=metadata[pmid]['metadata']; checked=verification[pmid]
    assert checked['pmidMatches'] and not checked.get('metadataError')
    doi=raw.get('doi'); assert not doi or checked['doiMatches']
    assert pmid not in baseline_pmids and (not doi or doi.lower() not in baseline_dois) and norm(raw['title']) not in baseline_titles, pmid
    assert not any('retract' in t.lower() for t in raw.get('pubTypeList',{}).get('pubType',[])), pmid
    urls=['https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/']
    if doi: urls.append('https://doi.org/'+doi)
    if raw.get('pmcid'): urls.append('https://pmc.ncbi.nlm.nih.gov/articles/'+raw['pmcid']+'/')
    cats,targets=['Clinical & self-report'],[]
    intervention=sound_interventions.get(pmid) or clinical_interventions.get(pmid)
    if note['collection']=='vision': intervention='Therapist-delivered EMDR'
    if note['topic']=='i' and note['collection']=='language':
        intervention='Transcendental Meditation' if 'transcendental' in raw['title'].lower() or pmid=='7591024' else 'Mantram repetition programme'
    evidence={'intervention':intervention,'comparator':comparators[pmid],'evidenceType':'Clinical or experimental outcomes',
       'sourceURLs':urls,'checkedAt':DATE,'reviewScope':'Primary publication abstract and bibliographic identity checked; comprehensive full-text risk-of-bias and Jadad assessment pending.',
       'discoverySources':daytab.get(pmid,[])+['https://europepmc.org/article/MED/'+pmid],
       'selectionPolicy':'Success-focused discovery selection, with material limitations retained; not an exhaustive systematic review.'}
    if pmid in molecular:
        cats,targets,measurement,intervention,sample=molecular[pmid]
        evidence.update(intervention=intervention,measurement=measurement,evidenceType=cats[0],measuredGenes=targets)
        if sample:evidence['sample']=sample
    if pmid in {'42671585','35655482'}: cats=['Brain & nervous system']
    if pmid in {'42307797','42116620','39427264','12883106'}: cats += ['Molecular biomarkers']
    if pmid in secondary: evidence['relationshipNote']=secondary[pmid]
    if pmid=='29900257':
        evidence['datasetAccession']='GSE108215'
        evidence['datasetURL']='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108215'
        evidence['reviewScope']='Abstract, full-text study design and GEO accession checked; independent reanalysis and comprehensive risk-of-bias assessment pending.'
    if pmid=='30279666':
        evidence['correctionURL']='https://doi.org/10.3389/fpsyt.2021.642245'
        evidence['reviewScope']='Abstract, full-text participant selection and limitations, and 2021 corrigendum checked.'
        note['finding']='In 22 veterans receiving MBSR, responders showed increased FKBP5 methylation while nonresponders showed decreased methylation, following the corrected direction reported in the 2021 corrigendum.'
        note['limitation']='The exploratory responder comparison was not randomised; its multiplicity-adjusted p value was 0.052. SLC6A4 did not show a corresponding positive result. Blood DNA methylation is distinct from brain gene expression.'
        urls.append(evidence['correctionURL'])
    abstract=raw.get('abstractText','')
    trialids=re.findall(r'\b(?:NCT\d{8}|ISRCTN\d{8}|ACTRN\d{14}|DRKS\d{8})\b',abstract)
    if trialids:evidence['trialId']='; '.join(dict.fromkeys(trialids))
    journal=raw.get('journalInfo',{}).get('journal',{}).get('title','')
    citation={'count':raw.get('citedByCount',0),'source':'Europe PMC','retrievedAt':DATE,'url':'https://europepmc.org/article/MED/'+pmid}
    assessment={'category':'peer-reviewed','peerReviewed':True,'basis':'Reported design checked against the primary publication abstract. Blinding terminology reflects the source report, not an independent risk-of-bias rating.','sources':urls,'assessedAt':DATE,'registeredMaskingBasis':False}
    record={'id':'pmid-'+pmid,'pmid':pmid,'doi':doi,'title':raw['title'],'journal':journal,'year':int(raw['pubYear']),
      'authors':raw.get('authorString',''),'collections':[note['collection']],'origin':'update','design':note['design'],
      'categories':cats,'targets':targets,'priority':0 if pmid in molecular else 1,'finding':note['finding'],'limitation':note['limitation'],
      'evidence':evidence,'urls':urls,'jadad':{'status':'not-applicable' if pmid in nonrandom else 'unassessed','score':None},
      'citationCount':citation['count'],'citationSnapshot':citation,'studyDesignAssessment':assessment,'archiveEntries':[],
      'verification':'PMID identity checked; DOI cross-checked with Crossref where supplied. Primary abstract reviewed.',
      'verifiedAt':DATE,'curationBatch':'daytab-20261026'}
    library['records'].append(record); additions.append(record)
    access['records'][record['id']]={'urls':urls,'doi':doi,'pmid':pmid,'publicationKey':'doi:'+doi.lower() if doi else 'pmid:'+pmid,'journalKey':journal_key(journal),'journalLabel':journal}
    topic=next(t for c in books[note['collection']]['chapters'] for t in c['topics'] if t['id']==note['topic'])
    paper={k:record[k] for k in ['title','journal','year','authors','design','urls','doi','pmid','targets','categories','citationCount','citationSnapshot','studyDesignAssessment','evidence']}
    paper.update(id=note['collection']+'-'+note['topic']+'-'+record['id'],recordId=record['id'],quotes=[],sourcePages=[],summary=record['finding'],limitation=record['limitation'])
    topic['papers'].append(paper)
    if pmid in conditions_by_pmid:
        conditions['records'][record['id']]=[{'id':condition,'evidence':[{'field':'curated primary-source summary','text':record['finding'],'matchedText':condition}]} for condition in conditions_by_pmid[pmid]]

library['reviewedAt']=DATE; library['summary']['records']=len(library['records'])
library['summary']['updatedRecords']=sum(r.get('origin')=='update' for r in library['records'])
access['checkedAt']=DATE
for name,book in books.items():
    papers=[p for c in book['chapters'] for t in c['topics'] for p in t['papers']]
    book['paperCount']=len(papers)
    book['uniquePaperCount']=len({('doi:'+p['doi'].lower()) if p.get('doi') else ('pmid:'+str(p['pmid'])) if p.get('pmid') else p['recordId'] for p in papers})
    save(name+'-edition',book)
save('library',library);save('publication-access',access);save('clinical-conditions',conditions)
(HERE/'additions.json').write_text(json.dumps({'selectionPolicy':'Success-focused selection with original-source verification and explicit limitations; not a systematic review.','baselinePublications':len(baseline),'addedPublications':len(additions),'records':additions},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'added':len(additions),'collections':dict(Counter(r['collections'][0] for r in additions)),'dois':sum(bool(r['doi']) for r in additions),'jadad':dict(Counter(r['jadad']['status'] for r in additions))}))
