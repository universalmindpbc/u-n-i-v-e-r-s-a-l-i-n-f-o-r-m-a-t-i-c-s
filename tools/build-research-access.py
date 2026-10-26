#!/usr/bin/env python3
"""Build publication access and journal metadata without changing quotations or scores."""
import json,re,html,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'research/informatics/library';C=Path(os.environ.get('RESEARCH_LINK_CACHE','/private/tmp/research-links-20261022'))
catalog=json.loads((P/'publication-catalog-20261022.json').read_text())['records'];overrides=json.loads((P/'publication-link-overrides-20261022.json').read_text());http={d['url']:d for d in json.loads((C/'http.json').read_text())};out={};audit=[]
def key(s):return re.sub('[^a-z0-9]','',re.sub(r'\([^)]*\)|\s*[:=].*$','',html.unescape(s).lower().replace('&','and').removeprefix('the ')))
def reachable(u):
 d=http.get(u,{})
 return d.get('status')==200 and (d.get('bytes',0)>500 or d.get('curlExit')==63 and 'pdf' in d.get('contentType','')) and not re.search('captcha|just a moment|access denied|not found|unavailable|^error|checking your browser|robot',d.get('title',''),re.I)
def known_dead(u):return http.get(u,{}).get('status') in [404,410]
for id,d in catalog.items():
 o=overrides.get(id,{})
 urls=list(dict.fromkeys(o.get('urls',[])+d['candidates']));ok=[u for u in urls if reachable(u)];original=d['originalUrls']
 # Stable indexed records remain valid even when a publisher blocks this audit client.
 # Prioritize an accessible original link, then a verified replacement, then an indexed DOI/PMID.
 stable=[]
 pmid=o.get('pmid') or d.get('pmid');doi=o.get('doi') or d.get('doi')
 if pmid:stable.append('https://pubmed.ncbi.nlm.nih.gov/'+str(pmid)+'/')
 if doi:stable.append('https://doi.org/'+doi)
 if o.get('urls'):selected=o['urls']
 elif reachable(original[0]):selected=[original[0]]+ok+stable
 elif ok:selected=ok+stable
 elif stable:selected=stable
 else:selected=[u for u in original if not known_dead(u)] or original
 selected=list(dict.fromkeys(u for u in selected if u.startswith(('https://','http://'))))
 if any(not known_dead(u) for u in selected):selected=[u for u in selected if not known_dead(u)]
 # Keep a compact set of usable alternate destinations; retain every original in the audit.
 selected=selected[:3]
 jl=o.get('journalLabel') or d['journalLabel'];jk=key(jl) if jl else ''
 out[id]={'publicationKey':o.get('publicationKey'),'urls':selected,'journalKey':jk,'journalLabel':jl,'doi':doi,'pmid':pmid}
 status='http-accessible' if any(reachable(u) for u in selected) else 'indexed-access-restricted' if stable else 'unresolved'
 if o.get('urls'):status='archived-original' if 'web.archive.org' in selected[0] else 'manually-located'
 if status=='unresolved':
  codes={http.get(u,{}).get('status') for u in selected}
  challenge=any(re.search('captcha|just a moment|access denied|checking your browser|robot',http.get(u,{}).get('title',''),re.I) for u in selected)
  status='dead-link' if codes and codes <= {404,410} else 'access-restricted' if challenge or codes & {403,402,418,429,202,203} else 'unconfirmed'
 audit.append({'id':id,'title':d['title'],'status':status,'originalUrls':original,'urls':selected,'primaryChanged':selected[0]!=original[0],'evidence':d['evidence']+([{'source':'Manual publication search','detail':o['evidence']}] if 'evidence' in o else []),'checks':[http.get(u,{'url':u,'status':None}) for u in selected]})
(R/'Informatics/library-data/publication-access.json').write_text(json.dumps({'checkedAt':'2026-10-22','records':out},ensure_ascii=False,indent=2)+'\n')
from collections import Counter
summary={'records':len(audit),'distinctUrlsChecked':len(http),'primaryLinksChanged':sum(d['primaryChanged'] for d in audit),'statuses':dict(Counter(d['status'] for d in audit))}
edition=json.loads((R/'Informatics/library-data/sound-edition.json').read_text())
papers=[p for c in edition['chapters'] for t in c['topics'] for p in t['papers']]
sound_ids={p['recordId'] for p in papers};aliases={a for p in papers for a in p.get('aliases',[])}
library=json.loads((R/'Informatics/library-data/library.json').read_text())['records']
published_ids={r['id'] for r in library if r['id'] not in aliases and any(c!='sound' or r['id'] in sound_ids or r.get('origin') in ['update','portal'] for c in r['collections'])}
published=[d for d in audit if d['id'] in published_ids]
summary['publishedRecords']={'recordsBeforePublicationDeduplication':len(published),'primaryLinksChanged':sum(d['primaryChanged'] for d in published),'statuses':dict(Counter(d['status'] for d in published))}
method='All original URLs, stable DOI/PMID links and located replacements were checked with redirect-following HTTP requests. Replacements retain the same publication identity. Captchas, publisher access restrictions and timeouts are recorded separately from 404/410 failures. Archive fallbacks preserve the original pages. HTTP access is not full-paper scientific appraisal.'
(P/'publication-link-audit-20261022.json').write_text(json.dumps({'checkedAt':'2026-10-22','method':method,'summary':summary,'records':audit},ensure_ascii=False,indent=2)+'\n')
(P/'publication-link-http-20261022.json').write_text(json.dumps(list(http.values()),ensure_ascii=False,indent=2)+'\n')
print(summary)
