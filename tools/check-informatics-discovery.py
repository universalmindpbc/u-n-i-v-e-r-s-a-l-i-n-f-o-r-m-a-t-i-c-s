"""Check crawl policy, complete HTML discovery, source fidelity and safe links."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import hashlib, json, re, xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'Informatics'
report = json.loads((ROOT / 'research/informatics/discovery-20261026/build.json').read_text())

def groups(text):
    result, agents, rules = [], [], []
    for raw in text.splitlines() + ['User-agent: __end__']:
        line = raw.split('#', 1)[0].strip()
        if ':' not in line: continue
        key, val = [s.strip() for s in line.split(':', 1)]
        if key.lower() == 'user-agent':
            if rules: result.append((agents, rules)); agents, rules = [], []
            agents.append(val.lower())
        elif key.lower() in ('allow', 'disallow'): rules.append((key.lower(), val))
    return result

def allowed(text, agent, url):
    parsed = groups(text)
    selected = [r for a, r in parsed if agent.lower() in a]
    if not selected: selected = [r for a, r in parsed if '*' in a]
    hits = []
    for rules in selected:
        for mode, pattern in rules:
            expr = '^' + re.escape(pattern).replace(r'\*', '.*').replace(r'\$', '$')
            if pattern and re.search(expr, url): hits.append((len(pattern.replace('*', '').replace('$', '')), mode == 'allow'))
    return max(hits)[1] if hits else True

robots = (ROOT / 'robots.txt').read_text()
staging = (ROOT / 'robots-staging.txt').read_text()
agents = ['GPTBot', 'OAI-SearchBot', 'ChatGPT-User', 'Google-Extended', 'Googlebot', 'Applebot-Extended', 'ClaudeBot', 'Claude-SearchBot', 'Claude-User', 'CCBot', 'Bytespider', 'Amazonbot', 'meta-externalagent', 'PerplexityBot', 'Unknown-LLM-Crawler']
routes = ['/informatics', '/informatics/', '/informatics/index.html', '/informatics/read/access.html', '/informatics/pilot-v1/m9.json', '/informatics/pilot-v1/csv-records/m9.csv', '/informatics/exports/manifest.json', '/informatics/exports/part-example.bin', '/informatics/llms.txt']
for agent in agents:
    for route in routes:
        assert allowed(robots, agent, route), (agent, route)
        assert not allowed(staging, agent, route), (agent, route)
for agent in ['GPTBot', 'ClaudeBot', 'Google-Extended']:
    assert not allowed(robots, agent, '/vogue/iam/about.html'), agent
assert allowed(robots, 'Googlebot', '/vogue/iam/about.html')
assert not allowed(robots, 'Googlebot', '/vogue/iam/stakeholders/budget/index.html')

class Document(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]; self.canonical=[]; self.robots=[]; self.scripts=0; self.title=False
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag=='a' and 'href' in a: self.links.append(a['href'])
        if tag=='link' and a.get('rel')=='canonical': self.canonical.append(a['href'])
        if tag=='meta' and a.get('name')=='robots': self.robots.append(a.get('content',''))
        if tag=='script': self.scripts+=1
        if tag=='title': self.title=True

pages=list((SITE/'read').rglob('*.html'))
assert len(pages)==report['pages']
assert hashlib.sha256((SITE/'exports/manifest.json').read_bytes()).hexdigest()==report['datasetManifestSHA256'], 'Reading pages are stale. Run node tools/build-informatics-reading.mjs.'
for record in json.loads((SITE/'pilot-v1/index.json').read_text())['records']:
    digest=hashlib.sha256((SITE/record['file']).read_bytes()).hexdigest()
    assert digest==record['sha256'] and digest in (SITE/f"read/atlas/m{record['modulus']}.html").read_text(), 'Atlas reading page is stale; rebuild reading pages.'
for experiment in json.loads((SITE/'quantum/experiments/index.json').read_text())['experiments']:
    digest=hashlib.sha256((SITE/experiment['json']).read_bytes()).hexdigest()
    assert digest==experiment['json_sha256'] and digest in (SITE/f"read/experiments/{experiment['id']}-{experiment['revision']}.html").read_text(), 'Experiment reading page is stale; rebuild reading pages.'
assert len(list((SITE/'read/atlas').glob('m*.html')))==1001
assert len(list((SITE/'read/publications').glob('*.html')))-1==report['publications']
ns={'s':'http://www.sitemaps.org/schemas/sitemap/0.9'}
urls={n.text for n in ET.parse(ROOT/'sitemap.xml').findall('s:url/s:loc',ns)}
links=0
for p in list(SITE.glob('*.html'))+list((SITE/'molecular').glob('*.html'))+list((SITE/'quantum').glob('*.html'))+pages:
    d=Document();d.feed(p.read_text())
    assert len(d.canonical)==1 and d.title,p
    assert d.robots and all('noindex' not in x and 'nofollow' not in x for x in d.robots),p
    if p in pages:
        assert not d.scripts,p
        assert d.canonical[0] in urls,p
        for href in d.links:
            u=urlsplit(href)
            assert u.scheme in ('','https','http'),(p,href)
            if u.scheme or u.netloc: continue
            target=unquote(u.path)
            if not target: continue
            local=ROOT/target.lstrip('/') if target.startswith('/') else p.parent/target
            assert local.exists(),(p,href)
            links+=1
for s in report['sourceHashes']:
    assert hashlib.sha256((SITE/s['path']).read_bytes()).hexdigest()==s['sha256'],s['path']
assert 'informatics_staging' in (SITE/'.htaccess').read_text()
assert '/informatics/' in (ROOT/'llms.txt').read_text()
assert 'all llm providers' in (SITE/'llms.txt').read_text().lower()
print(json.dumps({'passed':True,'robotDecisions':len(agents)*len(routes)*2,'HTMLPages':len(pages),'localLinks':links,'publicationRecords':report['publications'],'atlasRecords':1001,'experimentRecords':report['experiments'],'sourceFilesMatched':len(report['sourceHashes'])}))
