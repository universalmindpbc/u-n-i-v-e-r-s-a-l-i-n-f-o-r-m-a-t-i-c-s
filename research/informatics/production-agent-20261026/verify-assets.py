import concurrent.futures,hashlib,json,subprocess,urllib.request,re,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE='https://universalmind.tv/'
paths=subprocess.check_output(['git','diff','--name-only','97962ec3a0c86e55153ba43eb1828cc1d4e11e69','HEAD','--','Informatics','llms.txt','sitemap.xml'],text=True,cwd=ROOT).splitlines()
paths=[p for p in paths if (ROOT/p).is_file() and not p.endswith(('.php','.htaccess','.bin'))]
paths+=['robots.txt']
def check(path):
 route=path.replace('Informatics/','informatics/',1)
 req=urllib.request.Request(BASE+route+'?release=production-20261026',headers={'User-Agent':'Universal-Informatics-Release-QA'})
 with urllib.request.urlopen(req,timeout=90) as response:
  body=response.read()
 raw=hashlib.sha256(body).hexdigest()
 if path.endswith('.html'): body=re.sub(rb'<script[^>]+src="https://static\.cloudflareinsights\.com/beacon[^>]*></script>\n?',b'',body)
 if path.endswith('.html'):
  def decode_email(m):
   data=bytes.fromhex(m.group(1).decode());return html.escape(bytes(v^data[0] for v in data[1:]).decode(),quote=False).encode()
  body=re.sub(rb'<a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="([a-f0-9]+)">.*?</a>',decode_email,body)
  body=re.sub(rb'<script[^>]*src="/cdn-cgi/scripts/[^"]*/cloudflare-static/email-decode.min.js"[^>]*></script>',b'',body)
 actual=hashlib.sha256(body).hexdigest()
 expected=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
 if actual!=expected: raise AssertionError(path)
 return {'path':path,'sha256':actual,'served_sha256':raw,'cloudflare_html_transforms_reversed':raw!=actual}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 records=list(pool.map(check,paths))
out=Path(__file__).resolve().parent/'production';out.mkdir(exist_ok=True)
(out/'asset-proof.json').write_text(json.dumps({'passed':True,'origin':BASE,'files':records},indent=2)+'\n')
print(json.dumps({'passed':True,'files':len(records)}))
