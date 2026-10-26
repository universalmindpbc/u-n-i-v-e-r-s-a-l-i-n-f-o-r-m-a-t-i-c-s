"""Render verified archive receipts without changing the scientific dataset.

Usage: python tools/build-informatics-provenance.py VERIFIED_BUNDLE_DIRECTORY
"""
from pathlib import Path
from html import escape
import hashlib,json,shutil,sys

ROOT=Path(__file__).resolve().parents[1]
bundle=Path(sys.argv[1]).resolve()
site=ROOT/'Informatics/read'
out=site/'archive/20261026'
out.mkdir(parents=True,exist_ok=True)
# Public certificates use a text download so the private-key ignore rule remains intact.
(out/'timestamp-chain.pem').unlink(missing_ok=True)
read=lambda name:json.loads((bundle/name).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read('manifest.json');digest=sha(bundle/'manifest.json')
syn=read('synapse-verification.json');desci=read('desci-archive-receipt.json');stamp=read('timestamp-verification.json')
assert all(x['manifestSHA256']==digest for x in [syn,desci,stamp])
assert syn['status']=='uploaded-and-download-verified' and all(x['downloadVerified'] for x in syn['files'])
assert desci['status']=='verified-private-IPFS-archive' and all(x['verified'] for x in desci['files'])
assert stamp['signatureAndTrustChainVerified'] and stamp['queryNonceVerified']
assert desci['published'] is False
# Archive receipts are documents, outside the scientific export inventory.
# Keep exact manifest bytes: its digest is the timestamped commitment.
for source,destination in [('manifest.json','manifest.json.txt'),('integrity.json','integrity.json.txt'),('source-files.json','source-files.json.txt'),('desci-storage-map.json','desci-storage-map.json.txt'),('manifest.json.ots','manifest.json.ots'),('manifest.tsq','manifest.tsq'),('manifest.tsr','manifest.tsr'),('timestamp-chain.pem','timestamp-chain.pem.txt'),('VERIFY-TIMESTAMP.txt','VERIFY-TIMESTAMP.txt')]:
    shutil.copyfile(bundle/source,out/destination)
public_syn={k:syn[k] for k in ['project','folder','permissions','manifestSHA256','status','checkedAt']}
public_syn['files']=[{k:x[k] for k in ['name','id','version','bytes','sha256','downloadVerified','createdOn','modifiedOn']} for x in syn['files']]
public_desci={k:desci[k] for k in ['provider','draftId','title','published','manifestSHA256','status','files','storageRepresentation','completedAt']}
for k in ['manifestCID','prepublishCompleted','prepublishError','finalFileTreeVerifiedAt','directoryCID','combinedManifestRecovered','recoveryMethod','manifestCIDIndependentlyVerified','directoryCIDIndependentlyComputedAndProviderVerified','documentServiceRestored','recoveredAt']:
    if k in desci:public_desci[k]=desci[k]
for name,data in [('synapse-verification.json.txt',public_syn),('desci-verification.json.txt',public_desci),('timestamp-verification.json.txt',stamp)]:
    (out/name).write_text(json.dumps(data,indent=2)+'\n')
shutil.copyfile(ROOT/'robots.txt',out/'production-robots.txt')
if desci.get('combinedManifestRecovered'):
    shutil.copyfile(bundle/'desci-recovered-manifest.canonical.json',out/'desci-recovered-manifest.json.txt')
    shutil.copyfile(bundle/'desci-directory-metadata.json',out/'desci-directory-metadata.json.txt')
    shutil.copyfile(ROOT/'research/informatics/discovery-20261026/desci-recovery.md',out/'desci-recovery.txt')

prefix='archive/20261026/'
def link(name,label,download=None):
    return '<a href="'+prefix+name+'"'+(' download="'+escape(download)+'"' if download else '')+'>'+escape(label)+'</a>'
def code(value):return '<code>'+escape(str(value))+'</code>'
def definition(items):return '<dl>'+''.join('<dt>'+escape(k)+'</dt><dd>'+v+'</dd>' for k,v in items)+'</dl>'
scope=manifest['scope']
body='<h1>Archive and timestamp record</h1>'
body+='<p class="intro">A verifiable snapshot of Universal Informatics research data and public source, archived on 26 October 2026.</p>'
body+='<p>The complete JSON and CSV exports preserve '+str(scope['structuredScientificSources'])+' structured scientific sources. The archive also records '+str(scope['publicSourceFiles'])+' public code and method files, with URLs and hashes for '+format(scope['linkedAssets'],',')+' linked assets.</p>'
body+=definition([('Manifest SHA-256',code(digest)),('Dataset content version',code(manifest['datasetContentVersion'])),('Archived source commit',code(manifest['sourceCommit']))])
body+='<p>'+link('manifest.json.txt','Download exact archive manifest','manifest.json')+' · '+link('source-files.json.txt','Source file inventory','source-files.json')+'</p>'
body+='<h2>Synapse archive</h2><p>The dated snapshot is stored in the Universal Informatics Synapse project. All '+str(len(syn['files']))+' files were retrieved and verified against their local SHA-256 hashes. The receipt records file identifiers, versions and provider timestamps.</p>'
body+='<p>Project access remains owner-only. This archive record is public; the stored research package requires an authorised Synapse account.</p>'
body+='<p><a href="https://www.synapse.org/Synapse:'+escape(syn['folder'])+'">Open the Synapse snapshot ↗</a> · '+link('synapse-verification.json.txt','Read the Synapse verification receipt','synapse-verification.json')+'</p>'
body+='<h2 id="desci">DeSci and IPFS</h2><p>'+escape(desci['title'])+' is stored as a private DeSci draft. Each of its '+str(len(desci['files']))+' archived files has an IPFS content identifier independently recomputed from the exact local bytes and matched to the provider’s stored file tree.</p>'
body+='<p>The complete CSV is preserved as ordered 8 MiB segments to meet the upload limit. The storage map records their order and hashes; concatenating those segments recovers the original compressed CSV byte for byte.</p>'
body+=definition([('DeSci draft',code(desci['draftId'])),('Publication state','Private draft; no public signed publication has been made.')])
if desci.get('combinedManifestRecovered'):
    body+='<p>The combined archive manifest has been recovered and stored through DeSci’s supported draft-update API. Its CID was independently recomputed from the saved manifest bytes. The reconstructed directory CID matches DeSci’s stored 33-file directory.</p>'
    body+=definition([('DeSci archive manifest CID',code(desci['manifestCID'])),('Archive directory CID',code(desci['directoryCID']))])
    body+='<p>'+link('desci-recovered-manifest.json.txt','Download the recovered DeSci manifest','desci-recovered-manifest.json')+' · '+link('desci-directory-metadata.json.txt','Stored directory verification','desci-directory-metadata.json')+'</p>'
    body+='<p>DeSci’s live document editor still reports a missing document reference. The archive manifest and files are intact; editor-service restoration remains a provider issue. A repair request was submitted to DeSci support on 26 October 2026 and receipt was confirmed. Restoration remains pending their response. No signed public DeSci publication has been made.</p>'
    body+='<p>'+link('desci-recovery.txt','Read the diagnosis and recovery record','desci-recovery.txt')+'</p>'
else:
    body+='<p>DeSci’s manifest service currently returns an empty draft document, so preparation of a combined archive manifest remains incomplete. The individual files are saved and CID-verified. A signed DeSci publication has not been made.</p>'
body+='<p>An IPFS identifier detects content changes. Continued availability depends on storage and pinning; this record does not claim permanent storage.</p>'
body+='<p><a href="https://publish.desci.com/">Open DeSci Publish ↗</a> · '+link('desci-verification.json.txt','File CIDs and verification receipt','desci-verification.json')+' · '+link('desci-storage-map.json.txt','CSV reconstruction map','desci-storage-map.json')+'</p>'
body+='<h2>Verified timestamp</h2><p>DigiCert signed the manifest hash at <strong>26 October 2026, 10:29:52 am Sydney time</strong> (25 October 2026, 23:29:52 UTC). Verification passed for the signature, certificate chain, original request nonce and exact manifest bytes, using an independent system CA trust store.</p>'
body+='<p>The manifest binds the archived files to that timestamp. This establishes evidence of existence and integrity at the recorded time; it does not assess scientific validity or determine legal ownership.</p>'
body+='<ul>'+''.join('<li>'+link(f,label,download)+'</li>' for f,label,download in [('manifest.tsr','Signed RFC3161 timestamp response','manifest.tsr'),('manifest.tsq','Original timestamp request','manifest.tsq'),('timestamp-chain.pem.txt','Timestamp certificate chain','timestamp-chain.pem'),('timestamp-verification.json.txt','Verification receipt','timestamp-verification.json'),('VERIFY-TIMESTAMP.txt','Independent verification instructions','VERIFY-TIMESTAMP.txt')])+'</ul>'
body+='<p><a href="https://knowledge.digicert.com/general-information/rfc3161-compliant-time-stamp-authority-server">DigiCert timestamp authority documentation ↗</a></p>'
body+='<h2>Bitcoin timestamp status</h2><p>An additional OpenTimestamps proof has been submitted. Its calendar attestations are present; Bitcoin confirmation remains pending. No Bitcoin-verified timestamp is claimed for this release.</p><p>'+link('manifest.json.ots','Download the OpenTimestamps proof','manifest.json.ots')+' · <a href="https://opentimestamps.org/">OpenTimestamps verification ↗</a></p>'
body+='<h2>AI access policy</h2><p>The prepared production policy permits automated reading and training by all LLM providers on Universal Mind-owned public Informatics content. Third-party attribution and licence terms remain attached to the source records. Staging remains excluded from crawling and indexing.</p><p><a href="access.html">Read the access and reuse policy</a> · '+link('production-robots.txt','Inspect the prepared production robots policy')+'</p>'
page=site/'provenance.html'
old=page.read_text();head=old.split('<main>')[0];footer=old.split('</main>',1)[1]
if 'rel="icon"' not in head:head=head.replace('</head>','<link rel="icon" href="/vogue/iam/favicon.png"></head>')
page.write_text(head+'<main>'+body+'</main>'+footer)
report={'manifestSHA256':digest,'synapseFiles':len(syn['files']),'desciFiles':len(desci['files']),'timestampVerified':True,'bitcoinConfirmed':False,'files':[{'path':str(p.relative_to(site)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir())]}
(ROOT/'research/informatics/discovery-20261026/archive-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='files'}))
