"""Bind verified Wolfram evidence and the shared literature to the expanded data."""
import hashlib, json
from pathlib import Path
from expand_atlas import ROOT, OUT, publish, signature

proof_dir = ROOT / 'research/informatics/data/m1-m1001-v1'
batches = []
for lo, hi in [(x, x + 99) for x in range(1, 901, 100)] + [(901, 1001)]:
    code = (proof_dir / f'verify-{lo}-{hi}.wl').read_bytes()
    evidence = json.loads((proof_dir / f'wolfram-{lo}-{hi}.json').read_text())
    assert evidence['verification_code_sha256'] == hashlib.sha256(code).hexdigest()
    assert not evidence.get('isError')
    text = next(x['text'] for x in evidence['content'] if x['type'] == 'text')
    result = json.loads(json.loads(text.split('=', 1)[1].strip()))
    assert result['passed'] and result['failures'] == [] and result['range'] == [lo, hi]
    batches.append({**result, 'input_sha256': evidence['verification_code_sha256']})
assert sum(b['records'] for b in batches) == 1001
scope = ('Native Pisano period; complete Alpha, primary Gamma indices, Beta interval sums and residues, '
         'primary equilateral-triangle and hexagon indices. Independent integer Fibonacci evaluation '
         'and SHA-256 comparisons. Extended catalogue, Chevron and native/cycle-based Epsilon are Python-checked.')
proof = {'version': 'm1001-core-verification-v1', 'records_checked': 1001,
         'comparisons_passed': sum(b['comparisons'] for b in batches), 'passed': True,
         'scope': scope, 'batches': batches,
         'python_scope': 'Full native sequences, first state return, every formation interval reconstruction, regular polygon spacing and Epsilon routes. All 111 M9-family projections preserve original alignment.'}
(OUT / 'm1001-verification.json').write_text(json.dumps(proof, indent=2) + '\n')
register = json.loads((ROOT / 'research/informatics/research-register.json').read_text())
register['scope'] = register['scope'].replace('M1–M48', 'M1–M1001')
register['version'] = '2026-10-18.1'
for path in (ROOT / 'research/informatics/research-register.json', OUT / 'research-register.json'):
    path.write_text(json.dumps(register, ensure_ascii=False, indent=2) + '\n')
(ROOT / 'Informatics/science-studies.js').write_text("'use strict';\n// Generated from research/informatics/research-register.json.\nwindow.InformaticsResearchRegister = " + json.dumps(register, ensure_ascii=False, indent=2) + ';\nwindow.InformaticsStudies = window.InformaticsResearchRegister.publications;\n')
for m in range(1, 1002):
    path = OUT / f'm{m}.json'
    d = json.loads(path.read_text())
    d['research_register'] = register
    d['research']['background_literature'].update(register_version=register['version'], scope=register['scope'])
    if m > 48:
        d['verification'].update(wolfram='Passed', wolfram_scope=scope, evidence='m1001-verification.json')
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2 if m <= 48 else None,
                               separators=None if m <= 48 else (',', ':')) + '\n')
publish()
