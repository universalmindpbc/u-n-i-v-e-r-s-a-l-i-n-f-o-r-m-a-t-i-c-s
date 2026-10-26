"""Resumable Notion payloads derived from the published Atlas JSON."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Informatics/pilot-v1'
WEB = 'https://staging.universalmind.tv/informatics/'
PAPERS = json.loads((ROOT / 'research/informatics/notion-research-register.json').read_text())['papers']

def seq(values):
    return ' → '.join(map(str, values))

def prop(values, section):
    text = seq(values)
    return text if len(text) <= 1800 else f'{len(values)} values; complete ordered sequence in the {section} section and linked JSON.'

def sequence_content(values):
    return '\n\n'.join(seq(values[i:i+150]) for i in range(0, len(values), 150))

def page(m):
    d = json.loads((OUT / f'm{m}.json').read_text())
    b, g, e = d['beta'], d['gamma'], d['epsilon']
    ep = d.get('epsilon_path')
    shapes = {s['name']: s for s in d['geometry_summary']['shapes']}
    p = {'Name': d['name'], 'Modulus': m, 'Pisano period': d['pisano_period'],
         'Record ID': d['id'], 'M9 family': '__YES__' if m % 9 == 0 else '__NO__',
         'Alpha': prop(d['alpha']['display_values'], 'Alpha'),
         'Beta': prop(b['display_cycle'], 'Beta') if b else 'Not defined',
         'Gamma': prop(g['clockwise_values'], 'Gamma'),
         'Epsilon': seq(ep['native_values']) + ' (reduces to ' + seq(ep['reduced_m9_values']) + ')' if ep else seq(e['traversal_values']) if e else 'No qualifying native 3–6–9 path',
         'Gamma node indices': prop(g['alpha_indices'], 'Gamma'),
         'Gamma clock positions': prop(g['clock_positions'], 'Gamma'),
         'Epsilon clock route': seq(ep['clock_positions']) if ep else seq(e['clock_positions']) if e else '',
         'Native doubling': 'Present' if b and b['native_doubling'] else 'Absent' if b else 'Not defined',
         'Geometry': json.dumps([s['name'] for s in shapes.values() if s['count']]),
         'Geometry scope': d['geometry_summary']['scope'],
         'Other orientations': d['geometry_summary']['other_orientations_status'],
         'Chevron': shapes['Chevron']['status'], 'Chevron count': shapes['Chevron']['count'],
         'Triangle count': shapes['Equilateral triangle']['count'],
         'Hexagon count': shapes['Hexagon']['count'], 'Star count': shapes['2D stellated octahedron']['count'],
         'Unicursal hexagram': 'Present' if ep else shapes['Unicursal hexagram']['status'],
         'Unicursal hexagram count': 1 if ep else shapes['Unicursal hexagram']['count'],
         'Unicursal rule': 'Native 3–6–9 qualification: absent. Reference path: present only when normalized geometry and the M9-reduced traversal exactly match M9 with the fixed north start.' if ep else d['geometry_summary']['epsilon_rule'],
         'Rule version': d['geometry_summary']['rule_version'],
         'Python checks': d['verification']['python'], 'Wolfram checks': d['verification']['wolfram'],
         'Biological focus': 'OXTR gene expression', 'Biological significance': 'Not yet assessed',
         'Statistical significance': 'Not yet assessed', 'Quantum assessment': 'Not yet assessed',
         'Atomic assessment': 'Not yet assessed', 'Genomic assessment': 'Not yet assessed',
         'Annealing': 'Not yet performed', 'Background literature': list(PAPERS.values()),
         'Research register': 'https://www.notion.so/f394fdfe48d14ef191b19ec45b6aa8d9?v=3fc7b9c99c5681a18494000c76dbc0f9',
         'Register version': d['research']['background_literature']['register_version'],
         'Atlas staging': WEB + '?atlas=1&record=' + d['id'],
         'JSON staging': WEB + f'pilot-v1/m{m}.json?v=m1001-v1',
         'Source revision': 'm1-m1001-v1',
         'Reduction to M9': d['reduction_to_m9']['status'] if m % 9 == 0 else 'Outside the M9 family'}
    if b:
        p.update({'Beta interval sums': prop(b['interval_sums'], 'Beta calculation'),
                  'Beta recognised cycle': seq(b['recognised_doubling_cycle'] or []),
                  'Beta start interval': seq(b['start_interval_clock_positions']),
                  'Beta start value': b['start_value']})
    body = ['## Native modulus record', d['conventions']['node_rule'],
            '## Verification', 'Python checks passed. Wolfram core checks ' + d['verification']['wolfram'].lower() + ': ' + d['verification']['wolfram_scope'],
            '## M1–M1001 dataset', f'[Atlas staging]({p["Atlas staging"]}) · [JSON]({p["JSON staging"]}) · [CSV]({WEB}pilot-v1/m1-m1001.csv) · [Full data bundle]({WEB}pilot-v1/m1-m1001-data.zip)',
            '## Research status', 'Biological and statistical significance have not been assessed. The three WSS tiers are reserved for future evidence assessment; no score is assigned. The 14 linked publications, including four HERO studies, provide shared background context.',
            '## Geometry summary']
    body += [('- Unicursal hexagram: reference path present; native 3–6–9 absent; count 1 reference path.' if ep and s['name'] == 'Unicursal hexagram' else f'- {s["name"]}: {s["status"]}; count {s["count"]}.') for s in shapes.values()]
    if m % 9 == 0:
        v = d['m9_sequence_views']; other = d['other_geometry_summary']
        p.update({'M9 cycle repeat count': v['alpha']['repeat_count'],
                  'Beta M9 repeat count': v['beta']['repeat_count'],
                  'M9 reduced Chevron': 'Present', 'M9 reduction validation': 'Passed · original alignment preserved',
                  'M9 sequence audit': v['alpha']['family_audit']['summary'],
                  'M9 family Beta audit': v['beta']['family_audit']['summary'],
                  'M9 comparison insight': 'Alpha, Beta and Gamma recover repeated M9 cycles. Epsilon applies the fixed reference route within each 24-position block; native full-circle qualification is recorded separately.',
                  'Stellated count (Alpha)': other['all_alpha_count'],
                  'Stellated count (Gamma)': other['primary_gamma_count']})
        body += ['## M9 family', p['M9 sequence audit'], p['M9 comparison insight'],
                 f'{v["alpha"]["repeat_count"]} repeated M9 P24 cycles. Original phase and start are preserved.',
                 f'2D stellated octahedrons: {other["all_alpha_count"]} across full Alpha; {other["primary_gamma_count"]} within primary Gamma.',
                 'Exact modular reduction is distinct from the selected native random-shuffle probabilities.']
        for kind in ('alpha', 'beta', 'gamma', 'epsilon'):
            p['M9 reduced ' + kind.title()] = prop(v[kind]['reduced_values'], kind.title() + ' reduced to M9')
    for kind, values in [('alpha', d['alpha']['display_values']), ('beta', b['display_cycle'] if b else []), ('gamma', g['clockwise_values']), ('epsilon', ep['native_values'] if ep else e['traversal_values'] if e else [])]:
        view = next((v for v in d['views'] if v['id'] == kind), None)
        body += ['## ' + kind.title(), '### Native M' + str(m)]
        if view: body.append(f'![{kind.title()} native M{m}]({WEB}{view["png"]})')
        body.append(sequence_content(values) if values else ('Not defined.' if kind == 'beta' else d['epsilon_status']))
        if kind == 'epsilon' and ep: body.append(p['Unicursal rule'])
        if kind == 'beta' and b:
            body += ['### Beta calculation', b['endpoint_rule'], 'Interval sums:', sequence_content(b['interval_sums']),
                     'Original start interval: ' + seq(b['start_interval_clock_positions']) + '.']
        if kind == 'gamma': body += ['### Gamma clock positions', sequence_content(g['clock_positions'])]
        if m % 9 == 0:
            v = d['m9_sequence_views'][kind]
            body += ['### ' + kind.title() + ' reduced to M9', v['operation'],
                     f'![{kind.title()} reduced to M9]({WEB}{v["reduced_png"]})',
                     sequence_content(v['reduced_values'])]
    return {'properties': p, 'content': '\n\n'.join(body)}

if __name__ == '__main__':
    lo, hi = map(int, sys.argv[1:3])
    print(json.dumps([page(m) for m in range(lo, hi + 1)], ensure_ascii=False))
