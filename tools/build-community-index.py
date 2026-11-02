"""Publish community relevance tags from the separately retained editorial audit."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'research/informatics/communities-full-20261030/classification-audit.json'
RELATED = ROOT / 'research/informatics/communities-related-20261030/related-assessments.json'
TARGET = ROOT / 'Informatics/library-data/community-index.json'

def render():
    audit = json.loads(AUDIT.read_text())
    related = json.loads(RELATED.read_text())
    records = {row['recordId']: {'recordId': row['recordId'], 'matches': list(row['matches'])}
               for row in audit['records']}
    for link in related['links']:
        row = records[link['recordId']]
        assert link['categoryId'] in {'defence', 'indigenous', 'refugees', 'housing'}
        assert link['relation'] == 'related'
        assert not any(m['categoryId'] == link['categoryId'] for m in row['matches']), 'Duplicate community relation'
        row['matches'].append({'categoryId': link['categoryId'], 'relation': 'related'})
    public = {
        'schemaVersion': 'communities-contexts/v2',
        'title': 'Communities and Contexts',
        'updatedAt': related['assessedAt'],
        'recordCount': audit['recordCount'],
        'taxonomy': audit['taxonomy'],
        'records': list(records.values()),
    }
    approved=json.loads((ROOT/'Informatics/library-data/domestic-violence-expansion.json').read_text())
    merged={r['recordId']:r for r in public['records']}
    youth=json.loads((ROOT/'Informatics/library-data/youth-expansion.json').read_text())
    for row in [*approved['communities'],*youth['communities']]:
        old=merged.get(row['recordId'],{'matches':[]})
        replaced={m['categoryId'] for m in row['matches']}
        merged[row['recordId']]={**row,'matches':[m for m in old['matches'] if m['categoryId'] not in replaced]+row['matches']}
    public.update(records=list(merged.values()),recordCount=len(merged),updatedAt=youth['approvedAt'])
    return json.dumps(public, indent=2, ensure_ascii=False) + '\n'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    expected = render()
    if args.check:
        assert TARGET.read_text() == expected, 'Community index differs from the editorial audit.'
    else:
        TARGET.write_text(expected)
    print(json.dumps({'passed': True, 'records': json.loads(expected)['recordCount']}))
