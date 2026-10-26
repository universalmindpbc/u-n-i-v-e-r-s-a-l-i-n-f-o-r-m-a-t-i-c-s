"""Publish a complete CSV-only alternative to the Atlas JSON data bundle."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile
from csv_records import BASE_URL, read_record, rows, write_record

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'Informatics/pilot-v1'


def export():
    summary = DATA / 'm1-m1001.csv'
    with summary.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); fields = list(reader.fieldnames); records = list(reader)
    added = ['record_csv_file', 'epsilon_status', 'epsilon_reference_sequence']
    for sequence in ('alpha', 'beta', 'gamma', 'epsilon'):
        for mode in ('native', 'reduced_m9'):
            for fmt in ('png', 'svg'): added.append(f'{sequence}_{mode}_{fmt}_url')
    added += ['combined_png_url', 'combined_svg_url']
    for field in added:
        if field not in fields: fields.append(field)
    image_rows = []; verified = 0
    for row in records:
        m = int(row['modulus']); d = json.loads((DATA / f'm{m}.json').read_text())
        for field in added: row[field] = ''
        row['json'] = BASE_URL + f'pilot-v1/m{m}.json'
        row['record_csv_file'] = f'csv-records/m{m}.csv'
        row['epsilon_status'] = d['epsilon_status']
        if d.get('epsilon_path'):
            row['epsilon_reference_sequence'] = ' → '.join(map(str, d['epsilon_path']['native_values']))
        for view in d['views']:
            name = view['id']
            for fmt in ('png', 'svg'):
                field = f'{name}_native_{fmt}_url' if name in ('alpha', 'beta', 'gamma', 'epsilon') else f'combined_{fmt}_url'
                row[field] = BASE_URL + view[fmt]
        for name, view in d.get('m9_sequence_views', {}).items():
            for fmt in ('png', 'svg'):
                row[f'{name}_reduced_m9_{fmt}_url'] = BASE_URL + view[f'reduced_{fmt}']
        detail = DATA / row['record_csv_file']
        write_record(detail, d)
        assert read_record(detail) == d, m
        verified += 1
        seen = set()
        for field, type_name, value, url in rows(d):
            if url.endswith(('.png', '.svg')) and url not in seen:
                seen.add(url); file = ROOT / 'Informatics' / value
                image_rows.append({'record_id':d['id'], 'modulus':m, 'field':field, 'format':file.suffix[1:], 'url':url, 'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
    with summary.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    with (DATA / 'm1-m1001-images.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0])); writer.writeheader(); writer.writerows(image_rows)
    shared = ['dataset', 'index', 'm1-m1001', 'research-register', 'm9-sequence-family-audit', 'beta-m9-family-audit', 'm1001-core-signatures', 'm1001-verification', 'm9-probability', 'm18-probability']
    for name in shared:
        value = json.loads((DATA / f'{name}.json').read_text()); file = DATA / 'csv-shared' / f'{name}.csv'
        write_record(file, value); assert read_record(file) == value, name
    publications = json.loads((DATA / 'research-register.json').read_text())['publications']
    study_rows = []
    for study in publications:
        flat = {}
        for field, type_name, value, url in rows(study):
            if type_name not in ('array','object'): flat[field.lstrip('/').replace('/', '_')] = value
        study_rows.append(flat)
    study_fields = list(dict.fromkeys(key for row in study_rows for key in row))
    with (DATA / 'research-publications.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=study_fields); writer.writeheader(); writer.writerows(study_rows)
    readme = DATA / 'M1-M1001-CSV-README.txt'
    readme.write_text('''Universal Informatics — complete CSV data, M1–M1001

No JSON file is needed to use this archive.

m1-m1001.csv: one row per modulus, including the native/reduced sequences, geometry counts, verification status and direct production PNG/SVG URLs. The record_csv_file column locates the complete record inside this archive. A blank Epsilon URL means no qualifying or reference native Epsilon image; it is not an omitted upload. The epsilon_status and detailed record explain the rule. Reduced M9 images exist for all four sequences in each of the 111 M9-family records.

csv-records/m1.csv through m1001.csv: every field and value in the corresponding Atlas record, including full sequences, formation catalogues, calculation intervals, Epsilon assessments, native/M9 comparisons, verification, probability where present, research assessments and publication metadata. These are complete records, not just the overview table. Redundant formation interiors retain their documented endpoint-based reconstruction rule and full Alpha values.

m1-m1001-images.csv: all 8,938 distinct diagram URLs, with record identifiers, source fields, formats and file hashes. PNG and SVG files remain online; image pixels are not embedded in this archive.

research-publications.csv: all 14 studies, including the four HERO references, citations and appraisal fields. csv-shared/ preserves the full shared registers, audits, manifests and existing probability evidence as CSV.

Typed record format: field is a slash-separated path; type distinguishes text, integer, number, boolean, null, object and array. Null is distinct from empty text. Primitive lists use semicolon-separated CSV values within a cell (list:integer, list:text, etc.). Nested arrays and objects use indexed child paths. Empty containers are retained. ~1 represents a slash in a field name and ~0 represents a tilde. resolved_url provides a complete production URL for referenced Atlas resources. Original field values are preserved.

source/csv_records.py reads any typed record using only Python's standard CSV library. Example: from csv_records import read_record; record = read_record('csv-records/m999.csv'). The remaining Python sources document the Atlas calculations and drawing methods; rendering uses Pillow and an available font. No JSON parser is needed to load the CSV record. Website-specific build scripts retain their repository layout requirements.

The JSON link in the overview is optional. All computational fields are also in these CSV records. WSS remains awaiting assessment; arithmetic/geometry evidence does not assign biological significance.
''', encoding='utf-8')
    with zipfile.ZipFile(DATA / 'm1-m1001-csv.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for file in [summary, DATA/'m1-m1001-images.csv', DATA/'research-publications.csv', readme]: z.write(file, file.name)
        for folder in ('csv-records', 'csv-shared'):
            for file in sorted((DATA/folder).glob('*.csv')): z.write(file, str(file.relative_to(DATA)))
        for file in sorted(Path(__file__).parent.glob('*.py')): z.write(file, 'source/'+file.name)
        z.write(ROOT / 'Informatics/pilot-v1/explorer.js', 'source/atlas-geometry.js')
    result = {'passed': True, 'records_roundtripped_without_json_parser':verified, 'summary_columns':len(fields), 'image_urls':len(image_rows), 'publications':len(publications), 'shared_tables':len(shared), 'csv_archive_bytes':(DATA/'m1-m1001-csv.zip').stat().st_size}
    print(json.dumps(result, indent=2))
    if (DATA / 'oxtr-tfbs-v1/catalogue.json').exists():
        from package_oxtr_catalogue import publish
        publish()
    return result

if __name__ == '__main__': export()
