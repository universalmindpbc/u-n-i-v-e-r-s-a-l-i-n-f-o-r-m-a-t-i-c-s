"""Stream complete Informatics CSV records using Python's standard library.

Usage: python read_complete.py informatics-complete.csv.gz quantum/data/
"""
import csv
import gzip
import io
import sys


def records(stream):
    csv.field_size_limit(sys.maxsize)
    ident = None
    root = None
    for row in csv.DictReader(stream):
        if row['record_id'] != ident:
            if ident is not None: yield ident, root
            ident, root = row['record_id'], None
        value = row['value'][1:] if row['escaped'] == '1' else row['value']
        typ = row['type']
        def scalar(v, t):
            if t == 'null': return None
            if t == 'boolean': return v == 'true'
            if t == 'integer': return int(v)
            if t == 'number': return float(v)
            return v
        if typ == 'object': value = {}
        elif typ == 'array': value = []
        elif typ.startswith('list:'):
            value = [scalar(v, typ[5:]) for v in next(csv.reader(io.StringIO(value), delimiter=';'))]
        else: value = scalar(value, typ)
        if not row['field']:
            root = value
            continue
        parts = [p.replace('~1', '/').replace('~0', '~') for p in row['field'][1:].split('/')]
        parent = root
        for part in parts[:-1]: parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        if isinstance(parent, list):
            if int(parts[-1]) != len(parent): raise ValueError('Invalid array order')
            parent.append(value)
        else: parent[parts[-1]] = value
    if ident is not None: yield ident, root


if __name__ == '__main__':
    filename = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else ''
    opener = gzip.open if filename.endswith('.gz') else open
    with opener(filename, 'rt', encoding='utf-8-sig', newline='') as stream:
        for ident, value in records(stream):
            if ident.startswith(prefix): print(ident, type(value).__name__)
