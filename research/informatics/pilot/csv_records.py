"""Lossless typed CSV records. No JSON parser is needed to read these exports."""
import csv
import io
from pathlib import Path

BASE_URL = 'https://universalmind.tv/informatics/'

def kind(value):
    if value is None: return 'null'
    return {bool: 'boolean', int: 'integer', float: 'number', str: 'text', dict: 'object', list: 'array'}[type(value)]

def scalar(value):
    if value is None: return ''
    if isinstance(value, bool): return str(value).lower()
    return str(value)

def parse(value, type_name):
    if type_name == 'null': return None
    if type_name == 'boolean': return value == 'true'
    if type_name == 'integer': return int(value)
    if type_name == 'number': return float(value)
    return value

def pointer(key): return str(key).replace('~', '~0').replace('/', '~1')

def rows(value, field=''):
    type_name = kind(value)
    if type_name == 'array' and value and len({kind(v) for v in value}) == 1 and kind(value[0]) not in ('object', 'array'):
        stream = io.StringIO(newline='')
        csv.writer(stream, delimiter=';', lineterminator='\n').writerow([scalar(v) for v in value])
        yield field, 'list:' + kind(value[0]), stream.getvalue()[:-1], ''
    elif type_name in ('object', 'array'):
        yield field, type_name, str(len(value)), ''
        for key, child in (value.items() if isinstance(value, dict) else enumerate(value)):
            yield from rows(child, field + '/' + pointer(key))
    else:
        url = BASE_URL + value if isinstance(value, str) and value.startswith('pilot-v1/') else ''
        yield field, type_name, scalar(value), url

def write_record(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['field', 'type', 'value', 'resolved_url'])
        writer.writerows(rows(value))

def read_record(path):
    """Reconstruct all values using only Python's standard CSV library."""
    root = None
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            type_name, value = row['type'], row['value']
            if type_name == 'object': value = {}
            elif type_name == 'array': value = []
            elif type_name.startswith('list:'):
                value = [parse(v, type_name[5:]) for v in next(csv.reader(io.StringIO(value), delimiter=';'))]
            else: value = parse(value, type_name)
            if row['field'] == '': root = value; continue
            parts = [p.replace('~1', '/').replace('~0', '~') for p in row['field'][1:].split('/')]
            parent = root
            for part in parts[:-1]: parent = parent[int(part)] if isinstance(parent, list) else parent[part]
            if isinstance(parent, list):
                assert int(parts[-1]) == len(parent)
                parent.append(value)
            else: parent[parts[-1]] = value
    return root
