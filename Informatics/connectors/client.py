"""Read Universal Informatics through MCP, A2A or the REST API (Python 3.10+).

python client.py --endpoint https://informatics-universal-one-staging.universalmind.workers.dev --mode mcp
python client.py --endpoint URL --mode a2a --question 'Explain M9 probability assumptions.'
"""
import argparse
import json
import urllib.request
import urllib.error
import uuid


def request(endpoint, path, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(endpoint.rstrip('/') + path, data=data,
        headers={'Accept': 'application/json, text/event-stream', 'Content-Type': 'application/json', **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def rpc(endpoint, method, params=None, path='/mcp'):
    headers = {'A2A-Version': '1.0'} if path == '/a2a/v1' else {'MCP-Protocol-Version': '2025-11-25'}
    result = request(endpoint, path, {'jsonrpc': '2.0', 'id': str(uuid.uuid4()), 'method': method, 'params': params or {}}, headers)
    if result.get('error'):
        raise RuntimeError(result['error']['message'])
    return result['result']


def initialize(endpoint):
    return rpc(endpoint, 'initialize', {'protocolVersion': '2025-11-25', 'capabilities': {}, 'clientInfo': {'name': 'informatics-python', 'version': '1.0'}})


def call(endpoint, name, arguments):
    result = rpc(endpoint, 'tools/call', {'name': name, 'arguments': arguments})
    if result.get('isError'):
        raise RuntimeError(result['content'][0]['text'])
    return result.get('structuredContent') or json.loads(result['content'][0]['text'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--mode', choices=['mcp', 'a2a', 'api'], default='mcp')
    parser.add_argument('--question', default='Explain the M9 probability model and cite its assumptions.')
    args = parser.parse_args()
    if not args.endpoint.startswith('https://'):
        parser.error('Use the HTTPS endpoint from the connection guide.')
    if args.mode == 'api':
        result = request(args.endpoint, '/api/v1/inventory')
    elif args.mode == 'mcp':
        initialize(args.endpoint)
        result = call(args.endpoint, 'get_atlas_record', {'id': 'M9', 'section': 'probability'})
    else:
        result = rpc(args.endpoint, 'SendMessage', {'message': {'messageId': str(uuid.uuid4()), 'role': 'ROLE_USER', 'parts': [{'text': args.question}]}}, '/a2a/v1')
    print(json.dumps(result, ensure_ascii=False, indent=2))
