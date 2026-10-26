#!/usr/bin/env python3
"""Restore selected generated assets from the pinned upstream source revision."""
import argparse, hashlib, json, os
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--list',action='store_true',help='List asset counts and sizes; no downloads.')
    mode.add_argument('--all',action='store_true',help='Explicitly restore every recorded generated asset.')
    mode.add_argument('--prefix',help='Restore only paths beginning with this repository-relative prefix.')
    parser.add_argument('--limit',type=int,help='Limit the number of selected files, for a bounded recovery check.')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'docs/GENERATED_ASSETS.json').read_text())
    files=manifest['files']
    if not args.all and not args.prefix:
        print(f"{len(files)} generated assets; {sum(x['bytes'] for x in files):,} bytes at {manifest['source_commit']}.")
        print('Use --prefix for a selected subset, or --all for the entire data collection.')
        return
    if args.prefix:files=[x for x in files if x['path'].startswith(args.prefix)]
    if not files:parser.error('No assets match the selected prefix.')
    if args.limit is not None:
        if args.limit<1:parser.error('--limit must be positive.')
        files=files[:args.limit]
    for item in files:
        path=item['path'];destination=(ROOT/path).resolve()
        if not destination.is_relative_to(ROOT):raise ValueError('Unsafe asset path')
        header=b'blob '+str(item['bytes']).encode()+b'\0'
        if destination.exists():
            raw=destination.read_bytes()
            if len(raw)==item['bytes'] and hashlib.sha1(header+raw).hexdigest()==item['git_blob_sha1']:
                print('Verified existing:',path);continue
            raise RuntimeError('Existing file differs; refusing to overwrite: '+path)
        url='https://raw.githubusercontent.com/'+manifest['source_repository']+'/'+manifest['source_commit']+'/'+quote(path,safe='/')
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_name(destination.name+'.asset-download')
        digest=hashlib.sha1(header);size=0
        try:
            with urlopen(url,timeout=90) as response,temporary.open('xb') as output:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    output.write(chunk);digest.update(chunk);size+=len(chunk)
            if size!=item['bytes'] or digest.hexdigest()!=item['git_blob_sha1']:
                raise RuntimeError('Downloaded asset failed integrity verification: '+path)
            os.replace(temporary,destination)
            print('Restored and verified:',path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

if __name__=='__main__':main()
