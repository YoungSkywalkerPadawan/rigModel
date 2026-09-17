"""Fetch pinned public files and verify their official LFS/blob identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]


def verify(path, item):
    if not path.is_file() or path.stat().st_size != item['bytes']:
        return False
    if 'sha256' in item:
        with path.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
            # AutoDL's 2 GiB CPU quota includes the 1.44 GB checkpoint file cache.
            # Release only this verified file's clean cache before importing torch.
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            return digest == item['sha256']
    data = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == item['git_blob_sha1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='https://huggingface.co')
    args = parser.parse_args()
    items = json.loads((ROOT / 'configs/assets.lock.json').read_text())['files']
    for item in items:
        path = ROOT / item['destination']
        path.parent.mkdir(parents=True, exist_ok=True)
        if verify(path, item):
            print('VERIFIED_EXISTING', item['destination'], flush=True)
            continue
        partial = path.with_suffix(path.suffix + '.partial')
        url = f"{args.endpoint}/{item['repo_id']}/resolve/{item['revision']}/{item['filename']}"
        for attempt in range(4):
            try:
                # Public downloads: no authentication and no inherited Hub token.
                with requests.get(url, stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    with partial.open('wb') as output:
                        total = 0
                        last = time.monotonic()
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            output.write(chunk)
                            total += len(chunk)
                            if time.monotonic() - last > 20:
                                print('DOWNLOADING', item['destination'], total, '/', item['bytes'], flush=True)
                                last = time.monotonic()
                if not verify(partial, item):
                    raise ValueError('Downloaded size/hash mismatch: ' + item['destination'])
                os.replace(partial, path)
                print('VERIFIED', item['destination'], flush=True)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)
    print('ASSETS_READY', flush=True)


if __name__ == '__main__':
    main()
