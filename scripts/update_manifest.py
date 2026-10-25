"""Recompute SOURCE_MANIFEST.json after editing code, configs or evidence.

The notebook's first cell stops if a listed file differs from its recorded hash, so run this
after every change you commit:  python scripts/update_manifest.py
Hashes are SHA256 after CRLF-to-LF normalization, so Windows and Linux checkouts agree.
The notebook itself is not listed: saving it with outputs would change its own hash.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ['src/*.py', 'scripts/*.py', 'analysis/*.py', 'tests/*.py', 'configs/*.json', 'docs/*.txt',
            'requirements*.txt', 'results/evidence/**/*.json', 'scratch/.gitignore']


def main():
    files = sorted({p.relative_to(ROOT).as_posix() for pattern in PATTERNS for p in ROOT.glob(pattern) if p.is_file()})
    sha = {f: hashlib.sha256((ROOT / f).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for f in files}
    manifest = dict(hash_mode='SHA256 after CRLF-to-LF normalization', sha256=sha)
    (ROOT / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'{len(sha)} files hashed')


if __name__ == '__main__':
    main()
