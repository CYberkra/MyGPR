#!/usr/bin/env python3
"""Ensure every persisted MyGPR schema identifier is owned in the central catalog."""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CATALOG=ROOT/'config/schema_catalog.json'
PATTERN=re.compile(r'["\'](mygpr\.[A-Za-z0-9_.-]+\.v\d+)["\']')
SCOPES=('core','mygpr','scripts')


def discovered():
    found={}
    for scope in SCOPES:
        for path in (ROOT/scope).rglob('*.py'):
            if '__pycache__' in path.parts: continue
            text=path.read_text(encoding='utf-8',errors='ignore')
            for schema in PATTERN.findall(text):
                found.setdefault(schema,[]).append(path.relative_to(ROOT).as_posix())
    return found


def main():
    payload=json.loads(CATALOG.read_text(encoding='utf-8'))
    catalog={row['schema']:row for row in payload['schemas']}
    found=discovered(); errors=[]
    for schema,paths in sorted(found.items()):
        if schema not in catalog: errors.append(f'unowned schema {schema}: {paths[0]}')
    for schema,row in catalog.items():
        if not row.get('owner'): errors.append(f'missing owner: {schema}')
        if row.get('mutable') and not row.get('migration_policy'): errors.append(f'missing migration policy: {schema}')
    if errors:
        print('\n'.join(errors)); return 1
    print(f'schema catalog: PASS ({len(catalog)} owned schemas, {len(found)} referenced)'); return 0

if __name__=='__main__': raise SystemExit(main())
