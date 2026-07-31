#!/usr/bin/env python3
"""Bootstrap/update schema ownership catalog without deleting manual metadata."""
from __future__ import annotations
import json
from pathlib import Path
from scripts.check_schema_catalog import CATALOG, discovered


def owner(paths):
    first=paths[0]
    if 'report' in first: return 'reporting'
    if 'gis' in first or 'spatial' in first: return 'gis'
    if 'sensor' in first or 'trajectory' in first: return 'sync'
    if 'job' in first: return 'jobs'
    if 'project' in first or 'source_file' in first: return 'project-storage'
    if first.startswith('compatibility/'): return 'compatibility'
    return 'core'


def main():
    old={}
    if CATALOG.exists(): old={row['schema']:row for row in json.loads(CATALOG.read_text(encoding='utf-8')).get('schemas',[])}
    rows=[]
    for schema,paths in sorted(discovered().items()):
        row=old.get(schema,{"schema":schema,"owner":owner(paths),"mutable":False,"migration_policy":"immutable-evidence"})
        row['references']=sorted(paths)[:8]
        rows.append(row)
    CATALOG.write_text(json.dumps({"schema":"mygpr.schema_catalog.v1","schemas":rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(CATALOG)
    return 0
if __name__=='__main__': raise SystemExit(main())
