"""Project marker CLI; uses the same edit, preview and execution service as Qt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.markers.models import record_from_dict
from mygpr.interfaces.backend import MyGPRBackend


def main(argv=None):
    parser = argparse.ArgumentParser(description='MyGPR 项目标记与事件区间')
    parser.add_argument('project', type=Path)
    parser.add_argument('--line', required=True)
    parser.add_argument('--artifact', default='')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    edit = commands.add_parser('save')
    edit.add_argument('record', type=Path, help='单条点标记或区间 JSON；编辑须包含原 uid 和 revision')
    edit.add_argument('--interval', action='store_true')
    edit.add_argument('--revision', type=int, required=True, help='预期文档修订号')
    export = commands.add_parser('export')
    export.add_argument('destination', type=Path)
    export.add_argument('--format', choices=('csv', 'json', 'geojson'), required=True)
    preview = commands.add_parser('preview')
    preview.add_argument('--id', action='append', required=True)
    preview.add_argument('--mode', choices=('keep', 'exclude'), required=True)
    execute = commands.add_parser('execute')
    execute.add_argument('preview', type=Path, help='已核对的 preview JSON；变化时拒绝执行')
    args = parser.parse_args(argv)
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.open_project(args.project)
        view = backend.markers.view(project.project_id, args.line, args.artifact)
        if args.command == 'list':
            result = view.document.to_dict()
        elif args.command == 'save':
            if view.document.revision != args.revision:
                raise ValueError('标记文档已变化，请重新读取')
            record = record_from_dict(json.loads(args.record.read_text(encoding='utf-8-sig')), interval=args.interval)
            result = backend.markers.edit(view, record).to_dict()
        elif args.command == 'export':
            result = backend.projects.export_markers((view,), args.destination, args.format)
        elif args.command == 'preview':
            result = backend.markers.preview(view, args.id, args.mode)
        else:
            supplied = json.loads(args.preview.read_text(encoding='utf-8-sig'))
            if (supplied.get('project_id'), supplied.get('line_id'), supplied.get('input_artifact_id')) != (
                    project.project_id, args.line, args.artifact):
                raise ValueError('预览不属于指定项目、测线和输入成果')
            current = backend.markers.preview(view, [item['uid'] for item in supplied['intervals']], supplied['mode'])
            # JSON arrays become lists on disk; compare canonical serialized values.
            if json.dumps(current, sort_keys=True) != json.dumps(supplied, sort_keys=True):
                raise ValueError('记录或输入已变化，请重新预览')
            artifact = backend.markers.execute(current, context=ExecutionContext.null())
            result = {'artifact_id': artifact.artifact_id, 'line_id': args.line}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    finally:
        backend.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
