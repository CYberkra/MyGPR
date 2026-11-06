"""Build-owned trust configuration; user preferences cannot replace signing keys."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

RELEASES_URL = 'https://github.com/CYberkra/MyGPR/releases'


def windows_build_version(version: str) -> str:
    match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?', version)
    if not match:
        raise ValueError('安装版需要三段版本号及可选 a/b/rc 序号')
    numbers = [int(match[i]) for i in (1, 2, 3)]
    suffix = 60000 if not match[4] else {'a': 10000, 'b': 20000, 'rc': 30000}[match[4]] + int(match[5])
    if any(number > 65535 for number in numbers) or (match[5] and int(match[5]) >= 10000):
        raise ValueError('版本号超出 Windows 版本字段范围')
    return '.'.join(str(value) for value in (*numbers, suffix))


def https_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password or parts.fragment:
        raise ValueError('更新地址必须为不含身份信息的 HTTPS 地址')
    return value


@dataclass(frozen=True)
class UpdateConfiguration:
    public_key: str
    stable_feed: str
    preview_feed: str = ''

    @classmethod
    def load(cls, path: Path) -> UpdateConfiguration:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or payload.get('schema') != 'mygpr.windows_updates.v1':
            raise ValueError('更新配置版本不支持')
        key = str(payload.get('public_key') or '').strip()
        if not key:
            raise ValueError('此安装包尚未配置更新签名公钥，请从发布页下载')
        try:
            raw = base64.b64decode(key, validate=True)
        except ValueError as exc:
            raise ValueError('更新公钥格式无效') from exc
        if len(raw) != 32:
            raise ValueError('更新公钥必须为 Ed25519 公钥')
        stable = https_url(str(payload.get('stable_feed') or ''))
        preview = str(payload.get('preview_feed') or '')
        return cls(key, stable, https_url(preview) if preview else '')

    def feed(self, channel: str) -> str:
        if channel == 'stable':
            return self.stable_feed
        if channel == 'preview' and self.preview_feed:
            return self.preview_feed
        raise ValueError('预览版更新尚未发布，请使用稳定版渠道')
