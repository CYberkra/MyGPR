"""UI-independent field records; traces are zero-based, times are UTC ISO-8601."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

MARKER_SCHEMA = 'mygpr.marker_events.v1'
PURPOSES = ('停车等待', '外部干扰', '目标范围', '现场操作', '其他')


def utc_time(value: str) -> str:
    if not value:
        return ''
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('采集时间必须包含时区')
    return stamp.astimezone(timezone.utc).isoformat()


def _text(value: str, limit: int, title: str) -> None:
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f'{title}格式或长度无效')


@dataclass(frozen=True)
class MarkerAnchor:
    source_id: str
    trace: int

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id or len(self.source_id) > 512 or type(self.trace) is not int or self.trace < 0:
            raise ValueError('标记来源或道号无效')


@dataclass(frozen=True)
class MarkerRecord:
    uid: str = field(default_factory=lambda: str(uuid4()))
    name: str = ''
    purpose: str = '其他'
    note: str = ''
    evidence: str = '现场记录'
    anchor: MarkerAnchor | None = None
    revision: int = 1
    origin: str = 'project'
    original: dict[str, Any] = field(default_factory=dict)
    excluded: bool = False
    links: tuple[str, ...] = ()
    created_at: str = ''

    def __post_init__(self):
        _record_fields(self)
        if self.origin not in ('project', 'acquisition'):
            raise ValueError('标记来源类型无效')
        object.__setattr__(self, 'original', deepcopy(self.original))
        object.__setattr__(self, 'links', tuple(self.links))
        object.__setattr__(self, 'created_at', utc_time(self.created_at))


@dataclass(frozen=True)
class EventInterval:
    uid: str = field(default_factory=lambda: str(uuid4()))
    name: str = ''
    purpose: str = '其他'
    note: str = ''
    evidence: str = '现场记录'
    start: MarkerAnchor | None = None
    end: MarkerAnchor | None = None
    include_start: bool = True
    include_end: bool = True
    time_start: str = ''
    time_end: str = ''
    time_source_id: str = ''
    start_marker_id: str = ''
    end_marker_id: str = ''
    start_marker_revision: int = 0
    end_marker_revision: int = 0
    revision: int = 1
    review_state: str = 'unreviewed'
    excluded: bool = False
    links: tuple[str, ...] = ()

    def __post_init__(self):
        _record_fields(self)
        if self.review_state not in ('unreviewed', 'reviewed', 'needs_review'):
            raise ValueError('区间复核状态无效')
        if type(self.include_start) is not bool or type(self.include_end) is not bool:
            raise ValueError('边界包含状态必须为布尔值')
        if (self.start or self.end) and (self.time_start or self.time_end):
            raise ValueError('道号边界和采集时间边界不能同时使用')
        for side in ('start', 'end'):
            uid = getattr(self, side + '_marker_id')
            version = getattr(self, side + '_marker_revision')
            _text(uid, 256, '边界身份')
            if type(version) is not int or version < 0 or bool(uid) != (version > 0):
                raise ValueError('边界标记身份与版本不匹配')
        _text(self.time_source_id, 512, '时间来源')
        object.__setattr__(self, 'time_start', utc_time(self.time_start))
        object.__setattr__(self, 'time_end', utc_time(self.time_end))
        object.__setattr__(self, 'links', tuple(self.links))
        if self.start and self.end:
            if self.start.source_id != self.end.source_id or self.start.trace > self.end.trace:
                raise ValueError('区间边界必须来自同一数据版本且按道序排列')
        if self.time_start and self.time_end and datetime.fromisoformat(self.time_start) > datetime.fromisoformat(self.time_end):
            raise ValueError('区间开始时间晚于结束时间')

    @property
    def closed(self) -> bool:
        return bool((self.start and self.end) or (self.time_start and self.time_end))


def _record_fields(record) -> None:
    if not isinstance(record.uid, str) or not record.uid or len(record.uid) > 256:
        raise ValueError('标记身份无效')
    if type(record.revision) is not int or record.revision < 1:
        raise ValueError('标记版本无效')
    if type(record.excluded) is not bool or not isinstance(record.links, (tuple, list)):
        raise ValueError('标记状态或关联格式无效')
    for link in record.links:
        _text(link, 256, '关联身份')
    for name, limit in (('name', 512), ('purpose', 128), ('note', 16384), ('evidence', 512)):
        _text(getattr(record, name), limit, name)


def record_from_dict(payload: dict[str, Any], *, interval: bool = False):
    values = deepcopy(payload)
    for name in (('start', 'end') if interval else ('anchor',)):
        value = values.get(name)
        if value is not None:
            values[name] = MarkerAnchor(**value)
    cls = EventInterval if interval else MarkerRecord
    return cls(**values)


@dataclass(frozen=True)
class MarkerDocument:
    project_id: str
    line_id: str
    source_id: str
    revision: int = 0
    markers: tuple[MarkerRecord, ...] = ()
    intervals: tuple[EventInterval, ...] = ()
    timezone: str = 'UTC'

    def __post_init__(self):
        if not self.project_id or not self.line_id or not self.source_id:
            raise ValueError('标记文档缺少项目、测线或来源身份')
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('标记文档版本无效')
        ZoneInfo(self.timezone)
        ids = [item.uid for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError('标记身份重复')

    @property
    def records(self) -> tuple[MarkerRecord | EventInterval, ...]:
        return (*self.markers, *self.intervals)

    def to_dict(self) -> dict[str, Any]:
        return {'schema': MARKER_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MarkerDocument:
        values = deepcopy(payload)
        if values.pop('schema', None) != MARKER_SCHEMA:
            raise ValueError('不支持的标记文档版本')
        values['markers'] = tuple(record_from_dict(item) for item in values.get('markers', ()))
        values['intervals'] = tuple(record_from_dict(item, interval=True) for item in values.get('intervals', ()))
        return cls(**values)
