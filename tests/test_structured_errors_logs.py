from core.app_errors import InputDataError, error_info_from_exception
from core.log_events import LogEvent, LogEventBuffer, classify_log_event


def test_structured_error_to_dict_contains_stable_code():
    err = InputDataError("CSV 加载失败", technical_detail="empty file", context={"path": "x.csv"})
    payload = err.to_dict()
    assert payload["schema"] == "mygpr.error_info.v1"
    assert payload["error_code"] == "MYGPR_INPUT_DATA_ERROR"
    assert payload["category"] == "input_data"
    assert payload["context"]["path"] == "x.csv"


def test_error_info_from_plain_exception_is_serializable():
    info = error_info_from_exception(ValueError("bad"), category="processing")
    payload = info.to_dict()
    assert payload["error_type"] == "ValueError"
    assert payload["error_code"] == "MYGPR_PROCESSING_ERROR"
    assert payload["user_message"] == "bad"


def test_log_event_classification_and_buffer_pruning():
    assert classify_log_event("处理错误: bad") == "ERR"
    assert classify_log_event("报告已导出") == "EXPORT"
    buffer = LogEventBuffer(max_events=2)
    buffer.append(LogEvent.create("a"))
    buffer.append(LogEvent.create("b"))
    buffer.append(LogEvent.create("c"))
    events = buffer.to_list()
    assert len(events) == 2
    assert events[0]["message"] == "b"
    assert events[1]["schema"] == "mygpr.log_event.v1"
