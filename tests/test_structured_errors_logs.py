from mygpr.domain.common.errors import InputDataError, error_info_from_exception


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
