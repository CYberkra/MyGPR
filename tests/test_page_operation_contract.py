from core.page_operation_contract import get_page_contract, assert_page_allows


def test_display_page_is_display_only_contract():
    contract = get_page_contract("display")
    assert contract.mutates_data is False
    assert contract.allowed_operation_types == frozenset({"display_only", "compare", "screenshot_export"})


def test_page_contract_rejects_processing_on_display_page():
    try:
        assert_page_allows("display", "processing")
    except ValueError as exc:
        assert "does not allow" in str(exc)
    else:
        raise AssertionError("display page must not allow processing operations")


def test_processing_and_autotune_are_data_mutating_pages():
    assert get_page_contract("processing").mutates_data is True
    assert get_page_contract("autotune").mutates_data is True
    assert_page_allows("processing", "processing")
    assert_page_allows("autotune", "processing_recipe")
