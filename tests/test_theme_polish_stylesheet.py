from ui.theme import get_app_polish_stylesheet


def test_app_polish_stylesheet_renders_for_light_and_dark():
    for theme in ("light", "dark"):
        stylesheet = get_app_polish_stylesheet(theme)
        assert "QMainWindow" in stylesheet
        assert "AutoTuneTuningPage" in stylesheet
        assert "__" not in stylesheet
        assert "primary_bg" not in stylesheet
