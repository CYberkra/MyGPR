from __future__ import annotations

from core.online_basemap import OnlineBasemapSettings, PROVIDERS, resolved_provider, validate_settings


def test_esri_world_imagery_is_a_keyless_builtin_provider() -> None:
    provider = PROVIDERS["esri_world_imagery"]
    assert provider.requires_token is False
    assert "World_Imagery/MapServer/tile/{z}/{y}/{x}" in provider.url_template
    assert "Esri" in provider.attribution

    settings = OnlineBasemapSettings(enabled=True, provider_id="esri_world_imagery", max_tiles=16)
    validate_settings(settings)
    assert resolved_provider(settings) is provider
