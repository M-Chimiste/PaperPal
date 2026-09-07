"""Characterize settings: default orchestration config + secret encryption."""


def test_orchestration_defaults(client, empty_db, db, golden, monkeypatch):
    from pathlib import Path
    from theseus_insight.api.routers import settings
    fixture = Path(__file__).parents[1] / "goldens" / "orchestration_defaults.json"
    monkeypatch.setattr(settings, "get_config_path", lambda name: str(fixture))
    # On a DB with no stored orchestration config, the endpoint serves
    # code-level defaults. Freeze them: this shape feeds every pipeline.
    db.execute("DELETE FROM settings WHERE key = 'orchestration'")
    resp = client.get("/api/settings/orchestration")
    assert resp.status_code == 200
    golden("orchestration_defaults", resp.json())


def test_secret_setting_roundtrip_and_format(empty_db):
    from theseus_insight.data_access.settings import SettingsRepository
    SettingsRepository.set_secret_setting("characterization_secret", "hunter2-secret")
    assert SettingsRepository.get_secret_setting("characterization_secret") == "hunter2-secret"
    first = SettingsRepository.get("characterization_secret")
    SettingsRepository.set_secret_setting("characterization_secret", "hunter2-secret")
    assert first.startswith("fernet:v1:")
    assert first != SettingsRepository.get("characterization_secret")


def test_credentials_are_write_only(client, empty_db, monkeypatch):
    from theseus_insight.data_access.settings import SettingsRepository
    monkeypatch.setenv("OPENAI_API_KEY", "old-value")
    response = client.put('/api/settings/credentials', json={'OPENAI_API_KEY': 'new-secret'})
    assert response.status_code == 200
    assert SettingsRepository.get_secret_setting('OPENAI_API_KEY') == 'new-secret'
    response = client.get('/api/settings/credentials')
    assert 'new-secret' not in response.text
    assert response.json()['OPENAI_API_KEY']['configured']
    client.put('/api/settings/credentials', json={'OPENAI_API_KEY': ''})
    assert SettingsRepository.get_secret_setting('OPENAI_API_KEY') == 'new-secret'
    client.delete('/api/settings/credentials/OPENAI_API_KEY')
    assert SettingsRepository.get('OPENAI_API_KEY') is None
