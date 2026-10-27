from vectramark.config.settings import get_settings


def test_settings_have_expected_defaults():
    settings = get_settings()

    assert settings.app_name == "VectraMark"
    assert settings.app_version == "0.1.0"


def test_data_directories_are_paths():
    settings = get_settings()

    assert settings.data_dir.name == "data"
    assert settings.logs_dir.name == "logs"
