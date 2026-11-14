import pytest

from footnote import data


@pytest.fixture(autouse=True)
def _not_frozen(tmp_path, monkeypatch):
    """Every test starts pre-freeze unless it creates the freeze file itself."""
    monkeypatch.setenv("FOOTNOTE_FROZEN_PATH", str(tmp_path / "frozen.json"))
    return tmp_path / "frozen.json"


@pytest.fixture(scope="session")
def raw():
    if not data.DEFAULT_PATH.exists():
        pytest.skip("Real data file not present at data/raw/")
    return data.load_raw(allow_download=False)
