import pytest

from footnote import data


def test_item_lists():
    assert len(data.DOLLAR_ITEMS) == 26
    assert len(set(data.DOLLAR_ITEMS)) == 26
    assert len(data.RAW_ITEMS) == 28
    assert len(data.READY_RATIOS) == 14
    assert not set(data.RAW_ITEMS) & set(data.READY_RATIOS)


def test_wrong_size_refused(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("fyear,gvkey\n1990,1\n")
    with pytest.raises(data.ChecksumError):
        data.verify_file(p)


def test_right_size_wrong_hash_refused(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_bytes(b"x" * data.EXPECTED_BYTES)
    with pytest.raises(data.ChecksumError):
        data.verify_file(p)


def test_missing_file_no_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.ensure_file(tmp_path / "nope.csv", allow_download=False)


def test_real_file_checksum():
    if not data.DEFAULT_PATH.exists():
        pytest.skip("Real data file not present")
    data.verify_file(data.DEFAULT_PATH)
    assert data.sha256_file(data.DEFAULT_PATH) == data.EXPECTED_SHA256


def test_real_file_schema(raw):
    assert len(raw) == 146_045
    assert raw.shape[1] == 46 + 2  # plus row_id and tie_key
    assert raw["fyear"].min() == 1990 and raw["fyear"].max() == 2014
    assert raw["gvkey"].nunique() == 18_444
    assert not raw.duplicated(["gvkey", "fyear"]).any()
    # Checked on the unmasked file before 2003+ values are blanked out.
    miss = raw.attrs["schema_counts"]["missing_by_column"]
    assert all(miss[c] == 0 for c in data.RAW_ITEMS)
    assert raw.loc[raw.fyear <= 2002, data.RAW_ITEMS].notna().all().all()
    # Empty p_aaer means no case: present exactly on positive rows.
    assert raw.loc[raw.misstate == 1, "p_aaer"].notna().all()
    assert raw.loc[raw.misstate == 0, "p_aaer"].isna().all()


def test_tie_key_is_fixed_permutation(raw):
    import numpy as np

    assert sorted(raw["tie_key"]) == list(range(len(raw)))
    expected = np.random.default_rng(0).permutation(len(raw))
    assert (raw["tie_key"].to_numpy() == expected).all()
    # Not file order.
    assert (raw["tie_key"].to_numpy() != raw["row_id"].to_numpy()).mean() > 0.99
