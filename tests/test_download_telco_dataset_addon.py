import hashlib

import pytest

import scripts.download_telco_dataset as downloader


def test_validate_dataset_bytes_accepts_expected_schema(monkeypatch):
    content = b"customerID,TotalCharges,Churn\nA1,12.5,No\nA2,44.0,Yes\n"
    monkeypatch.setattr(downloader, "EXPECTED_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(downloader, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(downloader, "EXPECTED_COLUMNS", 3)
    metadata = downloader.validate_dataset_bytes(content)
    assert metadata["rows"] == 2
    assert metadata["columns"] == 3
    assert metadata["header"] == ["customerID", "TotalCharges", "Churn"]


def test_validate_dataset_bytes_rejects_checksum_mismatch():
    with pytest.raises(ValueError, match="checksum mismatch"):
        downloader.validate_dataset_bytes(b"customerID,TotalCharges,Churn\nA1,12.5,No\n")
