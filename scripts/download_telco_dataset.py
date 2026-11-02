"""Download and validate the public IBM Telco Customer Churn sample CSV.

The public CSV is fetched on demand rather than committed to the repository. It is fetched from
IBM's public GitHub repository on demand and verified against a known SHA-256.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
EXPECTED_ROWS = 7043
EXPECTED_COLUMNS = 21
REQUIRED_COLUMNS = {"customerID", "TotalCharges", "Churn"}
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "telco_customer_churn.csv"


def validate_dataset_bytes(content: bytes) -> dict[str, object]:
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            "SHA-256 checksum mismatch for the IBM Telco sample. "
            f"Expected {EXPECTED_SHA256}, received {actual_hash}. "
            "The upstream file may have changed; do not use the file without reviewing it."
        )
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The downloaded CSV is not valid UTF-8 text.") from exc
    reader = csv.reader(StringIO(text))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError("The downloaded CSV is empty.") from exc
    if len(header) != EXPECTED_COLUMNS:
        raise ValueError(f"Expected {EXPECTED_COLUMNS} columns, found {len(header)}.")
    missing = REQUIRED_COLUMNS - set(header)
    if missing:
        raise ValueError(f"The downloaded CSV is missing required columns: {sorted(missing)}")
    row_count = sum(1 for row in reader)
    if row_count != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} customer rows, found {row_count}.")
    return {
        "source_url": DATA_URL,
        "sha256": actual_hash,
        "rows": row_count,
        "columns": len(header),
        "header": header,
    }


def download_dataset(output_path: str | Path = DEFAULT_OUTPUT, timeout: int = 60) -> dict[str, object]:
    output_path = Path(output_path)
    request = Request(DATA_URL, headers={"User-Agent": "Customer-Risk-Intelligence-Data-Downloader/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            content = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not download the public IBM Telco dataset from {DATA_URL}. "
            "Check your internet connection and try again."
        ) from exc

    metadata = validate_dataset_bytes(content)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    metadata["output_path"] = str(output_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Destination CSV path.")
    parser.add_argument("--timeout", type=int, default=60, help="Download timeout in seconds.")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    try:
        metadata = download_dataset(output, timeout=max(1, args.timeout))
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Dataset download failed: {exc}") from exc
    print("IBM Telco Customer Churn sample downloaded and validated.")
    print(f"Rows: {metadata['rows']:,}; columns: {metadata['columns']}")
    print(f"SHA-256: {metadata['sha256']}")
    print(f"Saved to: {metadata['output_path']}")
    print("Source: IBM public GitHub repository (see data/README.md for attribution).")


if __name__ == "__main__":
    main()
