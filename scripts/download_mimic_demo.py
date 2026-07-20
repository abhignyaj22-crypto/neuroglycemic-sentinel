#!/usr/bin/env python3
"""Download only the MIMIC-IV Demo tables needed by the EHR glucose study."""

import argparse
import urllib.request
from pathlib import Path


BASE_URL = "https://physionet.org/files/mimic-iv-demo/2.2/hosp"
FILENAMES = ("patients.csv.gz", "admissions.csv.gz", "d_labitems.csv.gz", "labevents.csv.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/mimiciv-demo/2.2/hosp"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILENAMES:
        destination = args.output_dir / filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"Already present: {destination} ({destination.stat().st_size:,} bytes)")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, destination)
        print(f"Saved {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
