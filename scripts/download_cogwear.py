"""Download and checksum the real CogWear pilot streams used by main.py."""

import argparse
import hashlib
import os
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE_URL = "https://physionet-open.s3.amazonaws.com/consumer-grade-wearables/1.0.0"
CONDITIONS = ("baseline", "cognitive_load")
FILES = ("muse_eeg.csv", "empatica_bvp.csv", "empatica_eda.csv", "empatica_temp.csv")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_manifest() -> dict[str, str]:
    with urllib.request.urlopen(f"{BASE_URL}/SHA256SUMS.txt", timeout=60) as response:
        text = response.read().decode("utf-8")
    result: dict[str, str] = {}
    for line in text.splitlines():
        checksum, relative_path = line.split(maxsplit=1)
        result[relative_path.strip()] = checksum
    return result


def download_one(relative_path: str, destination_root: Path, expected_hash: str) -> tuple[str, str]:
    destination = destination_root / relative_path.removeprefix("pilot/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == expected_hash:
        return relative_path, "already verified"

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        f"{BASE_URL}/{relative_path}", headers={"User-Agent": "cogwear-aligned-research/0.1"}
    )
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)

    actual_hash = sha256(temporary)
    if actual_hash != expected_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {relative_path}: expected {expected_hash}, got {actual_hash}"
        )
    os.replace(temporary, destination)
    return relative_path, "downloaded + verified"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--participants",
        default="0-10",
        help="Inclusive range such as 0-10. The study config expects all 11 pilot participants.",
    )
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()

    start_text, stop_text = arguments.participants.split("-", maxsplit=1)
    participants = range(int(start_text), int(stop_text) + 1)
    manifest = checksum_manifest()
    destination_root = PROJECT_ROOT / "data" / "raw" / "cogwear" / "pilot"
    requested_paths = [
        f"pilot/{participant}/{condition}/{filename}"
        for participant in participants
        for condition in CONDITIONS
        for filename in FILES
    ]
    unavailable = [path for path in requested_paths if path not in manifest]
    paths = [path for path in requested_paths if path in manifest]

    print(f"Downloading {len(paths)} real CogWear files into {destination_root}")
    print("Source license and study documentation: https://physionet.org/content/consumer-grade-wearables/1.0.0/")
    if unavailable:
        print("Published files absent from the requested grid (recorded, not fabricated):")
        for path in unavailable:
            print(f"  - {path}")
    with ThreadPoolExecutor(max_workers=max(1, arguments.workers)) as pool:
        futures = {
            pool.submit(download_one, path, destination_root, manifest[path]): path for path in paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            relative_path, status = future.result()
            print(f"[{completed:02d}/{len(paths):02d}] {status}: {relative_path}")


if __name__ == "__main__":
    main()
