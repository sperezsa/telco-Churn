"""Download the training dataset into the configured project data directory."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

from telco_churn.config import DEFAULT_CONFIG_PATH, load_config, project_path


DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def download_dataset(url: str, output_path: Path) -> Path:
    """Stream a CSV to disk and atomically replace any previous copy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with urlopen(url, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent, delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                shutil.copyfileobj(response, temporary_file)
        if temporary_path.stat().st_size == 0:
            raise ValueError(f"Dataset download from {url} was empty.")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.getenv("DATASET_URL", DEFAULT_DATASET_URL),
        help="Direct URL to the CSV (defaults to DATASET_URL or the public sample).",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(DEFAULT_CONFIG_PATH)
    output_path = args.output or project_path(config["data"]["path"])
    downloaded_path = download_dataset(args.url, output_path)
    print(f"Dataset downloaded to {downloaded_path}")


if __name__ == "__main__":
    main()
