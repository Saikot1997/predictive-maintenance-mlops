"""
Kaggle থেকে dataset download করার script।
Terminal থেকে: python -m src.data.download_data
"""
import os
import shutil
from pathlib import Path
import yaml

def download_dataset(config_path: str = "configs/data_config.yaml") -> None:
    """Dataset download এবং সঠিক জায়গায় রাখো।"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_path = Path(config["data"]["raw_path"]).parent
    raw_path.mkdir(parents=True, exist_ok=True)

    dataset_slug = "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
    print(f"📥 Downloading: {dataset_slug}")

    os.system(
        f"kaggle datasets download -d {dataset_slug} "
        f"-p {raw_path} --unzip"
    )

    # Rename করো যদি দরকার হয়
    downloaded = raw_path / "ai4i2020.csv"
    target = raw_path / "predictive_maintenance.csv"
    if downloaded.exists() and not target.exists():
        shutil.move(str(downloaded), str(target))

    print(f"✅ Dataset saved: {target}")


if __name__ == "__main__":
    download_dataset()