from __future__ import annotations

from data.config import DATASET_PATHS, OUTPUT_PATHS
from data.run_pipeline import run_csv_pipeline


def main() -> None:
    rows = run_csv_pipeline(
        data_path=DATASET_PATHS["acdc"],
        images_root=OUTPUT_PATHS["images"],
        csv_root=OUTPUT_PATHS["csv"],
        internal_root=OUTPUT_PATHS["internal"],
        dataset="acdc",
        modality="Cardiac MRI",
    )
    print(f"Exported {len(rows)} ACDC rows.")


if __name__ == "__main__":
    main()
