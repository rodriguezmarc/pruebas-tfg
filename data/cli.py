from __future__ import annotations

from data.config import _build_output_csv_path, OUTPUT_PATHS
from data.datasets.acdc.cache import export_preprocessed_dataset


def main() -> None:
    rows = export_preprocessed_dataset(split="train")
    output_csv_path = _build_output_csv_path(OUTPUT_PATHS["csv"])
    print(f"Exported {len(rows)} ACDC rows.")
    print(f"CSV created at {output_csv_path}.")


if __name__ == "__main__":
    main()
