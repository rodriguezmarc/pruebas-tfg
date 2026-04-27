"""
Definition:
Brief map of shared dataset label constants used across the project.
---
Results:
Exports canonical label ids and their human-readable names.
"""

# --- labels: important to unify them with the rest of the datasets ---
RV_LABEL = 1
MYO_LABEL = 2
LV_LABEL = 3
LABEL_TO_NAME = {
    RV_LABEL: "RV",
    MYO_LABEL: "MYO",
    LV_LABEL: "LV",
}

from data.run_pipeline import run_csv_pipeline

__all__ = [
    "RV_LABEL",
    "MYO_LABEL",
    "LV_LABEL",
    "LABEL_TO_NAME",
    "run_csv_pipeline",
]
