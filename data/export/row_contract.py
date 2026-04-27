"""
Definition:
Brief map of the internal typed row contract used before MINIM CSV export.
---
Results:
Provides the canonical row model and validation helpers for dataset drivers.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class DataRow:
    """
    ########################################
    Definition:
    Store one canonical dataset row before CSV serialization.
    ---
    Params:
    path: Relative image path written to the MINIM manifest.
    text: Prompt text paired with the image.
    modality: Imaging modality label.
    patient_id: Patient identifier used for splitting.
    dataset: Dataset identifier that produced the row.
    ---
    Results:
    Provides the internal row contract shared across dataset drivers.
    ########################################
    """

    path: str
    text: str
    modality: str
    patient_id: str
    dataset: str

    def to_dict(self) -> dict[str, str]:
        """
        ########################################
        Definition:
        Convert the internal row model into a plain dictionary.
        ---
        Params:
        None.
        ---
        Results:
        Returns a serializable dictionary with canonical row fields.
        ########################################
        """
        return asdict(self)
