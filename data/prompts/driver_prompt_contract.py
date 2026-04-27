"""
Definition:
Dataset-driver contracts for prompt adaptation.
---
Results:
Provides structural typing for dataset prompt adapters.
"""

from __future__ import annotations

from typing import Protocol

from data.prompts.prompt_contract import PromptCapabilities, PromptPayload


class PromptMetadataAdapter(Protocol):
    """
    ########################################
    Definition:
    Describe the structural contract protocol that dataset drivers must satisfy to feed prompts.
    ---
    Results:
    Requires explicit capability declaration plus canonical payload adaptation.
    ########################################
    """

    def prompt_capabilities(self) -> PromptCapabilities:
        """
        ########################################
        Definition:
        Declare which prompt fields the dataset driver can provide.
        ---
        Results:
        Returns the capability set supported by the driver.
        ########################################
        """

    def to_prompt_payload(self, metadata: dict, ef: float, modality: str) -> PromptPayload:
        """
        ########################################
        Definition:
        Adapt dataset metadata into the canonical prompt payload.
        ---
        Results:
        Returns the canonical payload consumed by the prompt generator.
        ########################################
        """
