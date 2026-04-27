"""
Definition:
Contract-based prompt helpers for turning canonical cardiac metadata into MINIM text.
---
Results:
Provides EF normalization, payload validation, and segment-based prompt generation.
"""

from __future__ import annotations

from data.preprocess.dataset_utilities import (
    compute_age_group,
    compute_bmi_group,
    compute_disease_label,
    compute_sex_label,
)
from data.prompts.prompt_contract import (
    PromptCapabilities,
    PromptField,
    PromptPayload,
    PromptSegment,
    PromptSpec,
)
from data.preprocess.medical_utilities import (
    to_ef_percentage,
    classify_ef
)

"""
########################################
Definition:
Defines the prompt general recipe, contempling all those fields available to a cardiac prompt.
---
Results:
Returns a ready-to-use contract
########################################
"""
CARDIAC_PROMPT_SPEC = PromptSpec(
    hard_requirements=frozenset({PromptField.MODALITY, PromptField.VIEW, PromptField.BMI, PromptField.EF}),
    segments=(
        PromptSegment(
            required_fields=frozenset({PromptField.MODALITY}),
            render=lambda payload: payload.modality,
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.VIEW}),
            render=lambda payload: payload.view,
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.FRAME}),
            render=lambda payload: str(payload.frame),
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.SEX}),
            render=lambda payload: str(payload.sex),
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.AGE}),
            render=lambda payload: str(payload.age_group),
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.BMI}),
            render=lambda payload: str(payload.bmi_group),
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.EF}),
            render=lambda payload: str(payload.ef_group),
        ),
        PromptSegment(
            required_fields=frozenset({PromptField.GROUP}),
            render=lambda payload: str(payload.pathology_label),
        ),
    ),
)


def validate_prompt_contract(
    payload: PromptPayload,
    capabilities: PromptCapabilities,
    spec: PromptSpec = CARDIAC_PROMPT_SPEC,  # uses the cardiac spec (only modality is cardial)
) -> None:
    """
    ########################################
    Definition:
    Validate hard requirements and declared capabilities before rendering a prompt.
    ---
    Params:
    payload: Canonical prompt payload.
    capabilities: Declared dataset prompt capabilities.
    spec: Prompt specification to validate against.
    ---
    Results:
    Raises ValueError when a required or declared field is missing.
    ########################################
    """
    for field in spec.hard_requirements:  # required fields must be included
        if not payload.has(field):
            raise ValueError(f"Prompt contract violation: missing required field '{field.value}'.")

    for field in capabilities:            # all declared fields must have a value
        if not payload.has(field):
            raise ValueError(
                f"Prompt contract violation: capability '{field.value}' was declared without a value."
            )


def generate_prompt(
    payload: PromptPayload,
    capabilities: PromptCapabilities,
    spec: PromptSpec = CARDIAC_PROMPT_SPEC,  # same as before, only modality is cardiac 
) -> str:
    """
    ########################################
    Definition:
    Generate one deterministic prompt from a canonical payload and declared capabilities.
    ---
    Params:
    payload: Canonical prompt payload.
    capabilities: Declared dataset prompt capabilities.
    spec: Prompt specification that controls order and formatting.
    ---
    Results:
    Returns the final prompt string used in CSV export.
    ########################################
    """
    validate_prompt_contract(payload, capabilities, spec=spec)       # prior check to make sure nothing breaks

    rendered_segments: list[str] = []
    for segment in spec.segments:
        if payload.has(segment.required_fields):                     # firstly checks if value exists
            rendered_segments.append(segment.render(payload))        # then does payload processing

    if not rendered_segments:
        raise ValueError("Prompt generation failed: no prompt segments were rendered.")

    return f"{spec.separator.join(rendered_segments)}{spec.suffix}"  # return fully generated prompt


def build_cardiac_prompt_payload(
    metadata: dict,
    ef: float,
    modality: str = "Cardiac MRI",
    view: str = "short-axis view",
    frame: str | None = None,
) -> PromptPayload:
    """
    ########################################
    Definition:
    Build the canonical cardiac prompt payload from raw metadata and EF.
    ---
    Params:
    metadata: Patient or study metadata with demographic and disease information.
    ef: Ejection fraction value to classify.
    modality: Required modality descriptor.
    view: Required view descriptor.
    frame: Optional frame descriptor.
    ---
    Results:
    Returns the canonical prompt payload.
    ########################################
    """
    return PromptPayload(
        modality=modality,
        view=view,
        frame=frame,
        sex=compute_sex_label(metadata),  # they all need some sort of check -> None!
        age_group=compute_age_group(metadata),
        bmi_group=compute_bmi_group(metadata),
        ef_group=classify_ef(float(ef)),
        pathology_label=compute_disease_label(metadata),
    )

####################################################################################################


def infer_capabilities(payload: PromptPayload) -> PromptCapabilities:  # is it really used?
    """
    ########################################
    Definition:
    Infer capabilities from the fields currently populated in one payload.
    ---
    Params:
    payload: Canonical prompt payload.
    ---
    Results:
    Returns the set of populated fields, including required fields.
    ########################################
    """
    capabilities: PromptCapabilities = {PromptField.MODALITY, PromptField.VIEW}
    for field in (
        PromptField.FRAME,
        PromptField.SEX,
        PromptField.AGE,
        PromptField.BMI,
        PromptField.EF,
        PromptField.GROUP,
    ):
        if payload.has(field):
            capabilities.add(field)
    return capabilities


def build_cardiac_prompt(metadata: dict, ef: float) -> str:  # is it really used?
    """
    ########################################
    Definition:
    Backward-compatible wrapper that builds a cardiac prompt from raw metadata.
    ---
    Params:
    metadata: Patient or study metadata with body measurements and disease labels.
    ---
    Results:
    Returns the final prompt string used in CSV export.
    ########################################
    """
    payload = build_cardiac_prompt_payload(metadata, ef)
    capabilities = infer_capabilities(payload)
    return generate_prompt(payload, capabilities)
