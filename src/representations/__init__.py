"""Prepare saved VDF representations through ordered loading branches.

Workflow
--------
Stage 1: load the saved raw or Hermite representation.
Stage 2: prepare the complete raw VDF volume for the raw branch.
Stage 3: prepare a complete Hermite input for the Hermite branch.

The unnumbered ``model_input`` module constructs sample-wise CNN tensors.
PCA owns its restartable feature-batch iterator in the analysis package, and
the autoencoder owns its intentionally separate raw-plane preparation.
"""

from src.representations.model_input import (
    RepresentationTensorSpec,
    create_representation_tensor_sample,
    describe_representation_tensor,
)
from src.representations.step_01_load_saved_representation import (
    PUBLIC_REPRESENTATIONS,
    RAW_AXIS_ORDER,
    LoadedRepresentation,
    load_saved_representation,
)
from src.representations.step_02_prepare_raw_input import (
    RAW,
    describe_raw_input,
    prepare_raw_input,
)
from src.representations.step_03_prepare_hermite_input import (
    describe_hermite_input,
)


__all__ = [
    "PUBLIC_REPRESENTATIONS",
    "RAW_AXIS_ORDER",
    "RAW",
    "LoadedRepresentation",
    "RepresentationTensorSpec",
    "describe_hermite_input",
    "describe_raw_input",
    "prepare_raw_input",
    "create_representation_tensor_sample",
    "describe_representation_tensor",
    "load_saved_representation",
]
