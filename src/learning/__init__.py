"""Share focused learning operations across model workflows.

The package owns model-independent training concepts used by more than one
architecture. Physical X/O-point detection remains under ``src.physics``;
topology supervision consumes its saved metadata without becoming a model
input.
"""

from src.learning.topology_supervision import (
    TopologyTargetScaler,
    calculate_masked_topology_loss,
    create_topology_targets,
)


__all__ = [
    "TopologyTargetScaler",
    "calculate_masked_topology_loss",
    "create_topology_targets",
]
