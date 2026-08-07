"""Run PCA analysis through eight ordered stages.

Workflow
--------
Stage 0: Orchestrate one raw or Hermite PCA analysis.
Stage 1: Open the saved representation and yield prepared sample batches.
Stage 2: Fit global feature scaling statistics from those batches.
Stage 3: Fit an incremental PyTorch basis and calculate batched scores.
Stage 4: Fit internal KMeans partitions in leading PCA score space.
Stage 5: Calculate physical-class counts, purity, and neighbor diagnostics.
Stage 6: Fit t-SNE to leading PCA score columns for visualization.
Stage 7: Save two physical-class figures and one text metrics report.
"""

from src.analysis.step_00_run_pca import run_pca_analysis

__all__ = ["run_pca_analysis"]
