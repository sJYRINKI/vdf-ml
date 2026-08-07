"""Run memory-bounded label-blind PyTorch PCA for one representation.

This thin command loads one PCA configuration, applies focused numerical
overrides, and delegates all memory-mapped feature iteration and numerical
work to ``src.analysis``. Internal KMeans diagnostics, nearest-
neighbor purity, label-blind t-SNE, physical-class plotting, and the
consolidated metrics report remain package-owned stages.

Ordinary execution uses one selected CPU or CUDA device. ``--multi-gpu`` is
the thin torchrun boundary for the package's one-process-per-GPU,
feature-sharded NCCL path; only rank zero prints saved outputs.
"""

import argparse

from src.analysis import run_pca_analysis
from src.configuration import load_config


def main(argv=None):
    """
    Run one configured raw or Hermite PCA analysis.

    The command boundary collects only configuration, dataset,
    representation, method, device, batch size, downstream component width,
    neighbor count, distributed-mode selection, and output paths. The
    ordered analysis package performs three memory-bounded feature passes,
    including feature-sharded torchrun/NCCL passes when requested, followed
    by label-blind fitting, post-fit physical comparison, and writing of
    exactly two figures plus one report on rank zero.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero when the analysis outputs are saved.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run label-blind PyTorch PCA with internal physical-class "
            "diagnostics and t-SNE on one raw or Hermite representation."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="PCA analysis YAML configuration path.",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="input dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Representation-specific output directory.",
    )
    parser.add_argument(
        "--representation",
        required=True,
        choices=("raw", "hermite"),
        help="public representation to analyse.",
    )
    parser.add_argument(
        "--pca-method",
        choices=("full", "lowrank"),
        help="Override PyTorch PCA method with full or lowrank.",
    )
    parser.add_argument(
        "--device",
        help="Override PCA device with auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the number of saved samples per PCA batch.",
    )
    parser.add_argument(
        "--analysis-components",
        type=int,
        help=(
            "Override the leading PCA width shared by KMeans, neighbors, "
            "and t-SNE."
        ),
    )
    parser.add_argument(
        "--nearest-neighbors",
        type=int,
        help="Override the PCA-space physical-label neighbor count.",
    )
    parser.add_argument(
        "--multi-gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override feature-sharded multi-GPU PCA launched by torchrun."
        ),
    )
    args = parser.parse_args(argv)

    result = run_pca_analysis(
        config=load_config(args.config),
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        representation=args.representation,
        pca_method=args.pca_method,
        device=args.device,
        batch_size=args.batch_size,
        analysis_components=args.analysis_components,
        nearest_neighbors=args.nearest_neighbors,
        multi_gpu=args.multi_gpu,
    )
    if result is None:
        return 0
    print(f"Output: {result['output_dir']}")
    print(f"Representation: {result['representation']}")
    print(f"PCA method: {result['pca_method']}")
    print(f"Device: {result['device']}")
    print(f"Batch size: {result['batch_size']}")
    print(f"Analysis components: {result['analysis_components']}")
    print(f"Nearest neighbors: {result['nearest_neighbors']}")
    print(
        "Saved samples x features: "
        f"{result['sample_count']} x {result['feature_count']}"
    )
    print(f"Components: {result['n_components']}")
    print(f"Weighted KMeans purity: {result['weighted_purity']:.6f}")
    for path in result["paths"].values():
        print(f"Saved {path.name}")
    return 0


if __name__ == "__main__":
    main()
