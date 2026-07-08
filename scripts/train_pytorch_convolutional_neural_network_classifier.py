#python scripts/train_pytorch_convolutional_neural_network_classifier.py --config configs/train_pytorch_convolutional_neural_network_classifier.yaml --dataset-id 3408_100 --model-id v1.0

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.pytorch_cnn import (
    apply_pytorch_cnn_training_filter_override,
    train_pytorch_convolutional_neural_network_classifier,
)


def main(
    config_path,
    dataset_id,
    model_id,
    training_filter=None,
    dry_run=None,
    run_source_model=None,
    class_weight=None,
    class_weight_by_class=None,
    sampler=None,
    sampler_weight_by_class=None,
    embedding_k_neighbors=None,
    max_removed_fraction_per_class=None,
    min_point_neighbor_fraction=None,
    min_point_neighbor_fraction_by_class=None,
    max_same_class_fraction=None,
    max_same_class_fraction_by_class=None,
):
    """
    Train a PyTorch CNN VDF classifier.

    Parameters
    ----------
    config_path : str
        Path to training config.
    dataset_id : str
        Dataset identifier.
    model_id : str
        Model identifier.
    training_filter : str, optional
        Optional training-filter mode override.
    dry_run : bool, optional
        Optional training-filter dry-run override.
    run_source_model : bool, optional
        Optional CNN embedding source-model override.
    class_weight : str, optional
        Optional model class-weight override.
    class_weight_by_class : sequence of str, optional
        Optional manual class weights as ``class=value``.
    sampler : str, optional
        Optional training sampler override.
    sampler_weight_by_class : sequence of str, optional
        Optional manual sampler weights as ``class=value``.
    embedding_k_neighbors : int, optional
        Optional CNN embedding kNN neighbor-count override.
    max_removed_fraction_per_class : float, optional
        Optional maximum removed fraction per class.
    min_point_neighbor_fraction : float, optional
        Optional point-neighbor fraction threshold for filter candidates.
    min_point_neighbor_fraction_by_class : sequence of str, optional
        Optional class-specific point-neighbor thresholds as ``class=value``.
    max_same_class_fraction : float, optional
        Optional same-class fraction threshold for filter candidates.
    max_same_class_fraction_by_class : sequence of str, optional
        Optional class-specific same-class thresholds as ``class=value``.
    """

    config = load_config(config_path)
    apply_pytorch_cnn_training_filter_override(
        config=config,
        training_filter=training_filter,
        dry_run=dry_run,
        run_source_model=run_source_model,
        class_weight=class_weight,
        class_weight_by_class=class_weight_by_class,
        sampler=sampler,
        sampler_weight_by_class=sampler_weight_by_class,
        embedding_k_neighbors=embedding_k_neighbors,
        max_removed_fraction_per_class=max_removed_fraction_per_class,
        min_point_neighbor_fraction=min_point_neighbor_fraction,
        min_point_neighbor_fraction_by_class=min_point_neighbor_fraction_by_class,
        max_same_class_fraction=max_same_class_fraction,
        max_same_class_fraction_by_class=max_same_class_fraction_by_class,
    )
    train_pytorch_convolutional_neural_network_classifier(
        config=config,
        dataset_id=dataset_id,
        model_id=model_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PyTorch CNN classifier on VDF data."
    )
    parser.add_argument("--config", required=True, help="Path to CNN config.")
    parser.add_argument("--dataset-id", required=True, help="Dataset identifier.")
    parser.add_argument("--model-id", required=True, help="Model identifier.")
    parser.add_argument(
        "--training-filter",
        choices=["none", "pca", "cnn_embedding_knn"],
        default=None,
        help="Optional training-filter override. If omitted, use the YAML config.",
    )
    dry_run_group = parser.add_mutually_exclusive_group()
    dry_run_group.add_argument(
        "--training-filter-dry-run",
        dest="dry_run",
        action="store_true",
        default=None,
        help="Report filter candidates without removing them.",
    )
    dry_run_group.add_argument(
        "--training-filter-apply",
        dest="dry_run",
        action="store_false",
        default=None,
        help="Apply the training filter and remove selected in-memory samples.",
    )
    source_model_group = parser.add_mutually_exclusive_group()
    source_model_group.add_argument(
        "--cnn-embedding-run-source-model",
        dest="run_source_model",
        action="store_true",
        default=None,
        help="Train the source CNN used to create embedding kNN filter metrics.",
    )
    source_model_group.add_argument(
        "--cnn-embedding-use-existing-metrics",
        dest="run_source_model",
        action="store_false",
        default=None,
        help="Use existing embedding kNN metrics instead of training a source CNN.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "sqrt_balanced", "balanced", "manual"],
        default=None,
        help="Optional model class-weight override.",
    )
    parser.add_argument(
        "--class-weight-by-class",
        action="append",
        default=None,
        metavar="CLASS=VALUE",
        help="Optional manual class weight. Can be repeated.",
    )
    parser.add_argument(
        "--sampler",
        choices=["none", "balanced", "sqrt_balanced", "manual"],
        default=None,
        help="Optional training sampler override.",
    )
    parser.add_argument(
        "--sampler-weight-by-class",
        action="append",
        default=None,
        metavar="CLASS=VALUE",
        help="Optional manual sampler weight. Can be repeated.",
    )
    parser.add_argument(
        "--cnn-embedding-k-neighbors",
        type=int,
        default=None,
        help="Optional CNN embedding kNN neighbor-count override.",
    )
    parser.add_argument(
        "--filter-max-removed-fraction-per-class",
        type=float,
        default=None,
        help="Optional maximum removed fraction per class for training filter.",
    )
    parser.add_argument(
        "--filter-min-point-neighbor-fraction",
        type=float,
        default=None,
        help="Optional point-neighbor threshold for training-filter candidates.",
    )
    parser.add_argument(
        "--filter-min-point-neighbor-fraction-by-class",
        action="append",
        default=None,
        metavar="CLASS=VALUE",
        help=(
            "Optional class-specific point-neighbor threshold. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--filter-max-same-class-fraction",
        type=float,
        default=None,
        help="Optional same-class threshold for training-filter candidates.",
    )
    parser.add_argument(
        "--filter-max-same-class-fraction-by-class",
        action="append",
        default=None,
        metavar="CLASS=VALUE",
        help="Optional class-specific same-class threshold. Can be repeated.",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        dataset_id=args.dataset_id,
        model_id=args.model_id,
        training_filter=args.training_filter,
        dry_run=args.dry_run,
        run_source_model=args.run_source_model,
        class_weight=args.class_weight,
        class_weight_by_class=args.class_weight_by_class,
        sampler=args.sampler,
        sampler_weight_by_class=args.sampler_weight_by_class,
        embedding_k_neighbors=args.cnn_embedding_k_neighbors,
        max_removed_fraction_per_class=(
            args.filter_max_removed_fraction_per_class
        ),
        min_point_neighbor_fraction=args.filter_min_point_neighbor_fraction,
        min_point_neighbor_fraction_by_class=(
            args.filter_min_point_neighbor_fraction_by_class
        ),
        max_same_class_fraction=args.filter_max_same_class_fraction,
        max_same_class_fraction_by_class=(
            args.filter_max_same_class_fraction_by_class
        ),
    )
