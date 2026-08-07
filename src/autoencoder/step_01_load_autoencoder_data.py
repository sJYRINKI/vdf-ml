"""Stage 1: load saved representation samples for reconstruction.

This stage follows the autoencoder entry point and precedes timestep
splitting. It records the raw or Hermite array for memory-mapped,
sample-wise access and retains only the sample identity needed for the
chronological split.

The returned training-data record retains physical class identity only for
post-training reconstruction reporting. Partition DataLoaders read one
representation row at a time and pass only the input and its saved sample
identity to the reconstruction model; class labels never enter training.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.cnn.step_01_load_cnn_data import RepresentationMemmapReader
from src.representations.model_input import (
    RepresentationTensorSpec,
    describe_representation_tensor,
)
from src.representations.step_01_load_saved_representation import (
    load_saved_representation,
)


AUTOENCODER_RAW_XZ_FULL = "raw_xz_full"
AUTOENCODER_RAW_AXIS_ORDER = ("vx", "vz")


def prepare_autoencoder_raw_plane(vdf, *, slice_index, log_eps):
    """Prepare the autoencoder's existing middle-``vy`` raw plane.

    The autoencoder intentionally retains its established two-dimensional
    reconstruction target while PCA and CNN use complete raw volumes. This
    function selects the saved ``(vx, vz)`` plane, applies the positive
    phase-space-density floor, and computes ``log10`` without changing the
    source VDF.

    Parameters
    ----------
    vdf : numpy.ndarray
        Physical VDF in ``s^3 m^-6`` with shape ``(vx, vy, vz)``.
    slice_index : int
        Saved middle-``vy`` index used by the autoencoder.
    log_eps : float
        Positive floor in ``s^3 m^-6`` applied before ``log10``.

    Returns
    -------
    numpy.ndarray
        Writable C-contiguous float32 plane with shape ``(vx, vz)``.
    """

    values = np.asanyarray(vdf)
    plane = np.array(
        values[:, int(slice_index), :],
        dtype=np.result_type(values.dtype, np.float32),
        order="C",
        copy=True,
    )
    plane[plane <= 0.0] = float(log_eps)
    np.log10(plane, out=plane)
    return np.asarray(plane, dtype=np.float32, order="C")


def restore_autoencoder_raw_plane(log_plane):
    """Restore a raw autoencoder plane to physical VDF density.

    Raw autoencoder inputs use the base-10 logarithm created by
    :func:`prepare_autoencoder_raw_plane`. Final reconstruction plotting
    applies this inverse beside that forward owner so original and model
    output planes return to nonnegative phase-space density before entering
    the shared extraction Stage 6 renderer.

    Parameters
    ----------
    log_plane : numpy.ndarray
        Raw autoencoder plane in base-10 logarithmic representation units
        with shape ``(vx, vz)``.

    Returns
    -------
    numpy.ndarray
        New float32 physical VDF plane in ``s^3 m^-6`` with shape
        ``(vx, vz)``.

    Notes
    -----
    This display inverse does not modify model inputs, saved ``X.npy``
    values, normalization statistics, loss values, or checkpoints.
    """

    return np.power(
        np.float32(10.0),
        np.asarray(log_plane, dtype=np.float32),
    )


@dataclass(frozen=True)
class AutoencoderTrainingData:
    """Describe autoencoder samples without loading the full representation.

    Stage 1 creates this immutable record for a current dataset. It owns no
    open memory map; :class:`AutoencoderTrainingDataset` instances open the
    saved representation in their own process and read selected rows when
    PyTorch requests them.

    Parameters
    ----------
    dataset_dir : pathlib.Path
        Saved current-format dataset.
    representation : {"raw", "hermite"}
        Representation reconstructed by the model.
    source_path : pathlib.Path
        ``X.npy`` or ``X_hermite.npy`` memmap source.
    representation_spec : RepresentationTensorSpec
        Sample count, shape, dtype, and feature convention.
    sample_identity : pandas.DataFrame
        ``sample_index``, ``timestep``, ``cid``, ``class_id``,
        ``class_name``, and ``file_location`` rows in source order. Class
        columns are retained only for post-training reporting. The source
        location and CID let final raw reconstruction figures read the
        same-cell sparsity threshold directly from the original VLSV file.
    representation_metadata : dict
        Raw preprocessing or saved physical-VDF Hermite shape, order,
        rotation, and coefficient-axis metadata.

    Notes
    -----
    The model receives no labels, topology values, or plotting thresholds.
    ``sample_identity`` preserves saved ordering, supports complete-timestep
    splitting, associates final per-sample reconstruction errors with
    physical classes, and identifies the source VLSV cells whose thresholds
    are read only after optimization when raw figures are rendered.
    """

    dataset_dir: Path
    representation: str
    source_path: Path
    representation_spec: RepresentationTensorSpec
    sample_identity: object
    representation_metadata: dict

    def __post_init__(self):
        """Store dataset and representation locations as path objects."""

        object.__setattr__(self, "dataset_dir", Path(self.dataset_dir))
        object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def input_shape(self):
        """Return the representation shape without a channel axis.

        Model construction and normalization use this property for
        representation-shaped parameters independently of batch and
        singleton channel dimensions.

        Returns
        -------
        tuple of int
            Raw ``(vx, vz)`` or complete saved Hermite coefficient shape.
        """

        return self.representation_spec.tensor_shape

    def __len__(self):
        """Return the saved sample count."""

        return self.representation_spec.sample_count

    def create_reader(self):
        """Create the autoencoder-specific sample reader.

        Returns
        -------
        AutoencoderRepresentationMemmapReader
            Reader preserving the established raw plane or complete
            Hermite coefficient input.
        """

        return AutoencoderRepresentationMemmapReader(
            self.source_path,
            self.representation_spec,
        )


class AutoencoderRepresentationMemmapReader(RepresentationMemmapReader):
    """Read autoencoder inputs without sharing the CNN raw-input transform.

    Raw rows retain the established middle-``vy`` plane behavior, while
    Hermite rows retain the complete coefficient cube. Process-local
    memory-map ownership is inherited from the common reader mechanism.
    """

    def read(self, sample_index):
        """Read one unstandardized autoencoder reconstruction target.

        Dataset batches and training-only scaler fitting both call this
        method, so the retained raw-plane convention is identical in the
        model input and its normalization statistics. Hermite rows remain
        complete coefficient cubes.

        Parameters
        ----------
        sample_index : int
            Zero-based saved dataset row.

        Returns
        -------
        numpy.ndarray
            Float32 raw ``(vx, vz)`` plane or complete Hermite coefficient
            cube.
        """

        values = self._open()
        sample_index = int(sample_index)
        if self.representation_spec.representation == "raw":
            metadata = self.representation_spec.feature_metadata
            return prepare_autoencoder_raw_plane(
                values[sample_index],
                slice_index=metadata["slice_index"],
                log_eps=metadata["log_floor"],
            )
        return np.array(
            values[sample_index],
            dtype=np.float32,
            order="C",
            copy=True,
        )


class AutoencoderTrainingDataset(Dataset):
    """Expose one autoencoder partition through sample-wise array reads.

    Each instance owns a process-local representation reader. Indexing
    returns the selected raw or Hermite tensor and identity from the same
    saved row. The caller closes the reader after the DataLoader has
    completed its partition.

    Parameters
    ----------
    data : AutoencoderTrainingData
        Memory-mapped representation description and ordered identity.
    indices : sequence of int
        Saved row positions exposed by this dataset.

    Notes
    -----
    Raw items have input shape ``(1, vx, vz)``. Hermite items have
    ``(1, *saved_coefficient_shape)`` in unrotated or rotated Hermite-axis
    order. The leading dimension is the singleton model channel.
    """

    def __init__(self, data, indices):
        """Bind selected saved rows to a process-local representation reader."""

        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self._reader = data.create_reader()

    def __len__(self):
        """Return the selected sample count."""

        return len(self.indices)

    def __getitem__(self, index):
        """Return one representation tensor and its saved identity.

        Parameters
        ----------
        index : int
            Position within this timestep partition.

        Returns
        -------
        dict
            One-channel ``inputs`` plus scalar ``sample_index`` and ``cid``
            tensors from the same saved row.
        """

        source_index = int(self.indices[index])
        identity = self.data.sample_identity.iloc[source_index]
        values = self._reader.read(source_index)
        return {
            "inputs": torch.from_numpy(values[np.newaxis, ...]),
            "sample_index": torch.tensor(
                int(identity["sample_index"]),
                dtype=torch.int64,
            ),
            "cid": torch.tensor(int(identity["cid"]), dtype=torch.int64),
        }

    def close(self):
        """Close the process-local representation memory map.

        Training orchestration calls this after a partition loader is no
        longer needed. The immutable training-data record remains available
        for constructing another partition dataset.
        """

        self._reader.close()

    def __getstate__(self):
        """Return worker-safe state with a closed replacement reader."""

        state = self.__dict__.copy()
        state["_reader"] = self.data.create_reader()
        return state


def load_autoencoder_data(
    dataset_dir,
    representation,
    *,
    raw_config=None,
):
    """Load autoencoder identity and describe sample-wise representation access.

    This first autoencoder data stage reads sample identity for chronological
    splitting plus physical class identity for final reporting. It retains the
    source VLSV path and CID so the compact final raw figure can obtain its
    physical sparsity threshold through the same source-reading path as
    extraction Stage 6. The potentially large representation stays on disk
    and is opened through a read-only memory map by each partition dataset in
    later stages. Class identity and plotting thresholds are never yielded to
    the model or reconstruction objective.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Saved dataset containing metadata and the requested array.
    representation : {"raw", "hermite"}
        Representation reconstructed by the autoencoder.
    raw_config : mapping, optional
        Raw preprocessing containing ``log_eps``.

    Returns
    -------
    AutoencoderTrainingData
        Memory-map source, tensor convention, and minimal sample identity
        in saved row order.
    """

    dataset_dir = Path(dataset_dir)
    with load_saved_representation(dataset_dir, representation) as loaded:
        if representation == "raw":
            source_shape = tuple(int(value) for value in loaded.values.shape)
            slice_index = int(loaded.velocity_grid["slice_index"])
            tensor_shape = (source_shape[1], source_shape[3])
            feature_metadata = {
                "representation": "raw",
                "representation_version": AUTOENCODER_RAW_XZ_FULL,
                "source_array_filename": "X.npy",
                "source_axis_order": ["vx", "vy", "vz"],
                "source_shape": list(source_shape),
                "source_dtype": np.dtype(loaded.values.dtype).str,
                "slice_axis": "vy",
                "slice_index": slice_index,
                "output_axis_order": list(AUTOENCODER_RAW_AXIS_ORDER),
                "log_floor": float(raw_config["log_eps"]),
                "feature_shape_2d": list(tensor_shape),
                "flattened_feature_count": int(np.prod(tensor_shape)),
            }
            spec = RepresentationTensorSpec(
                representation="raw",
                source_filename=loaded.source_filename,
                source_shape=source_shape,
                source_dtype=np.dtype(loaded.values.dtype).str,
                sample_count=len(loaded.values),
                tensor_shape=tensor_shape,
                feature_metadata=feature_metadata,
            )
        else:
            spec = describe_representation_tensor(loaded)
        sample_identity = loaded.metadata.loc[
            :,
            [
                "sample_index",
                "timestep",
                "cid",
                "class_id",
                "class_name",
                "file_location",
            ],
        ].copy(deep=True)
    return AutoencoderTrainingData(
        dataset_dir=dataset_dir,
        representation=representation,
        source_path=dataset_dir / spec.source_filename,
        representation_spec=spec,
        sample_identity=sample_identity,
        representation_metadata=deepcopy(spec.feature_metadata),
    )


def create_autoencoder_dataloader(
    data,
    indices,
    *,
    batch_size,
    shuffle=False,
    random_seed=1234,
    num_workers=0,
    pin_memory=False,
):
    """Create a deterministic sample-wise reconstruction DataLoader.

    The returned loader reads only requested representation rows and yields
    no supervised targets. Stage 5 uses shuffled training traversal, while
    validation and test loaders preserve their supplied row order.

    Parameters
    ----------
    data : AutoencoderTrainingData
        Memory-mapped representation description and saved identity.
    indices : sequence of int
        Saved rows in one timestep partition.
    batch_size : int
        Number of samples per batch.
    shuffle : bool, optional
        Whether to shuffle traversal order for optimization.
    random_seed : int, optional
        Seed controlling DataLoader shuffling.
    num_workers : int, optional
        Number of PyTorch worker processes.
    pin_memory : bool, optional
        Whether host tensors are pinned for CUDA transfer.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader yielding one-channel inputs and sample identities.
    """

    generator = torch.Generator()
    generator.manual_seed(int(random_seed))
    return DataLoader(
        AutoencoderTrainingDataset(data, indices),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        generator=generator,
    )


__all__ = [
    "AUTOENCODER_RAW_XZ_FULL",
    "AutoencoderRepresentationMemmapReader",
    "AutoencoderTrainingData",
    "AutoencoderTrainingDataset",
    "create_autoencoder_dataloader",
    "load_autoencoder_data",
    "prepare_autoencoder_raw_plane",
    "restore_autoencoder_raw_plane",
]
