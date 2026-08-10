"""Stage 1: load complete three-dimensional autoencoder representations.

Raw samples remain memory mapped in ``X.npy`` with saved axis order
``(vx, vy, vz)``. Hermite samples remain memory mapped in
``X_hermite.npy`` with their complete unrotated ``(n_x, n_y, n_z)`` or
rotated ``(n_parallel, n_perp1, n_perp2)`` coefficient order. Dataset
workers copy and preprocess one row only when PyTorch requests it, so a
complete sample partition is never materialized in ordinary RAM.

The same saved row supplies the reconstruction input, six Earth-radius
topology targets, their finite-value mask, and sample identity. Topology is
an auxiliary target and is never appended to the physical model input.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.cnn.step_01_load_cnn_data import RepresentationMemmapReader
from src.representations.model_input import describe_representation_tensor
from src.representations.step_01_load_saved_representation import (
    load_saved_representation,
)
from src.learning.topology_supervision import create_topology_targets


RAW_AXIS_ORDER = ("vx", "vy", "vz")


def prepare_autoencoder_raw_volume(vdf, *, log_eps):
    """Prepare one complete raw VDF for full-volume reconstruction.

    The active raw autoencoder consumes every saved voxel. This operation
    preserves the former raw autoencoder's preprocessing exactly: values at
    or below zero are replaced by the configured positive density floor and
    ``log10`` is then applied voxelwise. The saved ``X.npy`` row is never
    modified, sliced, projected, cached, or downsampled.

    Parameters
    ----------
    vdf : numpy.ndarray
        Physical VDF in ``s^3 m^-6`` with shape ``(vx, vy, vz)``.
    log_eps : float
        Positive replacement in ``s^3 m^-6`` for nonpositive values.

    Returns
    -------
    numpy.ndarray
        Writable C-contiguous float32 volume with shape ``(vx, vy, vz)``.
    """

    values = np.asanyarray(vdf)
    volume = np.array(
        values,
        dtype=np.result_type(values.dtype, np.float32),
        order="C",
        copy=True,
    )
    volume[volume <= 0.0] = float(log_eps)
    np.log10(volume, out=volume)
    return np.asarray(volume, dtype=np.float32, order="C")


def restore_autoencoder_raw_volume(log_volume):
    """Restore a complete logarithmic raw volume to physical density.

    Parameters
    ----------
    log_volume : numpy.ndarray
        Base-10 logarithmic raw VDF with shape ``(vx, vy, vz)``.

    Returns
    -------
    numpy.ndarray
        New float32 physical VDF in ``s^3 m^-6`` with the same shape.

    Notes
    -----
    This inverse is used only for reconstruction visualization. Training
    loss compares complete normalized logarithmic volumes.
    """

    return np.power(
        np.float32(10.0),
        np.asarray(log_volume, dtype=np.float32),
    )


@dataclass(frozen=True)
class AutoencoderTrainingData:
    """Describe on-demand memory-mapped inputs and aligned topology targets.

    Parameters
    ----------
    dataset_dir : pathlib.Path
        Saved current-format dataset.
    representation : {"raw", "hermite"}
        Complete physical representation reconstructed by the model.
    source_path : pathlib.Path
        Read-only ``X.npy`` or ``X_hermite.npy`` source.
    representation_spec : RepresentationTensorSpec
        Sample count, full spatial shape, dtype, axes, and preprocessing.
    topology_targets : torch.Tensor
        Earth-radius values with shape ``(n_samples, 6)`` in the fixed
        project order: distances to X and O points, followed by X-point and
        O-point ``dx`` and ``dz`` displacements.
    topology_mask : torch.Tensor
        Boolean finite-value mask with shape ``(n_samples, 6)``.
    sample_identity : pandas.DataFrame
        Saved sample index, timestep, CID, class, and source location rows.
        Class identity is retained only for reconstruction reporting.
    representation_metadata : dict
        Raw ``(vx, vy, vz)`` or complete Hermite axis and transform metadata.

    Notes
    -----
    The large representation stays on disk. Small topology and identity
    arrays stay in CPU memory and remain row aligned with the memory map.
    Topology supervises the shared latent vector but never enters model
    inputs, split selection, or reconstruction weighting.
    """

    dataset_dir: Path
    representation: str
    source_path: Path
    representation_spec: object
    topology_targets: torch.Tensor
    topology_mask: torch.Tensor
    sample_identity: object
    representation_metadata: dict

    def __post_init__(self):
        """Store dataset and representation locations as path objects."""

        object.__setattr__(self, "dataset_dir", Path(self.dataset_dir))
        object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def input_shape(self):
        """Return the complete three-dimensional sample shape.

        Returns
        -------
        tuple of int
            Raw ``(vx, vy, vz)`` or the complete saved Hermite cube shape.
        """

        return self.representation_spec.tensor_shape

    def __len__(self):
        """Return the saved sample count."""

        return self.representation_spec.sample_count

    def create_reader(self):
        """Create one process-local complete-volume memory-map reader.

        Returns
        -------
        AutoencoderRepresentationMemmapReader
            On-demand reader using the autoencoder's raw preprocessing or
            direct signed Hermite coefficient copy.
        """

        return AutoencoderRepresentationMemmapReader(
            self.source_path,
            self.representation_spec,
        )


class AutoencoderRepresentationMemmapReader(RepresentationMemmapReader):
    """Read one complete raw or Hermite sample from a process-local memmap.

    Raw rows are transformed voxelwise without selecting a velocity plane.
    Hermite rows retain every signed coefficient. The inherited reader opens
    the saved array on the first sample request and reuses that read-only
    mapping within one DataLoader process, which bounds ordinary memory to
    requested samples and batches.
    """

    def read(self, sample_index):
        """Read one unstandardized complete reconstruction target.

        One requested memory-mapped row is copied before raw voxelwise
        logarithmic preprocessing or direct signed Hermite use. This keeps
        the saved dataset immutable and bounds resident sample data.

        Parameters
        ----------
        sample_index : int
            Zero-based saved dataset row.

        Returns
        -------
        numpy.ndarray
            Float32 raw ``(vx, vy, vz)`` volume or complete signed Hermite
            coefficient cube in its saved axis order.
        """

        values = self._open()
        sample_index = int(sample_index)
        if self.representation_spec.representation == "raw":
            return prepare_autoencoder_raw_volume(
                values[sample_index],
                log_eps=self.representation_spec.feature_metadata[
                    "log_floor"
                ],
            )
        return np.array(
            values[sample_index],
            dtype=np.float32,
            order="C",
            copy=True,
        )


class AutoencoderTrainingDataset(Dataset):
    """Expose one partition through on-demand aligned memory-mapped reads.

    Parameters
    ----------
    data : AutoencoderTrainingData
        Memory-mapped input description and aligned topology arrays.
    indices : sequence of int
        Saved row positions in stable partition order.
    topology_scaler : TopologyTargetScaler
        Scaler fitted from valid training-partition targets only.

    Notes
    -----
    Every item has input shape ``(1, d1, d2, d3)`` and aligned topology
    target and mask shape ``(6,)``. Class identity is not yielded and cannot
    become a model input or loss target.
    """

    def __init__(self, data, indices, *, topology_scaler):
        """Bind selected rows and training-derived topology scaling."""

        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        (
            self.topology_targets,
            self.topology_mask,
        ) = topology_scaler.transform_masked(
            data.topology_targets[self.indices],
            data.topology_mask[self.indices],
        )
        self._reader = data.create_reader()

    def __len__(self):
        """Return the selected sample count."""

        return len(self.indices)

    def __getitem__(self, index):
        """Return one complete input and aligned auxiliary target.

        Parameters
        ----------
        index : int
            Position within this complete-timestep partition.

        Returns
        -------
        dict
            One-channel full volume, scaled topology values and mask, and
            scalar ``sample_index`` and ``cid`` from the same saved row.
        """

        source_index = int(self.indices[index])
        identity = self.data.sample_identity.iloc[source_index]
        values = self._reader.read(source_index)
        return {
            "inputs": torch.from_numpy(values[np.newaxis, ...]),
            "topology_targets": self.topology_targets[index],
            "topology_mask": self.topology_mask[index],
            "sample_index": torch.tensor(
                int(identity["sample_index"]),
                dtype=torch.int64,
            ),
            "cid": torch.tensor(int(identity["cid"]), dtype=torch.int64),
        }

    def close(self):
        """Close the process-local representation memory map."""

        self._reader.close()

    def __getstate__(self):
        """Return worker-safe state with a closed replacement reader."""

        state = self.__dict__.copy()
        state["_reader"] = self.data.create_reader()
        return state


def load_autoencoder_data(dataset_dir, representation, *, raw_config=None):
    """Load topology metadata and describe on-demand full-volume access.

    Raw ``X.npy`` rows retain every ``(vx, vy, vz)`` voxel. Hermite
    ``X_hermite.npy`` rows retain every coefficient in their saved three-axis
    order. Only small identity and six-target topology arrays are loaded
    eagerly; physical representations are read from read-only memory maps
    only when their rows are requested.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Saved dataset containing metadata and the requested representation.
    representation : {"raw", "hermite"}
        Complete three-dimensional source reconstructed by the autoencoder.
    raw_config : mapping, optional
        Raw preprocessing containing ``log_eps`` in ``s^3 m^-6``.

    Returns
    -------
    AutoencoderTrainingData
        Memory-map description and row-aligned topology and identity data.
    """

    dataset_dir = Path(dataset_dir)
    with load_saved_representation(dataset_dir, representation) as loaded:
        spec = describe_representation_tensor(
            loaded,
            raw_config=raw_config,
        )
        metadata = loaded.metadata.copy(deep=True)
        topology_targets, topology_mask = create_topology_targets(metadata)
        sample_identity = metadata.loc[
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
        topology_targets=torch.from_numpy(
            topology_targets.astype(np.float32)
        ),
        topology_mask=torch.from_numpy(topology_mask.copy()),
        sample_identity=sample_identity,
        representation_metadata=deepcopy(spec.feature_metadata),
    )


def create_autoencoder_dataloader(
    data,
    indices,
    *,
    topology_scaler,
    batch_size,
    shuffle=False,
    random_seed=1234,
    num_workers=0,
    pin_memory=False,
):
    """Create a deterministic DataLoader with on-demand full-volume reads.

    The dataset preserves supplied partition order when unshuffled and binds
    each complete representation row to topology values and masks from the
    identical saved index.

    Parameters
    ----------
    data : AutoencoderTrainingData
        Complete-volume source and aligned auxiliary targets.
    indices : sequence of int
        Saved rows in one chronological timestep partition.
    topology_scaler : TopologyTargetScaler
        Scaling fitted from valid training targets only.
    batch_size : int
        Number of complete volumes per batch.
    shuffle : bool, optional
        Whether to shuffle training traversal.
    random_seed : int, optional
        Seed controlling DataLoader shuffling.
    num_workers : int, optional
        Processes used only for sample loading and preprocessing.
    pin_memory : bool, optional
        Whether host tensors are pinned for CUDA transfer.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader yielding complete inputs and aligned six-value targets.

    Notes
    -----
    ``num_workers`` affects input loading only. Consecutive model-stage
    placement is controlled independently by ``model_parallel_gpus``.
    """

    generator = torch.Generator()
    generator.manual_seed(int(random_seed))
    return DataLoader(
        AutoencoderTrainingDataset(
            data,
            indices,
            topology_scaler=topology_scaler,
        ),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        generator=generator,
    )


__all__ = [
    "AutoencoderRepresentationMemmapReader",
    "AutoencoderTrainingData",
    "AutoencoderTrainingDataset",
    "create_autoencoder_dataloader",
    "load_autoencoder_data",
    "prepare_autoencoder_raw_volume",
    "restore_autoencoder_raw_volume",
]
