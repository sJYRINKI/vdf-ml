"""Stage 1: load saved representations and aligned CNN targets.

This stage follows the training entry point and precedes chronological
timestep splitting. It records the requested raw or Hermite array for
memory-mapped, sample-wise access and reads the supervised targets from
``metadata.csv`` in saved row order.

The stage returns a lightweight training-data record. Partition datasets
created from that record open process-local read-only memory maps and yield
one processed representation row, its scaled 16-value plasma
context, class target, topology target vector, and saved sample identity at a
time.
"""

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.cnn.class_mapping import ClassMapping, as_integer_array
from src.representations.model_input import (
    RepresentationTensorSpec,
    create_representation_tensor_sample,
    describe_representation_tensor,
)
from src.representations.step_01_load_saved_representation import (
    load_saved_representation,
)
from src.learning.topology_supervision import (
    create_topology_targets as _create_topology_targets,
)
from src.representations.plasma_context import (
    PLASMA_CONTEXT_FILENAME,
    PlasmaContextMemmapReader,
)


@dataclass(frozen=True)
class CnnTrainingData:
    """Describe aligned CNN inputs and targets without loading every sample.

    Stage 1 creates one instance for the complete saved dataset. The record
    keeps metadata-derived targets in memory and identifies the raw or
    Hermite array that partition datasets read sample by sample. It owns no
    open file handle; each :class:`CnnTrainingDataset` owns its process-local
    memory map.

    Parameters
    ----------
    dataset_dir : pathlib.Path
        Current dataset directory.
    representation : {"raw", "hermite"}
        Selected model representation.
    source_path : pathlib.Path
        Saved ``X.npy`` or ``X_hermite.npy`` path.
    plasma_context_path : pathlib.Path
        Aligned ``plasma_context.npy`` path with shape ``(n_samples, 16)``
        and float32 dtype.
    representation_spec : RepresentationTensorSpec
        Sample tensor shape and preprocessing convention.
    class_targets : torch.Tensor
        Contiguous output indices with shape ``(n_samples,)``.
    project_class_ids : torch.Tensor
        Physical project class IDs with shape ``(n_samples,)``.
    topology_targets : torch.Tensor
        Six Earth-radius topology targets with shape ``(n_samples, 6)`` in
        the order defined by ``TOPOLOGY_TARGET_COLUMNS``. Missing entries
        are temporarily filled by zero for masked loss calculation.
    topology_mask : torch.Tensor
        Boolean validity mask with shape ``(n_samples, 6)``.
    sample_index : torch.Tensor
        Dataset row identities in saved order.
    cid : torch.Tensor
        VLSV cell IDs in saved order.
    metadata : pandas.DataFrame
        Metadata rows aligned with every tensor.
    representation_metadata : dict
        Raw preprocessing or saved physical-VDF Hermite shape, order,
        rotation, and coefficient-axis metadata.
    class_mapping : ClassMapping
        Physical class IDs mapped to model-output indices.

    Notes
    -----
    Raw samples later have model shape ``(1, vx, vy, vz)``. Unrotated
    Hermite samples have ``(1, n_x, n_y, n_z)`` and optionally rotated
    samples have ``(1, n_parallel, n_perp1, n_perp2)``. The leading
    singleton axis is the model channel, not a saved-array axis.
    """

    dataset_dir: Path
    representation: str
    source_path: Path
    plasma_context_path: Path
    representation_spec: RepresentationTensorSpec
    class_targets: torch.Tensor
    project_class_ids: torch.Tensor
    topology_targets: torch.Tensor
    topology_mask: torch.Tensor
    sample_index: torch.Tensor
    cid: torch.Tensor
    metadata: object
    representation_metadata: dict
    class_mapping: ClassMapping

    def __post_init__(self):
        """Store dataset and representation locations as path objects."""

        object.__setattr__(self, "dataset_dir", Path(self.dataset_dir))
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(
            self,
            "plasma_context_path",
            Path(self.plasma_context_path),
        )

    @property
    def input_shape(self):
        """Return the spatial tensor shape without its channel axis.

        Model construction and feature scaling use this property to
        allocate representation-shaped parameters independently of the
        batch and singleton channel dimensions.

        Returns
        -------
        tuple of int
            Raw ``(vx, vy, vz)`` or the complete saved Hermite coefficient
            shape in its extraction frame.
        """

        return self.representation_spec.tensor_shape

    def __len__(self):
        """Return the saved sample count."""

        return self.representation_spec.sample_count

    def create_reader(self):
        """Create a reader for complete raw or Hermite CNN tensors.

        Returns
        -------
        RepresentationMemmapReader
            Process-local sample-wise reader using the CNN representation
            transform.
        """

        return RepresentationMemmapReader(
            self.source_path,
            self.representation_spec,
        )

    def create_plasma_context_reader(self):
        """Create a process-local reader for aligned plasma context.

        Returns
        -------
        PlasmaContextMemmapReader
            Read-only sample-wise access to the 16-value plasma-context
            array.
        """

        return PlasmaContextMemmapReader(self.plasma_context_path)


class RepresentationMemmapReader:
    """Own a process-local read-only representation memmap.

    CNN partition datasets use this reader to avoid materializing their
    complete processed representation in memory. The first sample request
    in a process opens the saved array; subsequent requests reuse that
    mapping until :meth:`close` is called.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Saved raw or Hermite NumPy array.
    representation_spec : RepresentationTensorSpec
        Description used to convert each saved row to its model tensor.

    Notes
    -----
    Pickling drops the open mapping so each DataLoader worker opens its own
    file descriptor. The reader does not mutate the saved array.
    """

    def __init__(self, source_path, representation_spec):
        """Create a closed reader for one saved representation array."""

        self.source_path = Path(source_path)
        self.representation_spec = representation_spec
        self._values = None
        self._process_id = None

    def read(self, sample_index):
        """Read one representation tensor in model axis order.

        The saved row is converted by the current representation
        preprocessing stage. Raw input becomes the complete logarithmic
        ``(vx, vy, vz)`` VDF; Hermite input remains the complete coefficient
        volume.

        Parameters
        ----------
        sample_index : int
            Zero-based saved dataset row.

        Returns
        -------
        numpy.ndarray
            Float32 raw ``(vx, vy, vz)`` volume or complete saved Hermite
            coefficient volume in unrotated or rotated axis order.
        """

        return create_representation_tensor_sample(
            self._open(),
            self.representation_spec,
            sample_index,
        )

    def close(self):
        """Close the process-local memory map owned by this reader.

        Partition datasets call this when their DataLoader lifecycle ends.
        A later sample read may reopen the same saved array.
        """

        if self._values is not None:
            mmap_object = getattr(self._values, "_mmap", None)
            if mmap_object is not None:
                mmap_object.close()
        self._values = None
        self._process_id = None

    def __enter__(self):
        """Return this reader for bounded resource ownership."""

        return self

    def __exit__(self, exception_type, exception, traceback):
        """Close the mapping when a bounded read block ends."""

        self.close()

    def __getstate__(self):
        """Return worker-safe state without an open memmap."""

        state = self.__dict__.copy()
        state["_values"] = None
        state["_process_id"] = None
        return state

    def _open(self):
        """Open or reuse the read-only array mapping for this process.

        DataLoader workers may receive a pickled reader created in another
        process. Tracking the process ID prevents a worker from reusing the
        parent's file descriptor while retaining one mapping across all
        sample requests made by the same worker.

        Returns
        -------
        numpy.memmap
            Saved representation with sample index on axis zero.
        """

        process_id = os.getpid()
        if self._values is not None and self._process_id == process_id:
            return self._values
        self.close()
        self._values = np.load(
            self.source_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        self._process_id = process_id
        return self._values


class CnnTrainingDataset(Dataset):
    """Expose one CNN timestep partition through sample-wise array reads.

    A training, validation, or test dataset owns one process-local
    :class:`RepresentationMemmapReader`. Indexing reads the matching saved
    representation row and combines it with the already aligned class,
    topology, and identity tensors. Call :meth:`close` after the owning
    DataLoader is no longer needed.

    Parameters
    ----------
    data : CnnTrainingData
        Memory-mapped representation description and aligned targets.
    indices : sequence of int
        Ordered row positions for one timestep partition.
    topology_scaler : TopologyTargetScaler
        Training-derived scaler applied to the six auxiliary targets.
    plasma_context_scaler : PlasmaContextScaler
        Training-derived scaler applied to the 16 plasma features.

    Notes
    -----
    Each item contains an input tensor with shape ``(1, vx, vy, vz)`` for
    raw data or ``(1, *saved_hermite_shape)`` for Hermite data, aligned
    scaled context with shape ``(16,)``, scalar class and identity values,
    and two ``(6,)`` topology tensors. Context is read independently from
    the volume and is never broadcast over velocity space.
    """

    def __init__(
        self,
        data,
        indices,
        *,
        topology_scaler,
        plasma_context_scaler,
    ):
        """Bind selected saved rows and training-derived topology scaling."""

        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        (
            self.topology_targets,
            self.topology_mask,
        ) = topology_scaler.transform_masked(
            data.topology_targets[self.indices],
            data.topology_mask[self.indices],
        )
        self.plasma_context_scaler = plasma_context_scaler
        self._reader = data.create_reader()
        self._plasma_context_reader = data.create_plasma_context_reader()

    def __len__(self):
        """Return the selected sample count."""

        return int(len(self.indices))

    def __getitem__(self, index):
        """Return one aligned input, target set, and saved sample identity.

        Parameters
        ----------
        index : int
            Position within this timestep partition.

        Returns
        -------
        dict
            ``vdf_input`` is the one-channel representation,
            ``plasma_context`` is the scaled context with shape ``(16,)``,
            ``class_target`` is the scalar physical-class target, and the
            topology values and mask have shape ``(6,)``. ``sample_index``
            and ``cid`` identify the same saved row.
        """

        source_index = int(self.indices[index])
        input_tensor = self._reader.read(source_index)
        plasma_context = self.plasma_context_scaler.transform(
            self._plasma_context_reader.read(source_index)
        )
        return {
            "vdf_input": torch.from_numpy(input_tensor[np.newaxis, ...]),
            "plasma_context": torch.from_numpy(plasma_context),
            "class_target": self.data.class_targets[source_index],
            "project_class_ids": self.data.project_class_ids[source_index],
            "topology_targets": self.topology_targets[index],
            "topology_mask": self.topology_mask[index],
            "sample_index": self.data.sample_index[source_index],
            "cid": self.data.cid[source_index],
        }

    def close(self):
        """Close the dataset's process-local representation memory map.

        Training orchestration calls this after a partition loader is no
        longer needed; the immutable training-data record remains usable
        for creating another dataset.
        """

        self._reader.close()
        self._plasma_context_reader.close()

    def __getstate__(self):
        """Return worker-safe state with a closed replacement reader."""

        state = self.__dict__.copy()
        state["_reader"] = self.data.create_reader()
        state["_plasma_context_reader"] = (
            self.data.create_plasma_context_reader()
        )
        return state


def load_cnn_data(
    dataset_dir,
    representation,
    *,
    raw_config=None,
    class_id_order=None,
):
    """Load aligned CNN targets and describe sample-wise representation access.

    This is the data-loading entry point for CNN stage 1. It reads metadata
    columns eagerly because they are small and shared by later stages, but
    retains the representation and plasma-context paths together
    with the tensor convention for the potentially large arrays. Stage 2
    uses the returned metadata for the timestep split, and DataLoaders read
    aligned representation and context rows on demand.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Dataset containing metadata and the requested representation.
    representation : {"raw", "hermite"}
        Model input representation.
    raw_config : dict, optional
        Positive log floor applied to the complete raw VDF.
    class_id_order : sequence of int, optional
        Explicit physical-class order for model outputs.

    Returns
    -------
    CnnTrainingData
        Aligned metadata targets and a memory-mapped representation
        description in saved sample order.
    """

    dataset_dir = Path(dataset_dir)
    with load_saved_representation(dataset_dir, representation) as loaded:
        representation_spec = describe_representation_tensor(
            loaded,
            raw_config=raw_config,
        )
        metadata = loaded.metadata.copy(deep=True)
        class_mapping = ClassMapping.from_metadata(
            metadata,
            class_id_order=class_id_order,
        )
        project_class_ids = as_integer_array(metadata["class_id"])
        class_targets = class_mapping.encode(project_class_ids)
        topology_targets, topology_mask = _create_topology_targets(metadata)
        sample_index = as_integer_array(metadata["sample_index"])
        cid = as_integer_array(metadata["cid"])
        source_path = dataset_dir / representation_spec.source_filename

    return CnnTrainingData(
        dataset_dir=dataset_dir,
        representation=representation,
        source_path=source_path,
        plasma_context_path=dataset_dir / PLASMA_CONTEXT_FILENAME,
        representation_spec=representation_spec,
        class_targets=torch.from_numpy(class_targets.copy()),
        project_class_ids=torch.from_numpy(project_class_ids.copy()),
        topology_targets=torch.from_numpy(topology_targets.astype(np.float32)),
        topology_mask=torch.from_numpy(topology_mask.copy()),
        sample_index=torch.from_numpy(sample_index.copy()),
        cid=torch.from_numpy(cid.copy()),
        metadata=metadata,
        representation_metadata=deepcopy(
            representation_spec.feature_metadata
        ),
        class_mapping=class_mapping,
    )


def create_cnn_dataloader(
    data,
    *,
    indices,
    topology_scaler,
    plasma_context_scaler,
    batch_size,
    shuffle=False,
    random_seed=1234,
    num_workers=0,
    pin_memory=False,
):
    """Create a deterministic sample-wise CNN DataLoader.

    Stage 6 uses this factory for each chronological partition. The
    returned loader reads representation rows only when PyTorch requests
    them, applies the training-derived topology scaling, and preserves the
    supplied row order when shuffling is disabled.

    Parameters
    ----------
    data : CnnTrainingData
        Memory-mapped representation identity and aligned targets.
    indices : sequence of int
        Partition row positions in saved order.
    topology_scaler : TopologyTargetScaler
        Training-fitted six-target scaler.
    plasma_context_scaler : PlasmaContextScaler
        Training-fitted 16-feature plasma-context scaler.
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
        Loader visiting each selected sample once per epoch.
    """

    generator = torch.Generator()
    generator.manual_seed(int(random_seed))
    return DataLoader(
        CnnTrainingDataset(
            data,
            indices,
            topology_scaler=topology_scaler,
            plasma_context_scaler=plasma_context_scaler,
        ),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        generator=generator,
    )


__all__ = [
    "CnnTrainingData",
    "CnnTrainingDataset",
    "RepresentationMemmapReader",
    "create_cnn_dataloader",
    "load_cnn_data",
]
