"""Stage 4: orchestrate serial or timestep-parallel dataset extraction.

Stage 0 supplies planned sample records and parent-owned staged arrays. One
worker owns one timestep, reuses its VLSV reader and dense extractor, and
writes aligned temporary raw, plasma-context, and optional Hermite
memory maps. Samples remain sequential inside each worker.

Joblib may complete timesteps out of order, but its ordered result generator
and the parent's monotonically increasing output cursor preserve requested
timestep and within-timestep order. Only the parent copies temporary rows
into final staged arrays, so workers never write shared final memory maps or
transfer large arrays through process IPC.
"""

from pathlib import Path
import tempfile

import numpy as np
from joblib import Parallel, delayed

from src.data.timestep_sample_extraction import (
    extract_timestep_samples_to_temp,
    iter_timestep_sample_specs,
    transform_extracted_sample_to_hermite,
    write_extracted_sample_hermite,
    write_timestep_samples,
)
from src.physics.hermite_basis import DEFAULT_HERMITE_ORDER


def write_planned_timesteps(
    sample_specs_by_timestep,
    X,
    plasma_context,
    metadata,
    sample_index,
    timesteps,
    extraction_n_jobs,
    extraction_worker_count,
    sample_callback=None,
    X_hermite=None,
    velocity_grid=None,
    hermite_order=DEFAULT_HERMITE_ORDER,
    hermite_rotate=False,
    hermite_dtype=np.float32,
):
    """Extract and write every planned timestep through one ordered path.

    One configured job selects direct serial streaming. More than one job
    selects the existing timestep-level Joblib path for raw plus plasma rows
    and optional Hermite rows. Both paths preserve planned order
    and return the same next-row cursor.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mutable planned records keyed by timestep.
    X : numpy.ndarray
        Parent-owned raw array with shape ``(samples, vx, vy, vz)``.
    plasma_context : numpy.ndarray
        Parent-owned float32 context array with shape ``(samples, 16)``.
    metadata : list of dict
        Mutable output metadata in final row order.
    sample_index : int
        First unwritten dataset row.
    timesteps : sequence of int
        Planned timesteps in requested order.
    extraction_n_jobs : int
        Configured timestep-extraction worker count.
    extraction_worker_count : int
        Positive concurrency bound used for timestep chunks.
    sample_callback : callable, optional
        Serial-only aligned Hermite writer.
    X_hermite : numpy.ndarray, optional
        Parent-owned coefficient array.
    velocity_grid : dict, optional
        Shared endpoint velocity-grid descriptor.
    hermite_order : int, optional
        Coefficients saved along each Hermite axis.
    hermite_rotate : bool, optional
        Whether worker-local VDFs are rotated before projection.
    hermite_dtype : numpy.dtype, optional
        Saved coefficient dtype.

    Returns
    -------
    int
        Next unwritten dataset row.
    """

    if extraction_n_jobs == 1:
        return write_timesteps_serial(
            sample_specs_by_timestep=sample_specs_by_timestep,
            X=X,
            plasma_context=plasma_context,
            metadata=metadata,
            sample_index=sample_index,
            timesteps=timesteps,
            sample_callback=sample_callback,
        )
    return write_timesteps_parallel(
        sample_specs_by_timestep=sample_specs_by_timestep,
        X=X,
        plasma_context=plasma_context,
        metadata=metadata,
        sample_index=sample_index,
        timesteps=timesteps,
        extraction_n_jobs=extraction_n_jobs,
        extraction_worker_count=extraction_worker_count,
        X_hermite=X_hermite,
        velocity_grid=velocity_grid,
        hermite_order=hermite_order,
        hermite_rotate=hermite_rotate,
        hermite_dtype=hermite_dtype,
    )


def write_timesteps_serial(
    sample_specs_by_timestep,
    X,
    plasma_context,
    metadata,
    sample_index,
    timesteps,
    sample_callback=None,
):
    """Stream planned timesteps through one reader at a time.

    Serial execution shares the same per-timestep sample iterator as worker
    execution. Every raw row, sixteen-value context row, optional Hermite row,
    and metadata record uses one final sample index.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mutable planned records keyed by timestep.
    X : numpy.ndarray
        Writable raw dataset array.
    plasma_context : numpy.ndarray
        Writable float32 context array with sixteen columns.
    metadata : list of dict
        Mutable final metadata rows.
    sample_index : int
        First unwritten row.
    timesteps : sequence of int
        Planned timesteps in output order.
    sample_callback : callable, optional
        Aligned optional Hermite writer.

    Returns
    -------
    int
        Next unwritten row.
    """

    for timestep in timesteps:
        sample_specs = sample_specs_by_timestep.pop(int(timestep))
        sample_index = write_timestep_samples(
            X=X,
            plasma_context=plasma_context,
            metadata=metadata,
            timestep_samples=iter_timestep_sample_specs(sample_specs),
            sample_index=sample_index,
            sample_callback=sample_callback,
        )
    return sample_index


def write_timesteps_parallel(
    sample_specs_by_timestep,
    X,
    plasma_context,
    metadata,
    sample_index,
    timesteps,
    extraction_n_jobs,
    extraction_worker_count,
    X_hermite=None,
    velocity_grid=None,
    hermite_order=DEFAULT_HERMITE_ORDER,
    hermite_rotate=False,
    hermite_dtype=np.float32,
):
    """Extract timestep chunks and merge their aligned rows in the parent.

    Each Joblib task writes worker-owned raw and context memory maps plus an
    optional Hermite map. Results may finish in any order; ordered delivery
    and the current parent cursor define final ranges. The parent copies all
    arrays through the identical slice and places metadata at those indexes.
    No worker receives a writable final sink.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mutable planned records keyed by timestep.
    X : numpy.memmap
        Parent-owned raw final staging array.
    plasma_context : numpy.memmap
        Parent-owned float32 ``(samples, 16)`` staging array.
    metadata : list of dict
        Mutable final metadata rows.
    sample_index : int
        First unwritten final row.
    timesteps : sequence of int
        Planned timesteps in requested order.
    extraction_n_jobs : int
        Joblib process count.
    extraction_worker_count : int
        Maximum timesteps in one submitted chunk.
    X_hermite : numpy.memmap, optional
        Parent-owned coefficient staging array.
    velocity_grid : dict, optional
        Shared endpoint velocity-grid descriptor.
    hermite_order : int, optional
        Coefficients saved along each transform axis.
    hermite_rotate : bool, optional
        Whether workers rotate VDFs before projection.
    hermite_dtype : numpy.dtype, optional
        Saved coefficient dtype.

    Returns
    -------
    int
        Next unwritten final row.
    """

    with tempfile.TemporaryDirectory(
        prefix="extraction_",
        dir=Path(X.filename).parent,
    ) as temp_dir:
        temp_dir = Path(temp_dir)
        for timestep_chunk in iter_chunks(timesteps, extraction_worker_count):
            chunk_specs_by_timestep = {
                int(timestep): sample_specs_by_timestep.pop(int(timestep))
                for timestep in timestep_chunk
            }
            chunk_output_slices = []
            next_sample_index = sample_index
            for timestep in timestep_chunk:
                timestep_sample_count = len(
                    chunk_specs_by_timestep[int(timestep)]
                )
                chunk_output_slices.append(
                    slice(
                        next_sample_index,
                        next_sample_index + timestep_sample_count,
                    )
                )
                next_sample_index += timestep_sample_count
            chunk_results = extract_timestep_chunk_parallel(
                sample_specs_by_timestep=chunk_specs_by_timestep,
                timestep_chunk=timestep_chunk,
                extraction_n_jobs=extraction_n_jobs,
                temp_dir=temp_dir,
                sample_shape=X.shape[1:],
                dtype=X.dtype,
                hermite_enabled=X_hermite is not None,
                velocity_grid=velocity_grid,
                hermite_order=hermite_order,
                hermite_rotate=hermite_rotate,
                hermite_dtype=hermite_dtype,
            )
            for extracted_timestep, output_slice in zip(
                chunk_results,
                chunk_output_slices,
                strict=True,
            ):
                write_extracted_timestep(
                    X=X,
                    plasma_context=plasma_context,
                    X_hermite=X_hermite,
                    metadata=metadata,
                    extracted_timestep=extracted_timestep,
                    output_slice=output_slice,
                )
            sample_index = next_sample_index
    return sample_index


def write_extracted_timestep(
    X,
    plasma_context,
    metadata,
    extracted_timestep,
    output_slice,
    X_hermite=None,
):
    """Merge one completed timestep through one aligned final row slice.

    Joblib results may complete out of order, while the ordered generator
    presents them in planned submission order. The parent computes every
    timestep's final slice before submitting its chunk. Read-only temporary raw,
    context, and optional Hermite maps are copied through the same slice;
    metadata sample indexes are offset into that range. Only this parent
    process writes final dataset arrays.

    Parameters
    ----------
    X : numpy.ndarray
        Writable final raw staging array.
    plasma_context : numpy.ndarray
        Writable final context staging array with shape ``(samples, 16)``.
    metadata : list of dict
        Mutable final metadata rows.
    extracted_timestep : dict
        Worker descriptor containing temporary paths and local metadata.
    output_slice : slice
        Precomputed final row range for this timestep.
    X_hermite : numpy.ndarray, optional
        Writable final coefficient staging array.

    Returns
    -------
    None
        Final arrays and metadata are updated in place by the parent.
    """

    n_samples = int(extracted_timestep["n_samples"])
    if n_samples == 0:
        return None
    sample_index = int(output_slice.start)
    write_end = int(output_slice.stop)
    X_temp = np.load(extracted_timestep["X_path"], mmap_mode="r")
    plasma_context_temp = np.load(
        extracted_timestep["plasma_context_path"],
        mmap_mode="r",
    )
    X[sample_index:write_end] = X_temp[:n_samples]
    plasma_context[sample_index:write_end] = plasma_context_temp[:n_samples]
    if X_hermite is not None:
        X_hermite_temp = np.load(
            extracted_timestep["X_hermite_path"],
            mmap_mode="r",
        )
        X_hermite[sample_index:write_end] = X_hermite_temp[:n_samples]
        X_hermite_temp._mmap.close()
    for metadata_row in extracted_timestep["metadata"]:
        output_row = dict(metadata_row)
        output_row["sample_index"] = sample_index + int(
            metadata_row["sample_index"]
        )
        metadata.append(output_row)
    plasma_context_temp._mmap.close()
    X_temp._mmap.close()
    return None


def extract_timestep_chunk_parallel(
    sample_specs_by_timestep,
    timestep_chunk,
    extraction_n_jobs,
    temp_dir,
    sample_shape,
    dtype,
    hermite_enabled=False,
    velocity_grid=None,
    hermite_order=DEFAULT_HERMITE_ORDER,
    hermite_rotate=False,
    hermite_dtype=np.float32,
):
    """Submit one bounded ordered chunk of timestep extraction tasks.

    Plain planned records and numerical settings cross the Joblib process
    boundary. Each worker owns its local temporary arrays and returns only
    paths plus metadata; final writable arrays are not worker arguments.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Planned records keyed by timestep.
    timestep_chunk : sequence of int
        Timesteps submitted in output order.
    extraction_n_jobs : int
        Joblib worker count.
    temp_dir : str or pathlib.Path
        Staging-local temporary root.
    sample_shape : tuple of int
        Raw VDF shape in ``[vx, vy, vz]`` order.
    dtype : numpy.dtype
        Raw temporary storage dtype.
    hermite_enabled : bool, optional
        Whether workers write optional coefficient arrays.
    velocity_grid : dict, optional
        Shared physical velocity-grid descriptor.
    hermite_order : int, optional
        Coefficients saved along each axis.
    hermite_rotate : bool, optional
        Whether workers rotate before projection.
    hermite_dtype : numpy.dtype, optional
        Saved coefficient dtype.

    Returns
    -------
    iterable of dict
        Worker descriptors yielded in submission order.
    """

    parallel = create_extraction_parallel(extraction_n_jobs)
    return parallel(
        delayed(extract_timestep_samples_to_temp)(
            sample_specs=sample_specs_by_timestep[int(timestep)],
            temp_dir=temp_dir,
            sample_shape=sample_shape,
            dtype=dtype,
            hermite_enabled=hermite_enabled,
            velocity_grid=velocity_grid,
            hermite_order=hermite_order,
            hermite_rotate=hermite_rotate,
            hermite_dtype=hermite_dtype,
        )
        for timestep in timestep_chunk
    )


def create_extraction_parallel(extraction_n_jobs):
    """Create the ordered Joblib executor for timestep extraction.

    Parameters
    ----------
    extraction_n_jobs : int
        Configured process worker count.

    Returns
    -------
    joblib.Parallel
        Executor yielding timestep results in submission order.
    """

    return Parallel(n_jobs=extraction_n_jobs, return_as="generator")


def iter_chunks(items, chunk_size):
    """Yield consecutive bounded slices from an ordered sequence.

    Parameters
    ----------
    items : sequence
        Ordered values divided without reordering.
    chunk_size : int
        Maximum values in one returned slice.

    Yields
    ------
    sequence
        Consecutive slices containing at most ``chunk_size`` values.
    """

    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


__all__ = [
    "create_extraction_parallel",
    "extract_timestep_chunk_parallel",
    "extract_timestep_samples_to_temp",
    "iter_timestep_sample_specs",
    "transform_extracted_sample_to_hermite",
    "write_extracted_sample_hermite",
    "write_extracted_timestep",
    "write_planned_timesteps",
    "write_timestep_samples",
    "write_timesteps_parallel",
    "write_timesteps_serial",
]
