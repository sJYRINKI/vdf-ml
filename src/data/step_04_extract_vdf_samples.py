"""Stage 4: extract and write planned VDF samples.

This stage follows sample planning and precedes final dataset-file writing.
It reuses one VLSV reader and producer set per timestep, expands sparse VDFs
to ``[vx, vy, vz]`` cubes, and writes rows in stable sample order.

Inputs are ordered sample records and staged arrays. Outputs are aligned raw
rows, optional same-cell Hermite callbacks, and metadata records. Parallel
raw extraction uses bounded temporary arrays without changing final order.
"""

from pathlib import Path
import tempfile
import time

import analysator as pt
import numpy as np
from joblib import Parallel, delayed

from src.data.dense_vdf import VdfExtractor
from src.data.metadata import create_sample_metadata_row
from src.data.step_01_open_vlsv_files import (
    resolve_vlsv_source,
)
from src.physics.hermite_basis import DEFAULT_HERMITE_ORDER
from src.physics.hermite_rotation import rotate_vdf
from src.physics.hermite_transform import vdf_to_hermite
from src.physics.physical_context import get_hermite_rotation_context
from src.physics.vlsv_physical_values import resolve_bulk_velocity_source


def write_remaining_timesteps(
    sample_specs_by_timestep,
    X,
    metadata,
    sample_index,
    timesteps,
    extraction_n_jobs,
    extraction_worker_count,
    include_rotation_context=False,
    sample_callback=None,
):
    """Extract and write all timesteps after the first open-reader stream.

    Stage 0 calls this dispatcher after writing the first timestep used for
    shape discovery. It selects serial streaming when Hermite context must
    remain sample-wise and the configured parallel raw path otherwise; both
    branches return the same next-row index.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mutable mapping from timestep to planned sample records.
    X : numpy.ndarray
        Writable staged raw array with shape ``(samples, vx, vy, vz)``.
    metadata : list of dict
        Mutable output metadata rows in array order.
    sample_index : int
        First unwritten dataset row.
    timesteps : sequence of int
        Remaining timesteps in dataset order.
    extraction_n_jobs : int
        Configured joblib worker count.
    extraction_worker_count : int
        Positive worker count used for bounded timestep chunks.
    include_rotation_context : bool, optional
        Whether same-cell magnetic field and total bulk velocity are required
        for optional Hermite rotation.
    sample_callback : callable, optional
        Per-sample callback used for aligned Hermite writing.

    Returns
    -------
    int
        Next unwritten dataset row.
    """

    if extraction_n_jobs == 1 or sample_callback is not None:
        return write_timesteps_serial(
            sample_specs_by_timestep=sample_specs_by_timestep,
            X=X,
            metadata=metadata,
            sample_index=sample_index,
            timesteps=timesteps,
            include_rotation_context=include_rotation_context,
            sample_callback=sample_callback,
        )
    return write_timesteps_parallel(
        sample_specs_by_timestep=sample_specs_by_timestep,
        X=X,
        metadata=metadata,
        sample_index=sample_index,
        timesteps=timesteps,
        extraction_n_jobs=extraction_n_jobs,
        extraction_worker_count=extraction_worker_count,
    )


def write_timesteps_serial(
    sample_specs_by_timestep,
    X,
    metadata,
    sample_index,
    timesteps,
    include_rotation_context=False,
    sample_callback=None,
):
    """Stream remaining timestep samples through one process.

    The serial path opens and exhausts one timestep iterator at a time. It is
    also the path that invokes the aligned Hermite callback. When rotation is
    enabled, same-cell magnetic field and total bulk velocity remain available
    beside each raw sample.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mutable mapping from timestep to planned sample records.
    X : numpy.ndarray
        Writable staged raw array with shape ``(samples, vx, vy, vz)``.
    metadata : list of dict
        Mutable output metadata rows in array order.
    sample_index : int
        First unwritten dataset row.
    timesteps : sequence of int
        Remaining timesteps in dataset order.
    include_rotation_context : bool, optional
        Whether same-cell magnetic field and total bulk velocity are required
        for optional Hermite rotation.
    sample_callback : callable, optional
        Per-sample callback used for aligned Hermite writing.

    Returns
    -------
    int
        Next unwritten dataset row.
    """

    for timestep in timesteps:
        sample_specs = sample_specs_by_timestep.pop(int(timestep))
        timestep_samples = iter_timestep_sample_specs(
            sample_specs,
            include_rotation_context=include_rotation_context,
        )
        sample_index = write_timestep_samples(
            X=X,
            metadata=metadata,
            timestep_samples=timestep_samples,
            sample_index=sample_index,
            sample_callback=sample_callback,
        )
    return sample_index


def write_timesteps_parallel(
    sample_specs_by_timestep,
    X,
    metadata,
    sample_index,
    timesteps,
    extraction_n_jobs,
    extraction_worker_count,
):
    """Extract raw timesteps in parallel and copy them in stable order.

    Raw-only creation uses bounded timestep chunks so worker-owned arrays do
    not all reside in memory simultaneously. Joblib results are consumed in
    submission order and copied into the staged memory map without changing
    the requested timestep or within-timestep sample sequence.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Mutable mapping from timestep to planned sample records.
    X : numpy.memmap
        Writable staged raw array with shape ``(samples, vx, vy, vz)``.
    metadata : list of dict
        Mutable output metadata rows in array order.
    sample_index : int
        First unwritten dataset row.
    timesteps : sequence of int
        Remaining timesteps in dataset order.
    extraction_n_jobs : int
        Joblib worker count.
    extraction_worker_count : int
        Positive bound on concurrently materialized timestep arrays.

    Returns
    -------
    int
        Next unwritten dataset row.
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
            chunk_results = extract_timestep_chunk_parallel(
                sample_specs_by_timestep=chunk_specs_by_timestep,
                timestep_chunk=timestep_chunk,
                extraction_n_jobs=extraction_n_jobs,
                temp_dir=temp_dir,
                sample_shape=X.shape[1:],
                dtype=X.dtype,
            )
            for extracted_timestep in chunk_results:
                sample_index = write_extracted_timestep(
                    X=X,
                    metadata=metadata,
                    extracted_timestep=extracted_timestep,
                    sample_index=sample_index,
                )
    return sample_index


def extract_timestep_samples_to_temp(sample_specs, temp_dir, sample_shape, dtype):
    """Extract one timestep into a worker-owned temporary NumPy array.

    A joblib worker owns this intermediate file while the parent process owns
    the final staged dataset array. The function writes rows and metadata in
    timestep order, flushes and closes its memory map, and returns only the
    path and descriptors needed for the parent copy.

    Parameters
    ----------
    sample_specs : list of dict
        Ordered sample records for one timestep.
    temp_dir : str or pathlib.Path
        Existing staging-local directory for worker outputs.
    sample_shape : tuple of int
        Dense sample shape in ``[vx, vy, vz]`` order.
    dtype : numpy.dtype
        Raw staged-array dtype.

    Returns
    -------
    dict
        Sample count, ordered metadata, and temporary ``X.npy`` path.
    """

    if not sample_specs:
        return {"n_samples": 0, "metadata": [], "X_path": None}

    timestep = int(sample_specs[0]["timestep"])
    temp_dir = Path(temp_dir)
    X_path = temp_dir / f"timestep_{timestep}_X.npy"
    X_temp = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=dtype,
        shape=(len(sample_specs), *sample_shape),
    )
    metadata = []
    sample_index = write_timestep_samples(
        X=X_temp,
        metadata=metadata,
        timestep_samples=iter_timestep_sample_specs(sample_specs),
        sample_index=0,
    )
    X_temp.flush()
    X_temp._mmap.close()
    return {
        "n_samples": sample_index,
        "metadata": metadata,
        "X_path": str(X_path),
    }


def write_extracted_timestep(X, metadata, extracted_timestep, sample_index):
    """Copy one temporary timestep array into final staged row order.

    The parent extraction process calls this for joblib results in requested
    order. It copies the raw block, offsets each temporary metadata
    ``sample_index`` to the final dataset row, and closes the read-only
    temporary memory map before returning.

    Parameters
    ----------
    X : numpy.ndarray
        Writable staged raw dataset array.
    metadata : list of dict
        Mutable final metadata rows.
    extracted_timestep : dict
        Descriptor returned by :func:`extract_timestep_samples_to_temp`.
    sample_index : int
        First destination row for this timestep.

    Returns
    -------
    int
        Next unwritten dataset row.
    """

    n_samples = int(extracted_timestep["n_samples"])
    if n_samples == 0:
        return sample_index
    timestep_metadata = extracted_timestep["metadata"]
    X_temp = np.load(extracted_timestep["X_path"], mmap_mode="r")
    write_end = sample_index + n_samples
    X[sample_index:write_end] = X_temp[:n_samples]
    for metadata_row in timestep_metadata:
        output_row = dict(metadata_row)
        output_row["sample_index"] = sample_index + int(
            metadata_row["sample_index"]
        )
        metadata.append(output_row)
    X_temp._mmap.close()
    return write_end


def extract_timestep_chunk_parallel(
    sample_specs_by_timestep,
    timestep_chunk,
    extraction_n_jobs,
    temp_dir,
    sample_shape,
    dtype,
):
    """Submit a bounded timestep chunk to joblib.

    This function is the process-boundary owner for raw-only extraction.
    Independent workers write one temporary array per timestep, while
    joblib's ordered generator preserves the chunk sequence consumed by the
    parent writer.

    Parameters
    ----------
    sample_specs_by_timestep : dict
        Planned records keyed by timestep.
    timestep_chunk : sequence of int
        Timesteps submitted in output order.
    extraction_n_jobs : int
        Joblib worker count.
    temp_dir : str or pathlib.Path
        Staging-local temporary directory.
    sample_shape : tuple of int
        Dense ``[vx, vy, vz]`` sample shape.
    dtype : numpy.dtype
        Raw staged-array dtype.

    Returns
    -------
    iterable of dict
        Temporary timestep descriptors in submission order.
    """

    parallel = create_extraction_parallel(extraction_n_jobs)
    return parallel(
        delayed(extract_timestep_samples_to_temp)(
            sample_specs=sample_specs_by_timestep[int(timestep)],
            temp_dir=temp_dir,
            sample_shape=sample_shape,
            dtype=dtype,
        )
        for timestep in timestep_chunk
    )


def create_extraction_parallel(extraction_n_jobs):
    """Create the joblib executor used for timestep extraction.

    Keeping executor construction separate makes the ordered-generator
    behavior explicit and prevents stage 0 from depending on joblib details.

    Parameters
    ----------
    extraction_n_jobs : int
        Configured joblib worker count.

    Returns
    -------
    joblib.Parallel
        Generator-capable executor when supported by the installed joblib,
        otherwise its ordered list-returning equivalent.
    """

    return Parallel(
        n_jobs=extraction_n_jobs,
        return_as="generator",
    )


def iter_timestep_sample_specs(
    sample_specs,
    include_rotation_context=False,
):
    """Yield physical VDF samples from precomputed timestep records.

    This generator owns one Analysator reader and one cached
    :class:`VdfExtractor` for the entire timestep. Stage 4 consumes yielded
    rows immediately, which bounds memory and ensures optional magnetic field
    and total bulk velocity come from the same spatial cell as the raw VDF.

    Parameters
    ----------
    sample_specs : list of dict
        Ordered sample specifications for one source timestep.
    include_rotation_context : bool, optional
        Whether same-cell magnetic field and total bulk velocity are read for
        optional Hermite rotation.

    Yields
    ------
    dict
        Ordered sample containing a float32 ``[vx, vy, vz]`` VDF, class
        label, metadata, and optional same-cell rotation vectors.
    """

    if not sample_specs:
        return
    file_location = sample_specs[0]["file_location"]
    timestep = int(sample_specs[0].get("timestep", 0))
    velocity_grid = sample_specs[0].get("velocity_grid")
    population = (
        str(velocity_grid["population"])
        if velocity_grid is not None
        else None
    )
    extraction_start = time.perf_counter()
    reader = pt.vlsvfile.VlsvReader(str(file_location))
    resolved_source = resolve_vlsv_source(
        reader,
        population,
        require_magnetic_field=include_rotation_context,
    )
    velocity_source = (
        resolve_bulk_velocity_source(reader, resolved_source.population)
        if include_rotation_context
        else None
    )
    extractor = VdfExtractor(
        reader=reader,
        pop=resolved_source.population,
        velocity_mesh_shape=resolved_source.velocity_mesh_shape,
    )
    print(f"Timestep {timestep}: extracting {len(sample_specs)} samples")
    for sample_spec in sample_specs:
        cid = int(sample_spec["cid"])
        vdf = extractor.extract(cid=cid).astype(np.float32, copy=False)
        sample = {
            "vdf": vdf,
            "label": sample_spec["label"],
            "metadata": create_sample_metadata_row(
                sample_spec=sample_spec,
                cid=cid,
                file_location=file_location,
            ),
        }
        if include_rotation_context:
            sample["rotation_context"] = get_hermite_rotation_context(
                reader,
                cid,
                resolved_source=resolved_source,
                velocity_source=velocity_source,
            )
        yield sample
        del sample, vdf
    print(
        f"Timestep {timestep} extraction: "
        f"{time.perf_counter() - extraction_start:.2f} s"
    )


def write_timestep_samples(
    X,
    metadata,
    timestep_samples,
    sample_index,
    sample_callback=None,
):
    """Write ordered timestep samples to one staged dataset.

    This is the row-alignment boundary for current datasets. Each raw
    assignment, optional Hermite callback, and metadata append uses the same
    monotonically increasing ``sample_index`` before the next sample is read.

    Parameters
    ----------
    X : numpy.ndarray
        Writable raw array with shape ``(samples, vx, vy, vz)``.
    metadata : list of dict
        Mutable metadata rows accumulated in array order.
    timestep_samples : iterable of dict
        Ordered sample dictionaries from
        :func:`iter_timestep_sample_specs`.
    sample_index : int
        First output row to write.
    sample_callback : callable, optional
        Function called after each raw assignment for aligned Hermite
        writing.

    Returns
    -------
    int
        Next unwritten sample index.
    """

    for sample in timestep_samples:
        X[sample_index] = sample["vdf"]
        if sample_callback is not None:
            sample_callback(sample_index, sample)
        metadata_row = {"sample_index": sample_index}
        metadata_row.update(sample["metadata"])
        metadata.append(metadata_row)
        sample_index += 1
    return sample_index


def transform_extracted_sample_to_hermite(
    raw_vdf,
    velocity_grid,
    *,
    order=DEFAULT_HERMITE_ORDER,
    rotate=False,
    rotation_context=None,
    output_dtype=np.float32,
):
    """Project one aligned physical VDF into saved Hermite coefficients.

    The stage-0 callback projects the physical linear VDF with endpoint
    velocity coordinates and the supplied physical-velocity Hermite basis.
    Optional rotation first interpolates onto the expanded
    ``(parallel, perp1, perp2)`` grid using same-cell magnetic field and total
    bulk flow. Transformation occurs immediately after the aligned raw write;
    float64 coefficients are cast once to the configured storage dtype.

    Parameters
    ----------
    raw_vdf : numpy.ndarray
        Unchanged physical VDF in ``[vx, vy,vz]`` order.
    velocity_grid : dict
        Current velocity-grid descriptor with extents in metres per second.
    order : int, optional
        Physicists' Hermite coefficient count per velocity axis.
    rotate : bool, optional
        Whether to rotate the physical VDF before calculating its moments and
        coefficients.
    rotation_context : tuple of numpy.ndarray, optional
        Same-cell magnetic field in teslas and bulk velocity in metres per
        second, used only when ``rotate`` is true.
    output_dtype : numpy.dtype, optional
        Saved coefficient dtype.

    Returns
    -------
    numpy.ndarray
        Dimensional coefficient cube in ``(n_x, n_y, n_z)`` order when
        unrotated or ``(n_parallel, n_perp1, n_perp2)`` when rotated.
    """

    shape = tuple(int(value) for value in raw_vdf.shape)
    velocity_limits_mps = np.asarray(
        velocity_grid["extent_mps"],
        dtype=np.float64,
    )
    transform_vdf = raw_vdf
    if rotate:
        magnetic_field, bulk_velocity = rotation_context
        transform_vdf, shape, velocity_limits_mps, _ = rotate_vdf(
            raw_vdf,
            shape,
            velocity_limits_mps,
            magnetic_field,
            bulk_velocity,
        )
    coefficients = vdf_to_hermite(
        transform_vdf,
        shape,
        velocity_limits_mps,
        order=int(order),
    )
    return coefficients.astype(np.dtype(output_dtype), copy=False)


def iter_chunks(items, chunk_size):
    """Yield consecutive bounded slices from an ordered sequence.

    Parallel raw extraction uses these slices to cap concurrently materialized
    timestep arrays while leaving input ordering unchanged.

    Parameters
    ----------
    items : sequence
        Ordered values to divide without reordering.
    chunk_size : int
        Maximum number of values in one returned slice.

    Yields
    ------
    sequence
        Consecutive slices containing at most ``chunk_size`` values.
    """

    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]
