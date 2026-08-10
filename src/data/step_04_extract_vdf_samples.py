"""Stage 4: extract and write planned VDF samples.

This stage follows sample planning and precedes final dataset-file writing.
It reuses one VLSV reader and producer set per timestep, expands sparse VDFs
to ``[vx, vy, vz]`` cubes, and writes rows in stable sample order.

Inputs are ordered sample records and staged arrays. Outputs are aligned raw
rows, optional physical-VDF Hermite rows, and metadata records. Parallel
extraction assigns one timestep to each joblib task. A worker reuses one VLSV
reader and dense extractor while processing that timestep's samples
sequentially into worker-owned memory maps. Only the parent copies temporary
raw and Hermite rows into final staged arrays, preserving requested timestep
and within-timestep order without transferring full samples through process
IPC or permitting concurrent final-array writes.
"""

from functools import partial
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
    X_hermite=None,
    velocity_grid=None,
    hermite_order=DEFAULT_HERMITE_ORDER,
    hermite_rotate=False,
    hermite_dtype=np.float32,
):
    """Extract and write all timesteps after the first open-reader stream.

    Stage 0 calls this dispatcher after writing the first timestep used for
    shape discovery. One configured worker selects serial streaming for raw
    or paired raw/Hermite rows. More than one worker selects timestep-level
    temporary extraction for either representation, followed by a parent-only
    merge into the staged arrays. Both branches return the same next-row
    index and preserve the planned ordering.

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
        Per-sample callback used only by serial aligned Hermite writing.
    X_hermite : numpy.ndarray, optional
        Writable final staged Hermite array with shape
        ``(samples, order, order, order)``. Its presence enables paired
        worker output in the parallel path.
    velocity_grid : dict, optional
        Dataset velocity-grid descriptor used by the physical Hermite
        transform.
    hermite_order : int, optional
        Number of saved coefficients along each Hermite axis.
    hermite_rotate : bool, optional
        Whether workers rotate each physical VDF before projection.
    hermite_dtype : numpy.dtype, optional
        Storage dtype for temporary and final Hermite coefficients.

    Returns
    -------
    int
        Next unwritten dataset row.
    """

    if extraction_n_jobs == 1:
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
        X_hermite=X_hermite,
        velocity_grid=velocity_grid,
        hermite_order=hermite_order,
        hermite_rotate=hermite_rotate,
        hermite_dtype=hermite_dtype,
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
    X_hermite=None,
    velocity_grid=None,
    hermite_order=DEFAULT_HERMITE_ORDER,
    hermite_rotate=False,
    hermite_dtype=np.float32,
):
    """Extract timesteps in parallel and merge aligned rows in the parent.

    Raw-only and raw-plus-Hermite creation use bounded timestep chunks. Each
    worker processes one timestep sequentially into local memory-mapped
    arrays, while the parent consumes joblib results in submission order and
    copies both representations through the same final slice. Workers may
    finish out of order, but they never receive the writable final sinks;
    ordered result delivery and the parent's monotonically increasing row
    cursor preserve requested timestep, sample, and metadata order.

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
    X_hermite : numpy.memmap, optional
        Parent-owned final Hermite array with shape
        ``(samples, order, order, order)``.
    velocity_grid : dict, optional
        Velocity-grid descriptor shared by all physical Hermite projections.
    hermite_order : int, optional
        Number of Hermite coefficients per transform axis.
    hermite_rotate : bool, optional
        Whether worker-local VDFs are rotated before projection. Rotation
        adds an interpolated VDF and grid coordinates to per-worker memory.
    hermite_dtype : numpy.dtype, optional
        Storage dtype of worker-local and final coefficient arrays.

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
                hermite_enabled=X_hermite is not None,
                velocity_grid=velocity_grid,
                hermite_order=hermite_order,
                hermite_rotate=hermite_rotate,
                hermite_dtype=hermite_dtype,
            )
            for extracted_timestep in chunk_results:
                sample_index = write_extracted_timestep(
                    X=X,
                    X_hermite=X_hermite,
                    metadata=metadata,
                    extracted_timestep=extracted_timestep,
                    sample_index=sample_index,
                )
    return sample_index


def extract_timestep_samples_to_temp(
    sample_specs,
    temp_dir,
    sample_shape,
    dtype,
    hermite_enabled=False,
    velocity_grid=None,
    hermite_order=DEFAULT_HERMITE_ORDER,
    hermite_rotate=False,
    hermite_dtype=np.float32,
):
    """Extract one timestep into aligned temporary raw and Hermite arrays.

    One joblib worker opens the timestep's VLSV source and reuses its dense
    VDF extractor for every planned sample. Samples remain sequential inside
    the worker: each physical VDF is extracted once, written to the local raw
    memory map, and optionally transformed into a Hermite coefficient cube at
    the identical local row index. Rotated Hermite extraction also holds one
    interpolated VDF and its coordinate grid while processing that sample.

    Worker-owned temporary arrays keep large VDF data out of the process
    result channel. The worker flushes and closes the arrays before returning
    their paths and small metadata records. The parent later copies both
    arrays into the next ordered final range; worker processes never receive
    or write the final dataset memory maps.

    Parameters
    ----------
    sample_specs : list of dict
        Ordered sample records for one timestep.
    temp_dir : str or pathlib.Path
        Existing staging-local directory receiving
        ``timestep_<timestep>_X.npy`` and, when enabled,
        ``timestep_<timestep>_X_hermite.npy``.
    sample_shape : tuple of int
        Dense sample shape in ``[vx, vy, vz]`` order. The raw temporary
        array has shape ``(n_samples, *sample_shape)``.
    dtype : numpy.dtype
        Raw temporary-array dtype, normally ``float32``.
    hermite_enabled : bool, optional
        Whether to write a sibling coefficient array with shape
        ``(n_samples, hermite_order, hermite_order, hermite_order)``.
    velocity_grid : dict, optional
        Velocity-grid descriptor used for physical-VDF moments, basis
        coordinates, optional rotation, and coefficient projection.
    hermite_order : int, optional
        Number of saved coefficients along each transform axis.
    hermite_rotate : bool, optional
        Whether to rotate each physical VDF before its Hermite transform.
    hermite_dtype : numpy.dtype, optional
        Temporary coefficient storage dtype, normally ``float32``.
        Calculation remains float64 in the physical transform and is cast
        once for storage.

    Returns
    -------
    dict
        Timestep sample count, ordered metadata, temporary raw path, and the
        optional temporary Hermite path consumed by the parent merge.

    Notes
    -----
    The timestep is the parallel work unit because all of its samples share
    one VLSV reader and dense extractor. The memory-mapped output arrays may
    cover the whole timestep on disk, but ordinary memory remains bounded to
    the active raw VDF, optional rotated VDF, and coefficient cube.
    """

    if not sample_specs:
        return {
            "timestep": None,
            "n_samples": 0,
            "metadata": [],
            "X_path": None,
            "X_hermite_path": None,
        }

    timestep = int(sample_specs[0]["timestep"])
    temp_dir = Path(temp_dir)
    raw_shape = tuple(int(value) for value in sample_shape)
    velocity_limits_mps = (
        np.asarray(velocity_grid["extent_mps"], dtype=np.float64)
        if hermite_enabled
        else None
    )
    X_path = temp_dir / f"timestep_{timestep}_X.npy"
    X_temp = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=dtype,
        shape=(len(sample_specs), *raw_shape),
    )
    X_hermite_path = None
    X_hermite_temp = None
    if hermite_enabled:
        X_hermite_path = temp_dir / f"timestep_{timestep}_X_hermite.npy"
        X_hermite_temp = np.lib.format.open_memmap(
            X_hermite_path,
            mode="w+",
            dtype=hermite_dtype,
            shape=(
                len(sample_specs),
                hermite_order,
                hermite_order,
                hermite_order,
            ),
        )

    metadata = []
    sample_callback = (
        partial(
            write_extracted_sample_hermite,
            X_hermite=X_hermite_temp,
            velocity_grid=velocity_grid,
            order=hermite_order,
            rotate=hermite_rotate,
            raw_shape=raw_shape,
            velocity_limits_mps=velocity_limits_mps,
            output_dtype=hermite_dtype,
        )
        if hermite_enabled
        else None
    )
    sample_index = write_timestep_samples(
        X=X_temp,
        metadata=metadata,
        timestep_samples=iter_timestep_sample_specs(
            sample_specs,
            include_rotation_context=hermite_enabled and hermite_rotate,
        ),
        sample_index=0,
        sample_callback=sample_callback,
    )
    X_temp.flush()
    X_temp._mmap.close()
    if X_hermite_temp is not None:
        X_hermite_temp.flush()
        X_hermite_temp._mmap.close()
    return {
        "timestep": timestep,
        "n_samples": sample_index,
        "metadata": metadata,
        "X_path": str(X_path),
        "X_hermite_path": (
            str(X_hermite_path) if X_hermite_path is not None else None
        ),
    }


def write_extracted_timestep(
    X,
    metadata,
    extracted_timestep,
    sample_index,
    X_hermite=None,
):
    """Merge one temporary timestep result into aligned final array rows.

    Timestep workers may complete out of order, while joblib exposes their
    results to this parent writer in submission order. The current parent row
    cursor therefore defines the final planned offset. Raw and optional
    Hermite memory maps are opened read-only and copied through the identical
    destination slice; metadata indexes are placed at that same offset. Only
    this parent process writes final staged dataset arrays.

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
    X_hermite : numpy.ndarray, optional
        Writable parent-owned Hermite array aligned with ``X``.

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
    if X_hermite is not None:
        X_hermite_temp = np.load(
            extracted_timestep["X_hermite_path"],
            mmap_mode="r",
        )
        X_hermite[sample_index:write_end] = X_hermite_temp[:n_samples]
        X_hermite_temp._mmap.close()
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
    hermite_enabled=False,
    velocity_grid=None,
    hermite_order=DEFAULT_HERMITE_ORDER,
    hermite_rotate=False,
    hermite_dtype=np.float32,
):
    """Submit a bounded timestep chunk to joblib.

    This function is the only process boundary for raw-only and paired
    raw/Hermite extraction. Independent workers receive plain configuration
    values and write one aligned temporary result per timestep. They never
    receive final writable sinks. Joblib's ordered generator preserves the
    chunk sequence consumed by the parent even when work finishes in another
    order.

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
    hermite_enabled : bool, optional
        Whether every worker writes aligned Hermite coefficients.
    velocity_grid : dict, optional
        Shared physical velocity-grid descriptor.
    hermite_order : int, optional
        Coefficient count along each Hermite axis.
    hermite_rotate : bool, optional
        Whether workers perform optional velocity-space rotation.
    hermite_dtype : numpy.dtype, optional
        Temporary Hermite storage dtype.

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
            hermite_enabled=hermite_enabled,
            velocity_grid=velocity_grid,
            hermite_order=hermite_order,
            hermite_rotate=hermite_rotate,
            hermite_dtype=hermite_dtype,
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
        Ordered generator executor for timestep task results.
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


def write_extracted_sample_hermite(
    output_sample_index,
    sample,
    *,
    X_hermite,
    velocity_grid,
    order=DEFAULT_HERMITE_ORDER,
    rotate=False,
    raw_shape=None,
    velocity_limits_mps=None,
    output_dtype=np.float32,
):
    """Transform one extracted VDF and write its aligned Hermite row.

    Serial extraction uses this operation with the parent-owned final
    ``X_hermite.npy`` memory map. A parallel timestep worker uses it with the
    worker-owned temporary ``X_hermite.npy`` map and supplies the raw shape
    and endpoint velocity limits determined once for that timestep. In both
    cases the input is the same in-memory physical VDF already written to the
    raw row, and the coefficient cube and Hermite metadata use the identical
    sample index.

    Parameters
    ----------
    output_sample_index : int
        Serial final or worker-local destination row shared with raw output.
    sample : dict
        Current physical VDF, metadata, and optional rotation context.
    X_hermite : numpy.ndarray
        Writable final or temporary coefficient array with shape
        ``(samples, order, order, order)``.
    velocity_grid : dict
        Physical velocity-grid descriptor with endpoint limits in metres per
        second.
    order : int, optional
        Number of coefficients retained along every transform axis.
    rotate : bool, optional
        Whether to rotate this physical VDF before projection.
    raw_shape : tuple of int, optional
        Reusable physical VDF shape in ``[vx, vy, vz]`` order.
    velocity_limits_mps : numpy.ndarray, optional
        Reusable float64 endpoint extent for one timestep.
    output_dtype : numpy.dtype, optional
        Saved coefficient dtype, normally ``float32`` after float64
        calculation.

    Returns
    -------
    None
        The coefficient row and its Hermite metadata fields are written in
        place.
    """

    coefficients = transform_extracted_sample_to_hermite(
        raw_vdf=sample["vdf"],
        velocity_grid=velocity_grid,
        order=order,
        rotate=rotate,
        rotation_context=sample.get("rotation_context"),
        raw_shape=raw_shape,
        velocity_limits_mps=velocity_limits_mps,
        output_dtype=output_dtype,
    )
    X_hermite[output_sample_index] = coefficients
    sample["metadata"]["hermite_order"] = order
    sample["metadata"]["hermite_rotate"] = rotate
    del coefficients


def transform_extracted_sample_to_hermite(
    raw_vdf,
    velocity_grid,
    *,
    order=DEFAULT_HERMITE_ORDER,
    rotate=False,
    rotation_context=None,
    raw_shape=None,
    velocity_limits_mps=None,
    output_dtype=np.float32,
):
    """Project one aligned physical VDF into saved Hermite coefficients.

    Serial extraction and timestep workers both call this operation on the
    same physical linear VDF written to the aligned raw row. Endpoint
    velocity coordinates and the supplied physical-velocity Hermite basis
    define the projection. Optional rotation first interpolates onto the
    expanded ``(parallel, perp1, perp2)`` grid using same-cell magnetic field
    and total bulk flow. Transformation occurs immediately after the aligned
    raw write; float64 coefficients are cast once to the configured storage
    dtype.

    Parameters
    ----------
    raw_vdf : numpy.ndarray
        Unchanged physical VDF in ``[vx, vy, vz]`` order.
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
    raw_shape : tuple of int, optional
        Reusable physical VDF shape in ``[vx, vy, vz]`` order. Serial callers
        derive it from ``raw_vdf``; a timestep worker supplies the shape
        determined once before processing its samples.
    velocity_limits_mps : numpy.ndarray, optional
        Reusable endpoint extent in metres per second. Serial callers derive
        it from ``velocity_grid``; a timestep worker supplies one float64
        array shared by every sample in that timestep.
    output_dtype : numpy.dtype, optional
        Saved coefficient dtype.

    Returns
    -------
    numpy.ndarray
        Dimensional coefficient cube in ``(n_x, n_y, n_z)`` order when
        unrotated or ``(n_parallel, n_perp1, n_perp2)`` when rotated.
    """

    if raw_shape is None:
        raw_shape = tuple(int(value) for value in raw_vdf.shape)
    if velocity_limits_mps is None:
        velocity_limits_mps = np.asarray(
            velocity_grid["extent_mps"],
            dtype=np.float64,
        )
    transform_vdf = raw_vdf
    if rotate:
        magnetic_field, bulk_velocity = rotation_context
        transform_vdf, transform_shape, transform_limits_mps, _ = rotate_vdf(
            raw_vdf,
            raw_shape,
            velocity_limits_mps,
            magnetic_field,
            bulk_velocity,
        )
    else:
        transform_shape = raw_shape
        transform_limits_mps = velocity_limits_mps
    coefficients = vdf_to_hermite(
        transform_vdf,
        transform_shape,
        transform_limits_mps,
        order=int(order),
    )
    return coefficients.astype(np.dtype(output_dtype), copy=False)


def iter_chunks(items, chunk_size):
    """Yield consecutive bounded slices from an ordered sequence.

    Parallel raw or paired raw/Hermite extraction uses these slices to cap
    concurrently materialized timestep arrays while leaving input ordering
    unchanged.

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
