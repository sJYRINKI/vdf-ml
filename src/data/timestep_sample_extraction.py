"""Extract one timestep's aligned raw, plasma, and Hermite sample rows.

This focused worker-side module owns the reader lifetime and sequential
sample loop for one timestep. It opens one VLSV source, reuses one dense VDF
extractor and one resolved plasma-producer mapping, extracts every raw VDF
once, and writes raw and sixteen-value plasma rows at identical local
indexes. Optional Hermite projection consumes the same raw VDF and the same
saved magnetic-field and bulk-velocity vectors.

Parallel workers write only staging-local NumPy memory maps. They return
paths and small metadata mappings to Stage 4; only the parent process copies
those rows into final dataset arrays. This keeps large arrays out of process
IPC and preserves deterministic timestep and within-timestep ordering.
"""

from functools import partial
from pathlib import Path
import time

import analysator as pt
import numpy as np

from src.data.dense_vdf import VdfExtractor
from src.data.metadata import create_sample_metadata_row
from src.data.step_01_open_vlsv_files import resolve_vlsv_source
from src.physics.hermite_basis import DEFAULT_HERMITE_ORDER
from src.physics.hermite_rotation import rotate_vdf
from src.physics.hermite_transform import vdf_to_hermite
from src.physics.plasma_context import (
    PLASMA_CONTEXT_FEATURE_NAMES,
    create_plasma_context_row,
    prepare_plasma_context_sources_for_cells,
    read_plasma_values_for_cell,
    resolve_plasma_context_sources,
)


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
    """Extract one timestep into aligned temporary dataset arrays.

    One joblib worker owns one timestep. It opens the VLSV source once,
    reuses one dense VDF extractor and one plasma producer mapping, and
    processes planned samples sequentially. Each physical VDF is extracted
    once, written to local ``X.npy``, paired with one float32 sixteen-feature
    row in local ``plasma_context.npy``, and optionally projected into a
    Hermite cube at the same local row index. Complete B, E, and velocity
    components enter the context array in Cartesian order. B and V are reused
    directly when Hermite rotation is enabled.

    Worker-owned memory maps avoid transferring VDF or coefficient arrays
    through process IPC. Stage 4 later opens the temporary files read-only
    and copies every array through one preplanned final slice. Worker
    processes never receive or write final dataset memory maps. Optional
    rotation increases per-worker memory through one interpolated VDF and
    its coordinate arrays.

    Parameters
    ----------
    sample_specs : list of dict
        Planned samples for one timestep in stable within-timestep order.
    temp_dir : str or pathlib.Path
        Staging-local root. The worker creates one timestep subdirectory
        containing temporary ``X.npy``, ``plasma_context.npy``, and
        optional ``X_hermite.npy`` files.
    sample_shape : tuple of int
        Raw VDF shape in ``[vx, vy, vz]`` order.
    dtype : numpy.dtype
        Raw storage dtype, normally float32.
    hermite_enabled : bool, optional
        Whether the worker also writes coefficient cubes.
    velocity_grid : dict, optional
        Physical endpoint velocity-grid descriptor for Hermite projection.
    hermite_order : int, optional
        Coefficients retained along each Hermite axis.
    hermite_rotate : bool, optional
        Whether to rotate each physical VDF before Hermite projection.
    hermite_dtype : numpy.dtype, optional
        Saved coefficient dtype after float64 calculation.

    Returns
    -------
    dict
        Timestep identity, local sample count, aligned metadata rows, and
        temporary raw, plasma-context, and optional Hermite paths.

    Notes
    -----
    Timestep ownership is the parallel unit so all selected CIDs reuse one
    reader and extractor. Ordinary memory holds one active sample plus
    transient physical values; complete timestep outputs remain memory
    mapped on disk.
    """

    if not sample_specs:
        return {
            "timestep": None,
            "n_samples": 0,
            "metadata": [],
            "X_path": None,
            "plasma_context_path": None,
            "X_hermite_path": None,
        }

    timestep = int(sample_specs[0]["timestep"])
    temp_dir = Path(temp_dir) / f"timestep_{timestep}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    raw_shape = tuple(int(value) for value in sample_shape)
    velocity_limits_mps = (
        np.asarray(velocity_grid["extent_mps"], dtype=np.float64)
        if hermite_enabled
        else None
    )
    X_path = temp_dir / "X.npy"
    plasma_context_path = temp_dir / "plasma_context.npy"
    X_temp = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=dtype,
        shape=(len(sample_specs), *raw_shape),
    )
    plasma_context_temp = np.lib.format.open_memmap(
        plasma_context_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(sample_specs), len(PLASMA_CONTEXT_FEATURE_NAMES)),
    )
    X_hermite_path = None
    X_hermite_temp = None
    if hermite_enabled:
        X_hermite_path = temp_dir / "X_hermite.npy"
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
        plasma_context=plasma_context_temp,
        metadata=metadata,
        timestep_samples=iter_timestep_sample_specs(sample_specs),
        sample_index=0,
        sample_callback=sample_callback,
    )
    X_temp.flush()
    plasma_context_temp.flush()
    X_temp._mmap.close()
    plasma_context_temp._mmap.close()
    if X_hermite_temp is not None:
        X_hermite_temp.flush()
        X_hermite_temp._mmap.close()
    return {
        "timestep": timestep,
        "n_samples": sample_index,
        "metadata": metadata,
        "X_path": str(X_path),
        "plasma_context_path": str(plasma_context_path),
        "X_hermite_path": (
            str(X_hermite_path) if X_hermite_path is not None else None
        ),
    }


def iter_timestep_sample_specs(sample_specs):
    """Yield aligned VDF and plasma values from one reused timestep reader.

    One Analysator reader, dense VDF extractor, and resolved plasma source
    mapping serve every planned sample in the timestep. Samples remain in
    their planned order and are consumed immediately. Each yielded mapping
    contains the unchanged float32 ``[vx, vy, vz]`` raw VDF, a float64
    sixteen-value context row, full B/E/V values, and existing sample metadata.
    The components enter the saved row directly without redundant magnitudes.

    Parameters
    ----------
    sample_specs : list of dict
        Planned samples for one source timestep in final row order.

    Yields
    ------
    dict
        One physical VDF, vector plasma context, same-cell plasma values, class
        label, and metadata mapping for the same CID.

    Notes
    -----
    The reader and producer choices are reused instead of reopening or
    resolving the source for every CID. Optional Hermite rotation later uses
    the yielded magnetic field and bulk velocity without another source read.
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
    resolved_source = resolve_vlsv_source(reader, population)
    plasma_sources = resolve_plasma_context_sources(
        reader,
        resolved_source.population,
    )
    prepare_plasma_context_sources_for_cells(
        reader,
        plasma_sources,
        (sample_spec["cid"] for sample_spec in sample_specs),
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
        plasma_values = read_plasma_values_for_cell(
            reader,
            cid,
            plasma_sources,
        )
        sample = {
            "vdf": vdf,
            "plasma_context": create_plasma_context_row(**plasma_values),
            "plasma_values": plasma_values,
            "label": sample_spec["label"],
            "metadata": create_sample_metadata_row(
                sample_spec=sample_spec,
                cid=cid,
                file_location=file_location,
            ),
        }
        yield sample
        del sample, plasma_values, vdf
    print(
        f"Timestep {timestep} extraction: "
        f"{time.perf_counter() - extraction_start:.2f} s"
    )


def write_timestep_samples(
    X,
    plasma_context,
    metadata,
    timestep_samples,
    sample_index,
    sample_callback=None,
):
    """Write raw, context, optional Hermite, and metadata rows together.

    Each raw assignment and its sixteen-value float32 context assignment use
    the same monotonically increasing sample index. The
    optional Hermite callback runs on that same in-memory VDF and transient
    B/V values before the matching metadata row is appended.

    Parameters
    ----------
    X : numpy.ndarray
        Writable raw array with shape ``(samples, vx, vy, vz)``.
    plasma_context : numpy.ndarray
        Writable context array with shape ``(samples, 16)`` and float32
        storage.
    metadata : list of dict
        Mutable metadata rows accumulated in array order.
    timestep_samples : iterable of dict
        Ordered same-CID VDF and plasma sample mappings.
    sample_index : int
        First output row to write.
    sample_callback : callable, optional
        Aligned optional Hermite writer called after raw and context writes.

    Returns
    -------
    int
        Next unwritten row shared by every dataset output.
    """

    for sample in timestep_samples:
        X[sample_index] = sample["vdf"]
        plasma_context[sample_index] = sample["plasma_context"]
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

    Serial and worker-local extraction pass the same physical VDF already
    written to the raw row. When rotation is enabled, the magnetic-field and
    fluid-velocity vectors come from the same-cell plasma read; they are not
    reopened and match their saved Cartesian context components.

    Parameters
    ----------
    output_sample_index : int
        Destination row shared with raw, plasma, and metadata output.
    sample : dict
        Current raw VDF, transient plasma values, and metadata.
    X_hermite : numpy.ndarray
        Writable coefficient array with cubic Hermite sample axes.
    velocity_grid : dict
        Endpoint velocity grid in metres per second.
    order : int, optional
        Coefficients retained along each transform axis.
    rotate : bool, optional
        Whether to rotate the current physical VDF before projection.
    raw_shape : tuple of int, optional
        Reusable ``[vx, vy, vz]`` raw shape.
    velocity_limits_mps : numpy.ndarray, optional
        Reusable float64 velocity extent.
    output_dtype : numpy.dtype, optional
        Saved coefficient dtype after float64 calculation.

    Returns
    -------
    None
        Coefficients and existing Hermite metadata are written in place.
    """

    plasma_values = sample["plasma_values"]
    rotation_context = (
        plasma_values["magnetic_field"],
        plasma_values["bulk_velocity"],
    )
    coefficients = transform_extracted_sample_to_hermite(
        raw_vdf=sample["vdf"],
        velocity_grid=velocity_grid,
        order=order,
        rotate=rotate,
        rotation_context=rotation_context,
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

    The unchanged linear raw VDF supplies moments and coefficients with the
    current float64 mathematics. Optional rotation consumes the transient
    same-cell magnetic field and bulk velocity already read for plasma
    context. The coefficient cube is cast once to its saved dtype.

    Parameters
    ----------
    raw_vdf : numpy.ndarray
        Physical VDF in ``[vx, vy, vz]`` order.
    velocity_grid : dict
        Endpoint velocity-grid descriptor in metres per second.
    order : int, optional
        Physicists' Hermite coefficient count per axis.
    rotate : bool, optional
        Whether to rotate the VDF before moments and projection.
    rotation_context : tuple of numpy.ndarray, optional
        Same-cell magnetic field and bulk velocity vectors.
    raw_shape : tuple of int, optional
        Reusable physical VDF shape.
    velocity_limits_mps : numpy.ndarray, optional
        Reusable float64 endpoint extent.
    output_dtype : numpy.dtype, optional
        Saved coefficient dtype.

    Returns
    -------
    numpy.ndarray
        Coefficient cube in unrotated ``(n_x, n_y, n_z)`` or rotated
        ``(n_parallel, n_perp1, n_perp2)`` order.
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


__all__ = [
    "extract_timestep_samples_to_temp",
    "iter_timestep_sample_specs",
    "transform_extracted_sample_to_hermite",
    "write_extracted_sample_hermite",
    "write_timestep_samples",
]
