# Data, physics, and representations

## Scientific inputs

Extraction uses:

- a Vlasiator VLSV bulk file for each requested timestep;
- the matching flux-function file used by X/O detection;
- configured physical regions and class coordinates; and
- the VDF population and velocity mesh exposed by the source reader.

Source templates in `configs/data/extraction.yaml` are site-specific and
must be reviewed before use. A simulation timestep is an index, not
automatically a time in seconds.

## VLSV input boundary

Producers and the selected population's velocity mesh are resolved once per
opened VLSV file, not once per sample.

The velocity mesh must have exactly three positive dimensions. Spatial maps
prefer stored three-component bulk velocity in this order:
`<population>/vg_v`, `<population>/V`, `vg_v`, then `V`. Older files without
one of those producers use the matched legacy pair `rho_v / rho`. Thermal,
nonthermal, backstream, and drift velocities are not substituted for total
bulk flow.

Unrotated Hermite generation reads only the physical VDF and velocity grid.
When `representations.hermite.rotate` is true, the same source must also
provide the current cell-centred magnetic-field and total bulk-velocity
producers used to construct the optional rotation frame. `MinValue` is never
an input to the physical-VDF Hermite transform.

The workflow reads the configured producers directly and lets Analysator
propagate its ordinary exception if an expected variable is unavailable.
Physical VDF plotting resolves the threshold producer in Analysator-era order:
`MinValue`, `<population>/MinValue`,
`<population>/EffectiveSparsityThreshold`, then
`<population>/vg_effectivesparsitythreshold`. Prediction retains that
selection once per open source and reads its value from the same VLSV cell as
every displayed VDF. Raw-only extraction with VDF plotting disabled and
unrotated Hermite generation do not read a magnetic field or sparsity
threshold. Plotting's threshold read does not make that quantity a raw or
Hermite representation input.

## VDF cells and velocity grids

The source reader identifies cells that carry a VDF for the selected
population. Cell IDs are kept in stable order. Requested point coordinates
are mapped to actual VDF-carrying cells; metadata records the selected cell
ID and selected cell centre.

The saved velocity-grid descriptor contains:

- three positive dimensions;
- six ordered extents in m/s;
- population;
- axis order `[vx, vy, vz]`;
- slice axis `vy`; and
- the middle slice index.

Dense raw arrays use `[vx, vy, vz]`. Mesh dimensions are inferred and are
not hardcoded.

## Physical labels

The configured class mapping is:

| `class_id` | `class_name` |
|---:|---|
| 0 | `lobe` |
| 1 | `current_layer` |
| 2 | `o_point` |
| 3 | `x_point` |
| 4 | `solar_wind` |
| 5 | `magnetosheath` |
| 6 | `inner_magnetosphere` |

`metadata.csv` owns `class_id` and `class_name`.

The extraction flow:

1. discovers VDF-carrying cells;
2. detects physical X/O point records from the flux grid;
3. maps configured manual-class coordinates to their nearest VDF cells;
4. calculates manual and physical VDF-cell sets for each detected X/O point;
5. applies `manual`, `physical`, `consensus`, or `union` independently;
6. omits empty point results and ejects complete X/O source records whenever
   their final selected-cell sets intersect;
7. removes remaining contradictory class assignments and keeps their CIDs
   rejected;
8. appends still-unassigned magnetotail candidates as `current_layer` in
   ascending CID order; and
9. writes the actual selected CID and cell centre.

Manual classes are iterated in configuration order before X/O records.
Repeated selections of the same `(class_name, cid)` keep their first record;
a CID assigned to different classes loses every conflicting record and is
not refilled by the magnetotail fallback. This retains the established
deduplication, conflict, and sample-order behavior.

The four point-selection policies have direct set meanings: `manual` uses
only the manual box, `physical` uses only the physical geometry,
`consensus` uses their intersection, and `union` uses their union. There is
no automatic fallback for an empty result. Selection preserves stable VDF
order; consensus disagreement cells remain rejected from background
assignment. Detected source positions, configured request positions, and
actual selected VDF-cell positions are distinct and must not be substituted.

This set calculation is unchanged by colormap rendering. For union selection,
a shared manual/physical VDF cell suppresses only the manual search-area
outline; every union-selected cell, including manual-only cells, remains
selected and plotted. Disjoint union sets display both areas. Manual,
physical, and consensus displays are unchanged.

After final per-point selection, an X and O record sharing any selected CID
are both rejected as complete records, including their otherwise unique
cells. Connected conflicts reject every involved record. Same-kind records
retain historical source order and first-sample ownership of a repeated CID.
No priority rule or X/O overlap class is used.

### X/O detection

The flux grid is smoothed, and zero contours of its x and z gradients are
intersected. Each interior intersection is classified from the local 2-D
flux Hessian:

- negative determinant produces an X-point candidate;
- positive determinant with negative `Hxx` produces an O-point candidate.

Candidates outside configured point regions are excluded. For physical X
selection, the local ion inertial length and Hessian eigenvectors define the
selection box. For physical O selection, the workflow searches for the
smallest supported closed flux contour linked to an X-point. The configured
selection policy controls how physical and manual candidates are mapped to
VDF cells.

### Manual plasma regions and current layer

`lobe`, `solar_wind`, `magnetosheath`, and `inner_magnetosphere` are manual
classes. Their `class_coords_re` entries use
`[x_re, y_re, z_re]` in Earth radii. Each coordinate selects the nearest
VDF-carrying cell through the same generic selection operation. An empty
class list contributes no samples, so the checked-in empty lists do not
invent plasma-region locations.

`current_layer` has no coordinate list. It is the historical magnetotail
background assignment for VDF-cell centres in the inclusive region:

```text
-30 <= x_re <= -10
|z_re| <= 1
y unrestricted
```

Stage 3 applies this fallback only after manual classes, final X/O selection,
empty-point omission, and complete cross-kind conflict ejection. Already
assigned and rejected CIDs are excluded. The fallback is purely spatial and
does not use the current-density peak-core detector.

The additional positive-x region in `points.regions_re` remains only an X/O
candidate bound. It does not create a plasma-region class. Changing manual
coordinates, the magnetotail bounds, or the X/O regions changes the
scientific sample definition and requires review.

Comments in `configs/data/extraction.yaml` are the quickest key-by-key
reference for coordinate units, point-selection options, and optional
outputs.

## Point topology

Topology schema fixes this order:

1. `distance_to_x_point_re`
2. `distance_to_o_point_re`
3. `vdf_to_x_point_dx_re`
4. `vdf_to_x_point_dz_re`
5. `vdf_to_o_point_dx_re`
6. `vdf_to_o_point_dz_re`

All values use Earth radii. Direction components follow:

```text
reference point - selected VDF-cell centre
```

Availability fields and an aligned Boolean mask describe valid targets.
Missing topology remains missing; it is not physical zero. Optional `dy`
metadata is not part of the six-target ML schema.

## Physical context for Hermite

Hermite moments and coefficients use the physical linear VDF exactly as
supplied. The drift is calculated from its first moment; it is not read from
fluid `rho_v / rho` or a stored velocity producer. The transform applies no
sparsity threshold, logarithm, negative-value cleanup, or density
renormalization.

The original `[vx, vy, vz]` frame is the default and requires no magnetic
field or fluid velocity. Optional rotation reads the same selected CID's
current magnetic-field and total-bulk-velocity producers. Its orthonormal
rows are:

```text
parallel = B / |B|
perp1 = normalize(V - dot(V, parallel) * parallel)
perp2 = cross(parallel, perp1)
```

The rotation maps `v_rotated = R @ v_original`. It transforms all eight
velocity-cuboid corners to form new axis-aligned bounds and uses
`RegularGridInterpolator` with linear interpolation and zero outside the
original domain. It does not force the result back to the original bounds,
renormalize density, or apply a density tolerance.

## Dataset files

A raw dataset contains:

```text
X.npy
metadata.csv
velocity_grid.npz
```

When Hermite is enabled it also contains:

```text
X_hermite.npy
```

`X.npy` stores complete physical VDFs without slicing, logarithms,
normalization, rotation, or downsampling. `X_hermite.npy` stores aligned
coefficient volumes. `metadata.csv` stores sample identity, provenance,
physical class, selected-cell coordinates, and topology. It is the sole
classification-target source for current dataset consumers.

Current workflows assume these saved files are correctly formed. They load
only the arrays, metadata, and velocity-grid values needed for their
operation. A missing requested file produces a direct error.

Optional extraction postprocessing may add:

```text
plots/
├── colormaps/
│   └── frame_000000_timestep_0003072.png
├── vdfs/
│   └── cid_000001234/
│       └── frame_000000_timestep_0003072.png
└── hermite/
    └── cid_000001234/
        └── frame_000000_timestep_0003072.png

animations/
├── colormap_evolution.mp4
└── vdfs/
    └── cid_000001234.mp4
```

These are derived visual outputs. The core dataset files and metadata
columns remain unchanged.

## Transactional writing

The writer creates one hidden sibling staging directory and allocates final
`X.npy` and optional `X_hermite.npy` memory maps. The parent process streams
the first nonempty timestep while discovering the raw VDF shape. With
`extraction_n_jobs: 1`, it continues serially. With more than one extraction
job, each remaining timestep is one Joblib task for raw-only or paired
raw-plus-Hermite output. One worker opens the timestep source once, reuses
one VDF extractor, and processes its planned samples sequentially. Each raw
VDF is extracted once and its raw and Hermite rows are written at the same
local index in worker-owned memory maps beneath a staging-local temporary
directory, so large arrays do not travel through process IPC.

Workers never receive the final staged memory maps. The parent consumes
timestep descriptors in submission order and copies raw and optional
Hermite blocks into the same next final slice, placing metadata at the
corresponding indexes. Planned timestep order followed by within-timestep
sample order is therefore stable even when workers complete out of order.
The parent then writes `metadata.csv` and `velocity_grid.npz`, flushes and
closes the final maps, and renames the staging directory to the final path.
This execution-only change leaves raw-only numerical output and Hermite
coefficient values unchanged.

The writer does not reopen or scan the complete staged dataset before the
final rename. It uses only the staging directory, direct destination checks,
and ordinary rename.

## Raw representation: `raw`

`X.npy` stores:

```text
shape: (n_samples, vx, vy, vz)
axis order: [vx, vy, vz]
```

The raw PCA, CNN, and autoencoder transformation is:

```text
X.npy sample [vx, vy, vz]
    -> clamp every voxel to a minimum of log_eps
    -> log10
    -> complete [vx, vy, vz] volume
```

Values above the floor retain their values, the source sample is not
mutated, and the complete volume remains in saved axis order. PCA
flattens all `vx * vy * vz` values in stable C order. CNN training,
autoencoder training, and raw prediction add one channel to produce
`(batch, 1, vx, vy, vz)` and use `Conv3d`. These raw model-input paths perform
no velocity-plane slicing, projection, cropping, averaging, or downsampling.
Peak-centred `vx-vy`, `vx-vz`, and `vy-vz` planes remain visualization
products only; no plotted plane is a model input or reconstruction target.

## Hermite representation: `hermite`

`X_hermite.npy` stores:

```text
shape: (n_samples, order, order, order)
unrotated coefficient axes: [n_x, n_y, n_z]
rotated coefficient axes: [n_parallel, n_perp1, n_perp2]
default order: 22
saved dtype: float32
```

The complete signed coefficient cube is used by PCA, CNN, and autoencoder
workflows. CNN and autoencoder batches have shape `(batch, 1, n1, n2, n3)`
and use `Conv3d`. The cube is not sliced, truncated, downsampled, transposed,
converted to absolute values, logged, normalized per sample, or augmented
with metadata. Consumers derive the complete volume shape and order from the
saved array; new model checkpoints record that actual shape, order, and
rotation setting.

### PCA memory access

PCA keeps `X.npy` or `X_hermite.npy` as a read-only memory map and prepares
at most `pca.batch_size` samples at once. The shipped batch size is 16 and
`--batch-size` overrides it. The representation is read in three stable
passes: float64 Chan/Welford mean/M2 accumulation, incremental component
updates on standardized batches, and projection onto the final components.
Raw batches still include every `vx * vy * vz` cell and Hermite batches still
include every coefficient.

Smaller batches reduce the batched working set but
do not remove the memory required for one complete sample, the float64
statistics, the retained basis, or the decomposition workspace.
Full PCA uses exact reduced SVD for each update, although retained-rank
truncation makes a multi-batch fit incremental rather than globally
monolithic-exact. Low-rank PCA seeds its randomized incremental pass once.

The final scores remain in memory for downstream diagnostics and figures.
KMeans, nearest-neighbor search, and t-SNE receive only the shared leading
32 score columns by default and remain label blind. The physical class count
sets only the number of internal KMeans partitions; class names are applied
after fitting to report KMeans purity and concentration, local-neighbor
agreement, sample counts, and the two physical-class plots. The physical
class is the only color, marker, and legend category in those plots; KMeans
partition IDs are neither plotted nor saved. With the default 25-neighbor
setting, local purity uses training PCA scores as the reference for every
explicit split and excludes self only for training queries. An unsplit
dataset uses all samples as the reference and excludes every query itself.
The PCA command writes only
`pca_physical_classes.png`, `tsne_physical_classes.png`, and `metrics.txt`;
it does not persist score, KMeans-label, or t-SNE-coordinate arrays.

For each input axis `a`, the transform coordinates include both configured
endpoints:

```text
v_a = np.linspace(vmin_a, vmax_a, shape[a])
x_a = (v_a - drift[a]) / thermal_velocity
```

The Gaussian-weighted physicists' polynomials follow
`hp[0] = exp(-x**2 / 2)`, `hp[1] = 2*x*exp(-x**2 / 2)`, and
`hp[n] = 2*x*hp[n-1] - 2*(n-1)*hp[n-2]`. Mode `n` is divided by:

```text
sqrt(2**n * n! * sqrt(pi) * thermal_velocity)
```

The physical-VDF drift uses endpoint axes and the product of the three
`(vmax - vmin) / n_cells` spacings. The isotropic thermal standard deviation
uses `dv**3`, where `dv = (vxmax - vxmin) / nx`, for density and diagonal
pressure moments. The coefficient projection is:

```text
spectra = np.einsum(
    "ijk,ni,mj,lk->nml",
    physical_vdf,
    hermite_x,
    hermite_y,
    hermite_z,
    optimize=True,
)
coefficients = spectra * dv**3
```

Basis construction and coefficient accumulation use float64. Extraction
casts once to the configured saved dtype. Coefficients remain dimensional;
there is no logarithm, `MinValue` threshold, clipping, density scaling, or
post-projection normalization. Exactly zero density produces a zero drift
and zero coefficient cube, matching the supplied numerical convention.

Hermite datasets and Hermite-trained CNN or autoencoder checkpoints created
before this physical-VDF convention must be regenerated and retrained. No
old/new dispatch, migration, or compatibility adaptation is active.

Neither current representation includes labels, topology, spatial
coordinates, magnetic field, fluid velocity, or physical moments as
features.

## Extraction configuration

`configs/data/extraction.yaml` is owned by:

```text
python -m scripts.data.extract_dataset
```

Important sections are:

| Section | Purpose |
|---|---|
| `file_template_bulk` | VLSV path template using `{timestep}` |
| `file_template_flux` | matching flux-function path template |
| `output_dirs` | base destinations selected by `--dataset-kind` |
| `labels` | one-to-one physical class names and IDs |
| `creation` | planning and extraction job counts |
| `representations` | raw enablement and optional Hermite settings |
| `postprocessing` | optional dataset frames and animations |
| `points` | regions, point-selection policy, and physical parameters |
| `class_coords_re` | configured physical sample coordinates in `R_E` |

Raw output is required. Hermite defaults to order 22, optional rotation
defaults to false, and coefficients default to float32 storage after the
float64 calculation. The configured order controls the saved shape. There is
no density-error tolerance or raw downsampling option.

The supplied source paths and parallel job counts target an HPC
environment. Review I/O, memory, paths, region bounds, and selection methods
before a production run. Command-line X/O selection overrides change only
the corresponding configured method.

The extraction postprocessing settings are:

```yaml
postprocessing:
  plot_colormaps: false
  plot_vdfs: false
  plot_hermites: false
  animate_colormaps: false
  animate_vdfs: false
  animation_fps: 4
  animation_format: mp4
```

The fragment above disables all five operations. `plot_colormaps` writes one
spatial frame for each requested extraction timestep. `plot_vdfs` writes
one raw-VDF frame for every saved sample, grouped by spatial cell ID and
sorted by timestep. When Hermite extraction is enabled, `plot_hermites`
memory-maps `X_hermite.npy` and writes the signed `coefficients[:, 0, :]`
plane for every sample under `plots/hermite/cid_<cid>/`, ordered by timestep
and sample index. Unrotated axes are `(n_x, n_z)` with `n_y=0`; rotated axes
are `(n_parallel, n_perp2)` with `n_perp1=0`. Frames use symmetric `RdBu_r`
limits and are not animated.
`animate_colormaps` encodes the colormap sequence, and `animate_vdfs`
encodes one sequence for each cell ID represented at more than one
timestep. Enable the corresponding `plot_*` option in the same run when an
`animate_*` option is enabled, because animation consumes the Stage 6 PNG
directories directly. VDF frames use Matplotlib, colormap frames also use
Analysator, and only the animation options require ffmpeg.

## Data commands

See [Example commands](EXAMPLE_COMMANDS.md) for complete local and Slurm
invocations of extraction, colormap plotting, and VDF-slice plotting.

Extraction selects the configured output base by dataset kind and names the
dataset directory from the requested timestep range. After the six core
creation stages have completed, optional Stage 6 renders frames and optional
Stage 7 animates those frames. Both stages run inside the existing
extraction command; they do not add a command or Slurm job.

## Extraction-time colormap frames

Stage 6 uses x-directed physical bulk velocity in m/s as its fixed spatial
background. Direct stored velocity is preferred; legacy sources use
`rho_v / rho`. The `bwr` red-white-blue colormap is centred at zero with limits
of `-1.5e6` and `+1.5e6 m/s` for every timestep, so velocity direction and
magnitude have the same visual meaning throughout an animation. Thin black
magnetic-field streamlines with small arrowheads remain visible over the
background. The axes are x and z in Earth radii. The default spatial window
extends from `-30` to `+30` Earth radii along x so the configured solar-wind
source at `x = 25` Earth radii remains visible, and the title records the
simulation time or timestep.

Point colors describe their role in sample selection rather than their
physical-class palette:

- every VDF-carrying source cell is a small yellow point;
- configured manual coordinates and detected X/O source coordinates are
  blue, with marker shape distinguishing the current class or point kind;
- every selected dataset VDF cell is red, with the same marker shape as its
  blue source coordinate.

Only accepted source records are drawn. Empty point selections and complete
X/O records ejected by a shared selected-cell conflict contribute no source
marker, selected-cell marker, search area, or legend entry. The
`current_layer` class has only a red selected-cell marker because it has no
configured source coordinate. Point markers remain intentionally small so the
velocity background and selection geometry stay visible.

The compact multi-column legend is placed below the spatial axes and contains
only artists present in that frame. Selection-area artists reuse the exact
current geometry and are drawn once per source point:

- the physical X polygon uses the same local ion-inertial-length half-widths
  and Hessian eigenvectors as sample selection;
- the manual X box uses the configured Earth-radius half-widths when that
  geometry exists;
- the physical O area uses the actual selected closed flux contour; and
- the manual O box uses the configured Earth-radius half-widths.

Area visibility follows the selection method. For union results with any
shared manual/physical VDF cell, only the physical area is drawn; when the
sets are disjoint, every nonempty contributing area is drawn. Selected-cell
markers and metadata do not change.

No plotting-only approximation changes or replaces the selection geometry.
Frame names carry both a contiguous zero-padded frame index and a zero-padded
timestep, so non-contiguous simulation steps remain sortable.

## Extraction-time VDF frames and animations

VDF values are read from the completed physical `X.npy`, not re-extracted
from VLSV; only the plotting threshold is read from the same CID in the
source VLSV. Each `cid_<cell-id>/` directory contains its samples in ascending
timestep order. Every Stage 6 frame contains one square `vx-vz` panel fixed
at that sample's own three-dimensional peak `vy` index.

Every physical VDF panel follows the historical display preparation. With
the saved velocity-grid bounds and size,

```text
dv = (vymax - vymin) / nvy
```

The selected plane is copied as float32 and multiplied by `dv`. Values
strictly below the same-cell `threshold * dv` are set to zero; values equal
to that threshold are retained, and all nonpositive values are then masked. The
renderer leaves `nipy_spectral` unmodified and applies `LogNorm` to the
remaining visible support. Matplotlib's default bad mask is therefore
transparent over the white axes rather than forced black. Velocities are
shown in km/s, a grey Cartesian grid remains visible, and the vertical
colorbar is labelled `f(v)`. This display-only copy does not change `X.npy`
or any raw, Hermite, PCA, CNN, or autoencoder input. Titles identify the
timestep, cell ID, and physical class on one line. Figure dimensions and
visual limits remain fixed within each cell-ID animation.

A cell ID with more than one PNG can produce one time-evolution animation; a
one-frame cell retains only its static PNG.
Colormap animation keeps the Stage 6 timestep order. VDF animations are
organized only by cell ID, never by physical class. The configured frame
rate controls video playback rather than implying physical time between
simulation steps.

The standalone VDF command reuses the Stage 6 physical-plane extractor and
renderer. It can draw any requested combination of the peak-centred `xz`,
`xy`, and `yz` planes described below; Stage 6 remains fixed to `xz`.

## Dataset colormap

The command loads current metadata directly and calls the same renderer as
extraction Stage 6. It resolves the source VLSV from an explicit override,
`metadata.file_location`, or the extraction configuration, in that order.
The fixed background is x-directed bulk velocity with a symmetric
red-white-blue scale and black magnetic streamlines. Yellow marks all
VDF-carrying cells, blue marks configured or detected source coordinates,
and red marks selected dataset cells; marker shape identifies the current
class or point kind. Exact current X/O geometry, the union area-visibility
rule, and the compact below-axes legend are shared with Stage 6.

Useful overrides include `--timestep`, `--source-vlsv`,
`--spatial-box-re`, `--spatial-slice-y-re`,
`--show-all-vdf-cells`, `--show-labeled-cells`,
`--show-class-legend`, `--show-magnetic-streamlines`, and their Boolean
`--no-...` forms. The defaults reproduce the Stage 6 figure. The command
does not expose an alternate background variable. Because exact X/O contour
and Hessian geometry is not stored completely in dataset metadata, the
standalone command reconstructs accepted point records through the current
detection and selection path; Stage 6 instead reuses its in-memory planning
records.

## Physical VDF planes

Selection can use sample index, CID, or class name. A bounded requested
number of samples is chosen deterministically and remains in sample-index
order. Every plane crosses the sample's own three-dimensional VDF peak:

- `xz` displays `vx-vz` through that VDF's peak `vy` index;
- `xy` displays `vx-vy` through that VDF's peak `vz` index; and
- `yz` displays `vy-vz` through that VDF's peak `vx` index.

Omitting `--plane` draws `xz`, `xy`, and `yz` in that stable order. Repeat
`--plane {xz,xy,yz}` to draw one or several planes in the requested order.
Each figure contains one row per selected sample and one column per requested
plane, and one invocation writes one combined PNG. The shared historical
preparation and renderer use the saved physical velocity grid and same-cell
VLSV sparsity threshold, display axes in km/s, draw square panels with a grey
Cartesian grid and an `f(v)` colorbar, and retain the transparent bad mask
over white axes. A standalone `--plane xz` panel therefore uses the same
plane values, axes, normalization, unmodified `nipy_spectral`, and grid as
the corresponding Stage 6 frame.

Titles identify the plane, sample index, CID, timestep, and class. The command
loads the physical raw `X.npy`, never downsamples, and does not reconstruct
physical values from Hermite coefficients. Repeat `--sample-index`, `--cid`,
or `--class-name` to select several matching samples; these selector types
are mutually exclusive. `--number-of-samples` applies a deterministic
maximum, and `--velocity-limit-ms` supplies an optional symmetric display
limit in m/s before the shared renderer converts the axes to km/s.

## Units

| Quantity | Unit |
|---|---|
| Spatial coordinates and topology | Earth radii, `R_E` |
| Velocity-grid axes and velocity vectors | m/s |
| Magnetic field | tesla |
| Number density | m^-3 |
| VDF sparsity threshold | s³/m⁶ |
| Timestep | simulation index |

## Geometry limitations

- An accessible 3-D source may lack every supported exact same-cell sparsity
  threshold producer; physical VDF plotting fails rather than inventing a
  threshold. Hermite generation does not read that threshold.
- Small cross-runtime point-geometry differences must not be “fixed” by
  rounding or rewriting saved scientific metadata.
