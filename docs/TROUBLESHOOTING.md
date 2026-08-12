# Troubleshooting

The active runtime assumes current inputs and accesses expected files,
columns, shapes, configuration keys, checkpoint fields, and VLSV variables
directly. Tracebacks therefore come from Python or the underlying library;
there is no project-specific validation report or translated error message.

## Import errors

Run module commands from the repository root in the documented environment:

```bash
python -c "import src"
```

If that fails, confirm the current directory, Python interpreter, and
dependency installation. Do not add per-script `sys.path` modifications.

## Analysator cannot be imported

Confirm that the separately installed Analysator checkout and environment are
available by the method described in the
[root README](../README.md#install-analysator). Test
its import with the same interpreter used for project commands.

Do not interpret an Analysator failure from a different system Python as a
project regression.

## Matplotlib, LaTeX, or `PTNOLATEX` errors

Analysator plotting may initialize Matplotlib on demand. Use the environment
settings documented in the
[root README](../README.md#install-analysator) before importing the
plotting stack.

For headless execution, confirm that the selected Matplotlib backend can
write PNG files. Prediction rendering initializes the Analysator plot module
before creating its own axes to avoid mixed initialization order.

## A required VLSV variable is missing

The extraction or prediction stage accesses the expected current producer
directly. Inspect the VLSV variable list with Analysator and compare the
configured population and source convention. The project does not translate
the resulting Analysator, NumPy, or Python exception and does not substitute
an unrelated producer.

### Missing density

A configured-population number-density producer is part of every extracted
or predicted plasma-context row. Resolution uses `<population>/vg_rho`,
`<population>/rho`, then historical unprefixed `rho`. Mass-density,
thermal/nonthermal, or backstream densities are not substituted.

### Missing velocity

Plasma context and spatial maps check direct total bulk velocity in this
order: `<population>/vg_v`, `<population>/V`, `vg_v`, `V`, then `fg_v`. EGI normally uses
`proton/vg_v`; BFA-style sources may use `proton/V`. Only when no direct
producer exists does a BCH-style source use `rho_v / rho`. Thermal,
nonthermal, backstream, and drift velocities are not substitutes.

### Missing magnetic field

Every extraction and VLSV prediction reads a magnetic field for plasma
context. Resolution uses `vg_b_vol`, `B_vol`, `fg_b_vol`, `fg_b`, then
supported legacy `B`. Field-grid values are centred and sampled at the VDF
cell centre. `Bx`, `By`, and `Bz` are saved in tesla; rotated Hermite also
reuses the same vector to construct the
`(parallel, perpendicular bulk flow, cross-product)` frame.

### Missing electric field or pressure tensor

Electric field uses `vg_e_vol`, `E_vol`, `fg_e_vol`, `fg_e`, then `E`;
`vg_e_gradpe` is not total E. Pressure requires a complete current or
historical total diagonal/off-diagonal pair, or a supported complete
thermal/nonthermal or backstream/nonbackstream pair that can be summed. A
scalar pressure is not substituted. Analysator's off-diagonal order is
mapped into saved `(Pxy, Pxz, Pyz)` order.

### Missing `plasma_context.npy`

Current CNN and autoencoder datasets require this float32 aligned array with
shape `(n_samples, 16)`. Earlier context-layout and context-free datasets
require regeneration; the training path does not zero-fill context or migrate
an older layout. PCA remains a
representation-only consumer.

### Missing VDF sparsity threshold

Vlasiator output generations may store the physical VDF sparsity threshold
under different producers. Plotting checks them in Analysator order:

```text
MinValue
<population>/MinValue
<population>/EffectiveSparsityThreshold
<population>/vg_effectivesparsitythreshold
```

Prediction resolves the first available name once when opening the source
and reuses it for all displayed CIDs. For the EGI proton source this may be
`proton/vg_effectivesparsitythreshold`. There is no numerical or zero
fallback. Raw dataset values and raw machine-learning preprocessing do not
use this plotting threshold, but physical VDF panels still require one of the
supported producers.

Hermite generation does not use `MinValue`; it projects the physical VDF
exactly as supplied. A missing threshold therefore affects physical VDF
figures, not unplotted Hermite extraction or Hermite model input.

## Invalid velocity mesh

The selected population is expected to expose a three-dimensional
velocity-mesh shape with a consistent extent. Check population selection,
mesh size, and block size when downstream NumPy reshaping or interpolation
raises.

## Unsupported legacy-B geometry

The legacy magnetic-field reconstruction assumes an unrefined polar
`(nx, 1, nz)` SpatialGrid with usable interior neighbors. Refined, nonpolar,
3-D, or boundary geometry should provide a direct cell-centred volume field.
Plasma context always needs all three magnetic-field components; an
unrotated Hermite transform merely bypasses the field direction in its
coefficient mathematics.

## Unexpected Hermite values

The current transform projects the physical linear VDF exactly as supplied.
It does not clip negative values, apply `log10`, use a `MinValue` threshold,
renormalize density, or normalize the coefficient cube. Inspect endpoint
velocity extents, physical VDF values, the first-moment drift, isotropic
thermal velocity, and configured order. For optional rotation, also inspect
the same-CID magnetic field, total bulk velocity, expanded rotated bounds,
and interpolated grid. Ordinary NumPy/SciPy numerical behavior propagates;
there is no density tolerance or fallback frame.

## One-timestep training limitation

CNN and autoencoder training use train, validation, and test partitions
separated by configured timestep gaps. One timestep cannot form that
structure and will lead to an ordinary downstream indexing, loader, or
estimator exception.

Use a bounded forward test when appropriate. Do not create a random sample
split from the same timestep.

## CUDA unavailable

Use `--device cpu` or `--device auto`. The installed PyTorch build, driver,
and device visibility must all support CUDA. Install a PyTorch build matched
to the system rather than copying a platform-specific command blindly.

Missing CUDA does not invalidate CPU tests.

PCA uses the selected device only for batched feature standardization,
incremental component updates, and score projection. Internal KMeans,
nearest-neighbor search, and t-SNE run on CPU using the configured leading
PCA score columns. An explicit unavailable CUDA request is reported by
ordinary PyTorch operations; `auto` selects CPU when PyTorch reports no CUDA
device.

## Memory pressure

Reduce batch sizes and worker counts before changing scientific algorithms.
Useful first steps:

- reduce PCA `--batch-size` below its shipped value of 16;
- use low-rank PCA when a small retained component count is scientifically
  suitable, while remembering that it is an approximation;
- use a full-node GPU profile for complete-volume raw PCA, CNN, or
  autoencoder work when the workload fits the selected accelerator;
- raw CNN batch size one;
- raw autoencoder batch size one for its complete `268^3` input;
- reduce `model.bottleneck_shape` when dense latent projections or the
  bottleneck stage dominate autoencoder memory;
- normalization batch size one or two;
- start Hermite PCA/CNN/autoencoder batches at one when memory is tight;
- start large rotated 268 x 268 x 268 Hermite extraction with two to four
  timestep workers because each worker holds one active VDF, rotation
  interpolation arrays, and one coefficient cube; unrotated extraction may
  support more;
- keep `extraction_n_jobs` at or below the allocated physical CPU count and
  increase it only after measuring memory use and VLSV filesystem I/O;
- remember that Analysator temporarily materializes one field-grid producer
  at a time in each active timestep worker; the project immediately keeps
  only selected vectors and releases the grid, but worker count still
  multiplies that temporary source memory;
- one-cell region smoke;
- disable optional prediction plotting when testing inference alone; and
- measure with `/usr/bin/time -v`.

PCA batch size controls prepared sample rows, not the retained
`working_rank * n_features` component basis. Full-resolution raw PCA can
therefore still require tens of gibibytes once the basis, update matrix, and
PyTorch decomposition workspaces coexist, even with a small batch.

The removed monolithic raw-PCA path attempted to allocate approximately
844 GiB. Current PCA avoids that complete matrix by reading the source
memory map three times: float64 Chan/Welford statistics, incremental PCA
updates, and score projection. It writes no feature, standardized-feature,
or component cache. One complete sample, the float64 statistics, retained
basis, and update workspace still have to fit, so a full node or GPU does not
guarantee success. Measure the selected dataset, method, batch size, retained
rank, dtype, and device with `/usr/bin/time -v`.

When one GPU cannot hold the retained full-resolution basis and update
workspace, use the one-node feature-sharded launcher rather than merely
requesting more GPUs for the single-process command. `torchrun` gives each
rank one contiguous feature interval, and no rank owns the complete basis:

```bash
sbatch slurm/turso/gpu/run_pca_multi_gpu.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/pca \
    --representation raw \
    --pca-method lowrank \
    --batch-size 8
```

The launcher is single-node NCCL execution. A direct Python PCA command still
uses one device even when several GPUs are visible; it must be launched by
`torchrun` with `--multi-gpu` to pool component memory.

For CNN or autoencoder memory pressure, `--model-parallel-gpus N` distributes
consecutive model stages, optimizer state, and saved activations across visible
GPUs in one process. The autoencoder distributes encoder, bottleneck, decoder,
reconstruction, and topology stages. Neither workflow shards one `Conv3d`, so
an oversized convolution stage and its local activation must still fit on its
owner. Model parallelism may also be slower for small models because
activations cross device boundaries. `loader.num_workers` affects only
loading and preprocessing and cannot distribute the model.

For the autoencoder, `model.bottleneck_shape` is the maximum retained spatial
shape after encoding. Smaller values reduce both dense latent projections,
checkpoint size, and bottleneck-stage memory and compute, but increase spatial
compression. The effective shape never exceeds the encoded volume along any
axis.

Full mode performs exact reduced SVD within each update, but a multi-batch
fit is truncated between updates and is not generally identical to
monolithic PCA. A one-update fit is identical to the corresponding
monolithic reduced SVD. Single-device low-rank mode seeds PyTorch once before
its incremental randomized pass; distributed low-rank mode retains an
oversampled feature-sharded basis through exact small row-Gram eigensolves.
Downstream CPU analysis uses the shared leading 32 score columns by default.
KMeans creates one internal diagnostic partition per physical class present,
and physical labels enter only after fitting for purity, concentration,
neighbor agreement, sample reporting, and plot color.

## t-SNE classes form one dense cloud

t-SNE is label blind and does not guarantee that physical classes become
disconnected islands. Continuous CID or timestep trajectories, genuinely
overlapping plasma populations, and weak class separation in the leading PCA
space can all produce one connected cloud. Distances and empty gaps in a
t-SNE figure are not quantitative physical distances.

The current large-dataset defaults use the leading 32 PCA components,
perplexity 50, PCA initialization, automatic learning rate, early
exaggeration 16, 2000 iterations, marker size 3, and alpha 0.28. These small,
transparent points let dense structure remain visible instead of becoming an
opaque overplotted mass. Perplexities such as 30, 50, or 100 can be compared
in separate runs, but selecting an embedding because it creates the largest
class gaps would misrepresent the label-blind analysis.

Use `metrics.txt` to distinguish visual crowding from numerical local
agreement. It reports nearest-neighbor physical-label purity in PCA space,
internal KMeans purity and class concentration, and the KMeans silhouette
score. The figures use physical classes only; KMeans labels are neither
plotted nor saved. When metadata contains a training split, neighbor search
uses training PCA scores as the reference for every split and excludes each
training query from its own neighbor list. Without split metadata, all PCA
scores form the reference and every query excludes itself before label
agreement is summarized by split and physical class.

## PCA arrays or CSV tables are absent

This is intentional. PCA scores, KMeans labels, and t-SNE coordinates are
intermediate arrays kept only in memory. A completed PCA output directory
contains exactly:

```text
pca_physical_classes.png
tsne_physical_classes.png
metrics.txt
```

Explained and cumulative variance, sample counts, internal KMeans
diagnostics, and nearest-neighbor physical-label purity are consolidated in
`metrics.txt`. The workflow does not save score or label arrays, CSV tables,
or a separate explained-variance figure.

## Reading autoencoder reconstruction metrics

Autoencoder `metrics.txt` reports complete-volume reconstruction MSE in the
training-derived normalized representation space, masked topology Smooth L1
in scaled target space, and their weighted total objective. Validation total
loss selects the restored model. Compare the objective table across train,
validation, and test, then use the physical-class reconstruction table to
identify populations that are easier or harder to reconstruct. The topology
table reports each target's valid count, MAE, and RMSE after inverse scaling to
Earth radii. Missing metadata is masked rather than treated as zero.

Class labels are joined after reconstruction only. Topology metadata is an
auxiliary target that supervises the latent vector, not a model input or
decoder condition. `training_history.csv` records reconstruction, topology,
and total losses for training and validation. `reconstruction_examples.png`
shows deterministic original, reconstructed, and absolute-error panels from
the same restored best model. Raw panels first reverse preprocessing for the
complete physical volume, then use the extraction Stage 6 renderer on one x-z
plane through the original three-dimensional peak with a shared `LogNorm`,
unmodified `nipy_spectral`, physical km/s axes, and transparent masks over
white axes. That plane is visualization only. Hermite panels remain signed
coefficient views with a symmetric scale and white axes. Raw examples are
reconstructed one at a time and reduced to copied display planes before the
next row, preventing the figure builder from retaining all selected full
volumes.

## Representation mismatch

The command selector (`raw` or `hermite`), saved dataset, configuration, and
checkpoint are expected to agree. Raw PCA, CNN, and autoencoder input shapes
follow the complete saved training grid in `(vx, vy, vz)` order; Hermite order
and axes follow the saved array/checkpoint convention. Unrotated coefficients
use `(n_x, n_y, n_z)` and optional rotated coefficients use
`(n_parallel, n_perp1, n_perp2)`. Mismatches surface through ordinary
dictionary, NumPy, or PyTorch operations.

Current PCA, CNN, and autoencoder representation values are `raw` and
`hermite`. Removed sliced raw CNN checkpoints and retired two-dimensional or
reconstruction-only autoencoder checkpoints are not adapted. Neither are
Hermite datasets or checkpoints from the retired compact-log, fixed-order
convention: regenerate the dataset and retrain the model.

## Checkpoint from another workflow

Current CNN prediction reads checkpoint fields directly. Current
autoencoder loading reads checkpoint fields directly. Files from removed
historical workflows are not inspected or converted; recover the
appropriate historical commit in a separate directory.

Direct representation names, preprocessing values, architecture fields, and
topology target order define current model checkpoints. Context-aware CNN and
autoencoder checkpoints additionally require the exact 16 context names and
their training mean and scale; earlier context-layout and context-free models
require retraining.

The autoencoder checkpoint stores a CPU state dictionary and no runtime
CUDA identifiers, stage map, or requested GPU count. Choose CPU, one-GPU, or
multi-GPU placement when loading; changing the runtime GPU count does not
require rewriting the checkpoint.

## Class or topology mismatch

Class IDs/names and topology target order are saved model data. Do not
reorder classes alphabetically or reorder topology columns. Recreate the
intended model from a matching dataset and configuration.

The current project ID order is `lobe` (0), `current_layer` (1),
`o_point` (2), `x_point` (3), `solar_wind` (4), `magnetosheath` (5), and
`inner_magnetosphere` (6). CNN output width follows the classes represented
in dataset metadata. A manual class with an empty `class_coords_re` list
contributes no sample and therefore cannot appear in that dataset's
checkpoint mapping.

If `current_layer` samples are unexpectedly absent, inspect the inclusive
tail bounds (`-30 <= x_re <= -10`, `|z_re| <= 1`) and the earlier manual,
X/O, and rejected CID assignments. This class is a geometric magnetotail
fallback, not a manual-coordinate or current-density selection.

For X/O sample counts, first check the configured `manual`, `physical`,
`consensus`, or `union` method. Consensus is the set intersection and does not
fall back when the result is empty; union is the set union. A detected point
with no final VDF cell is intentionally absent from metadata and plotting. If
an X and O record share any final selected VDF cell, both complete records and
all their selected cells are omitted without point-kind priority or an overlap
class. Stage 6 and standalone colormaps therefore show only accepted blue
source coordinates and their red, same-shape selected-VDF markers.

## Config path or key exception

Confirm the current paths:

```text
configs/data/extraction.yaml
configs/analysis/pca.yaml
configs/models/cnn.yaml
configs/models/autoencoder.yaml
configs/prediction/coordinate.yaml
configs/prediction/region.yaml
```

Review the corresponding functionality guide for required sections and
units:

- [data and representations](DATA_PHYSICS_REPRESENTATIONS.md);
- [PCA](PCA.md);
- [CNN](CNN.md);
- [autoencoder](AUTOENCODER.md); or
- [prediction and plotting](PREDICTION_AND_PLOTTING.md).

For complete local and Slurm invocations, see
[Example commands](EXAMPLE_COMMANDS.md).

Configuration fields are accessed directly. A missing path or key therefore
uses the ordinary filesystem, YAML, or dictionary exception.
The inline comments above each YAML key are the quickest reference for
accepted values and units.

## Requested and selected coordinates differ

This is expected. Coordinate prediction accepts a requested location and
selects the nearest cell that actually carries a VDF. The one-row
`predictions_<timestep>.csv` preserves both positions; the optional figure
shows the request as a star and the selected VDF with its predicted-class
marker. The CSV is written before figure rendering.

## Topology vectors are absent from a prediction figure

This is expected. Coordinate and region CSV rows retain both predicted
distances and all four x-z displacement components in Earth radii. Combined
coordinate figures render each positive scalar X/O distance as an unfilled
circle around the selected VDF cell, but intentionally omit vector arrows,
inferred endpoints, and connectors. Region figures render no topology
circles or geometry; their CSV values remain identical. Rendering does not
clamp, normalize, or otherwise change the saved topology values.

## A dataset colormap cannot resolve its VLSV source

The colormap command resolves the source from an explicit CLI override,
`metadata.file_location`, or `configs/data/extraction.yaml`. If none points
to an existing source for the selected timestep, pass the source file
explicitly and confirm its timestep matches the plotted metadata.

Do not substitute a different timestep merely to produce a background.

## A VDF plot cannot select a sample

`--sample-index` refers to the saved row identity, while `--cid` refers to a
VDF-carrying spatial cell. Confirm the requested value exists in
`metadata.csv` and that `X.npy` is present. Class-name selection must use a
saved physical class name.

Selection is deterministic and preserves sample-index order. The command
does not reconstruct physical VDFs from `X_hermite.npy`.

## A VDF slice appears blank

Each plane must pass through that VDF's own three-dimensional peak:

- `vx-vy` uses the peak `vz`;
- `vx-vz` uses the peak `vy`; and
- `vy-vz` uses the peak `vx`.

A geometric centre slice can miss localized support and appear blank. Check
that the plotted indices are the per-sample peak indices, that the saved
velocity extents are finite, and that the selected `X.npy` row contains the
expected VDF. Do not add downsampling or replace the peak with a fixed centre
slice.

Extraction Stage 6 and the standalone command use the same peak-plane
extractor and renderer. Stage 6 always requests the square `vx-vz` plane
through the sample's peak `vy` index. The standalone command defaults to
`xz`, `xy`, and `yz`, each crossing the full three-dimensional peak, or uses
the order of repeated `--plane` options. Their axes are km/s.

For the selected CID, confirm that the source VLSV provides a finite positive
same-cell sparsity threshold under one of the four supported producer names
and that `dv = (vymax - vymin) / nvy` is positive. The renderer multiplies a
float32 plane copy by `dv`, sets only values strictly below `threshold * dv`
to zero, and masks all nonpositive entries. It then uses unmodified
`nipy_spectral` with `LogNorm` over the visible support. A blank white or
transparent region may therefore contain only thresholded or masked values;
the bad mask is not forced to black. This preparation does not modify
`X.npy` or a machine-learning input.
