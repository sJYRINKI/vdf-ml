# vdf-ml

## Overview

`vdf-ml` studies plasma velocity-distribution functions (VDFs) in
[Vlasiator](https://github.com/fmihpc/vlasiator) simulation output. It
extracts VDF-bearing spatial cells, assigns physics-based plasma-region and
X/O-point labels, stores aligned raw and Hermite data, and provides focused
analysis and machine-learning workflows.

The scientific goal is to compare label-blind structure in VDFs with
physical classes and to predict both a physical class and auxiliary X/O
topology from a VDF. The project also supports bounded prediction at one
coordinate or through a spatial region, with figures that connect spatial
results to representative source VDFs.

```text
Vlasiator VLSV + flux file
    -> VDF extraction and physical labeling
    -> saved dataset
    -> raw or Hermite representation
    -> PCA, CNN, or autoencoder
    -> coordinate/region predictions and plots
```

All executable project code is under `src`. Historical
implementations were removed; Git history is the recovery path for
superseded workflows.

## Main capabilities

- VDF extraction from configured Vlasiator timesteps
- manual plasma-region labels, a magnetotail current-layer fallback, and
  detected X/O points with historical manual/physical cell selection
- six masked auxiliary X/O distance and direction targets
- complete three-dimensional `raw` VDF features with no input slicing
- dimensional `hermite` coefficients from the physical VDF, in the original
  velocity frame by default or an optional magnetic/bulk-flow frame
- three-pass, sample-batched PyTorch feature standardization and incremental
  full or low-rank PCA, followed by physical-class plots and internal
  KMeans, nearest-neighbor, and t-SNE diagnostics
- a raw or Hermite 3-D multitask CNN
- a deterministic, fully three-dimensional raw/Hermite reconstruction
  autoencoder with auxiliary topology supervision
- coordinate and bounded-region prediction
- combined x-velocity maps and peak-centred VDF figures for predictions
- one shared Stage 6/standalone dataset-colormap style with role-based overlays
- shared peak-centred physical VDF planes for Stage 6 and standalone figures
- optional per-timestep x-velocity colormaps, per-cell-ID peak-`vy`
  `vx-vz` VDF frames, signed per-cell-ID Hermite frames, and raw-frame
  time-evolution animations created after extraction
- direct current-format outputs with ordinary library error propagation

## Repository structure

| Path | Responsibility |
|---|---|
| `src/physics/` | X/O topology, labels, physical context, and Hermite mathematics |
| `src/representations/` | ordered loading plus shared raw and Hermite preparation |
| `src/analysis/` | ordered PCA, physical-label diagnostics, t-SNE, and output stages |
| `src/cnn/` | ordered CNN loading, splitting, scaling, training, evaluation, and saving |
| `src/autoencoder/` | ordered reconstruction loading, training, evaluation, and saving |
| `src/learning/` | shared topology-target extraction, scaling, and masked supervision |
| `src/prediction/` | ordered shared inference stages plus coordinate and region orchestration |
| `src/plotting/` | dataset, VDF, prediction, and frame-animation visualization |
| `src/configuration/` | shared YAML configuration loading |
| `scripts/` | eight thin module commands |
| `configs/` | six pipeline configurations |
| `docs/` | current scientific and operational documentation |
| `docs/LEGACY.md` | concise historical recovery guidance |
| `externals/analysator/` | optional local Analysator checkout; ignored by Git |

## Installation

### Prerequisites

The verified environment uses:

- Python 3.12.3; and
- Analysator 0.9.8

Extraction additionally requires access to the relevant Vlasiator VLSV and
flux files.

### Create the environment

Clone the repository and work from its root:

```bash
git clone https://github.com/sJYRINKI/vdf-ml.git
cd vdf-ml
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirement files deliberately avoid arbitrary version pins. The
project uses NumPy, SciPy, pandas, Matplotlib, scikit-learn, PyTorch,
PyYAML, joblib, and Shapely.

These versions describe the verified environment rather than minimum
requirements:

| Package | Verified version | Project use |
|---|---:|---|
| NumPy | 1.26.4 | arrays and numerical operations |
| SciPy | 1.13.1 | Hermite interpolation and topology convolution |
| pandas | 2.2.2 | metadata and result tables |
| Matplotlib | 3.10.9 | analysis, dataset, and prediction figures |
| scikit-learn | 1.8.0 | KMeans, t-SNE, and metrics |
| PyTorch | 2.7.1+cu118 | feature scaling, PCA, CNN, and autoencoder |
| PyYAML | 6.0.3 | configuration loading |
| joblib | 1.4.2 | bounded extraction jobs |
| Shapely | 2.1.2 | point-topology geometry |

### Install Analysator

[Analysator](https://github.com/fmihpc/analysator) reads Vlasiator VLSV
files and supplies physical colormap functionality. It is maintained
separately and is not present in a fresh clone.

```bash
mkdir -p externals
git clone https://github.com/fmihpc/analysator.git externals/analysator
python -m pip install --editable externals/analysator
```

No project-specific `PYTHONPATH` is needed. The repository has no root
installation metadata; run its modules from the repository root so Python
can resolve `src` and `scripts`.

Analysator is required for VLSV-facing extraction, prediction, and
colormaps. Saved-array PCA, CNN, and autoencoder work can run without
opening a VLSV file.

For headless Analysator plotting without TeX, set these before launching
Python:

```bash
export PTNOLATEX=1
export PTNONINTERACTIVE=1
```

They are optional when the host already provides a suitable plotting
environment. There are no mandatory project-specific environment variables;
source and output locations are CLI arguments or YAML values.

### CPU and optional CUDA

For GPU execution, install the PyTorch build
compatible with the operating system, driver, and CUDA runtime using the
[official PyTorch selector](https://pytorch.org/get-started/locally/).
Do not reuse another system's CUDA wheel command without checking
compatibility. PCA, CNN, autoencoder, and prediction commands accept `cpu`,
`cuda`, `cuda:N`, or `auto` where a device override is supported.

### Verify the environment

```bash
PYTHONDONTWRITEBYTECODE=1 python -c \
  "import src; import analysator; import numpy; import pandas; import scipy; import sklearn; import torch; import yaml"
```

If `src` cannot be imported, confirm that the command is running from
the repository root. If Analysator cannot be imported, confirm that the
intended environment is active and repeat its editable installation.

Verify the command surface:

```bash
python -m scripts.data.extract_dataset --help
python -m scripts.data.plotting.plot_colormap --help
python -m scripts.data.plotting.plot_vdf_slices --help
python -m scripts.analysis.run_pca_analysis --help
python -m scripts.models.train_cnn --help
python -m scripts.models.train_autoencoder --help
python -m scripts.prediction.predict_coordinate --help
python -m scripts.prediction.predict_region --help
```

### Site-specific setup and common failures

The checked-in YAML files demonstrate one BCH/EGI HPC environment. Before
running elsewhere, review source templates, output roots, spatial regions,
and job counts.

Common setup failures are:

- `No module named src`: run from the repository root;
- `No module named analysator`: activate the intended environment and
  repeat the editable Analysator installation;
- VTK or yt installation errors: use a Python/platform combination
  supported by Analysator;
- TeX errors during Analysator plots: set `PTNOLATEX=1` before importing
  Analysator;
- CUDA unavailable: use CPU or install a host-compatible PyTorch build.

## Quick start

Review the site-specific source and output paths in the checked-in YAML files,
then create one dataset from the repository root:

```bash
python -m scripts.data.extract_dataset \
    --config configs/data/extraction.yaml \
    --start-timestep 3072 \
    --n-timesteps 1 \
    --dataset-kind train
```

Extraction postprocessing is controlled by explicit Boolean settings. The
same command can optionally render per-timestep colormaps, per-cell-ID peak-centred VDF
frames, signed per-cell-ID Hermite frames, and raw-frame time-evolution
animations through the `postprocessing` section of
`configs/data/extraction.yaml`. See
[Example commands](docs/EXAMPLE_COMMANDS.md#dataset-extraction) for
the exact keys and output directories.

Integrated Stage 6 colormaps use x-directed bulk velocity on a symmetric
red-white-blue scale, black magnetic-field streamlines, yellow VDF-bearing
cells, blue source coordinates, and red selected dataset cells. Its VDF
frames are square `vx-vz` cuts through each sample's own peak `vy`. Historical
display preparation resolves the source's VDF sparsity-threshold producer in
Analysator-era order: `MinValue`, then `<population>/MinValue`, then
`<population>/EffectiveSparsityThreshold`, then
`<population>/vg_effectivesparsitythreshold`. Prediction resolves that name
once per open VLSV source and reuses it for every displayed CID. The renderer
uses `dv = (vymax - vymin) / nvy`, multiplies a float32 plane copy by `dv`,
sets values strictly below `threshold * dv` to zero, and masks nonpositive
values. It retains unmodified `nipy_spectral` with `LogNorm`; the bad mask is
transparent over white axes rather than forced black. Velocity axes are
km/s. The standalone colormap uses the same renderer and defaults as Stage
6. The standalone VDF command uses the same preparation and drawing style:
omitting `--plane` draws peak-crossing `xz`, `xy`, and `yz`, while repeated
`--plane` options select their order. Stage 6 always requests only `xz`.
Prediction figures reuse the physical-panel preparation without colored
panel borders. All preparation is performed on display copies; saved VDFs
and machine-learning inputs are unchanged.

Complete verified local commands, output lists, CPU and CUDA variants, and
Turso/HILE Slurm submissions are in
[Example commands](docs/EXAMPLE_COMMANDS.md).

## Active commands and configurations

The eight public commands are:

```text
python -m scripts.data.extract_dataset
python -m scripts.data.plotting.plot_colormap
python -m scripts.data.plotting.plot_vdf_slices
python -m scripts.analysis.run_pca_analysis
python -m scripts.models.train_cnn
python -m scripts.models.train_autoencoder
python -m scripts.prediction.predict_coordinate
python -m scripts.prediction.predict_region
```

Their six configurations are:

```text
configs/data/extraction.yaml
configs/analysis/pca.yaml
configs/models/cnn.yaml
configs/models/autoencoder.yaml
configs/prediction/coordinate.yaml
configs/prediction/region.yaml
```

Comments immediately above the keys in each YAML file are the quickest
reference for configuration ownership, accepted choices, units, coordinate
order, and output effects. The subsystem guides provide the surrounding
scientific conventions.

## Dataset files

A raw dataset contains:

```text
X.npy
metadata.csv
velocity_grid.npz
```

When Hermite generation is enabled, it also contains:

```text
X_hermite.npy
```

Optional extraction postprocessing adds only:

```text
plots/
animations/
```

The five frame and animation switches control only derived output; the core
dataset layout is unchanged by postprocessing.

`X.npy` stores complete VDFs in `[vx, vy, vz]` order. `metadata.csv` owns
physical `class_id` and `class_name`. Current workflows trust these saved
files and directly load the data they need. Planning and extraction use
separate `planning_n_jobs` and `extraction_n_jobs` settings. The parent
streams the first nonempty timestep used to discover the raw shape. With
more than one extraction job, remaining timesteps run in Joblib workers;
each worker reuses one source reader and VDF extractor while processing its
samples sequentially into worker-local raw and optional Hermite memory maps.
Only the parent copies those aligned rows into the hidden sibling staging
directory in planned timestep and sample order, flushes and closes the final
arrays, and renames the directory to the final path. Setting
`extraction_n_jobs: 1` keeps extraction serial.

The current physical-class IDs are:

| `class_id` | `class_name` | Assignment |
|---:|---|---|
| 0 | `lobe` | configured coordinate |
| 1 | `current_layer` | remaining VDF cells in the magnetotail sampling region |
| 2 | `o_point` | accepted O-point VDF-cell selection |
| 3 | `x_point` | accepted X-point VDF-cell selection |
| 4 | `solar_wind` | configured coordinate |
| 5 | `magnetosheath` | configured coordinate |
| 6 | `inner_magnetosphere` | configured coordinate |

Manual coordinates use `[x_re, y_re, z_re]` in Earth radii and select the
nearest VDF-carrying cell. An empty coordinate list contributes no samples.
`current_layer` is not a manual class: after manual and X/O assignment, it
receives unassigned VDF cells in the inclusive magnetotail box
`-30 <= x_re <= -10`, `|z_re| <= 1`. This fallback does not use a
current-density detector, and accepted X/O assignments precede the fallback.

X/O source detection is unchanged. For each detected point, `manual` uses
the configured box, `physical` uses the Hessian-aligned X box or O flux
contour, `consensus` keeps their intersection, and `union` keeps either
selection. Empty results are omitted. If any final X and O selections share
a stored VDF cell, both complete source records and every cell they selected
are omitted; there is no point-kind priority or overlap class. Dataset
colormaps show accepted sources in blue and their selected VDF cells in red
with the same deliberately small marker shape, while all candidate VDF cells
remain yellow. Legend entries are paired by class or point kind;
`current_layer` has only a red selected-cell entry because it has no source
coordinate.

## Representations

### `raw`

`X.npy` stores complete VDFs in `(sample, vx, vy, vz)` order. Raw PCA, CNN,
and autoencoder preprocessing apply the configured positive floor and
`log10` to every velocity-space cell. PCA flattens all `vx * vy * vz` values
in stable C order; both neural models add one channel and preserve all three
velocity axes for `Conv3d`. None of these paths slices, crops, projects,
averages, or downsamples the physical VDF. Peak-centred planes shown in
figures are visualization only and never become model inputs or loss targets.

### `hermite`

The transform projects the physical linear VDF directly; it applies no
`log10`, `MinValue` threshold, clipping, density normalization, or per-sample
coefficient normalization. Velocity coordinates use endpoint-inclusive
`np.linspace(vmin, vmax, n_cells)`. The drift is the VDF first moment and the
thermal scale is its isotropic standard deviation. Gaussian-weighted
physicists' Hermite polynomials are divided by
`sqrt(2**n * n! * sqrt(pi) * vth)`, contracted in float64, and multiplied by
`dv**3` from the first velocity-axis extent and sample count. The resulting
coefficients are dimensional.

The default order is 22, but every consumer derives the actual cube shape
from `X_hermite.npy` or the checkpoint. Without rotation the axes are
`[n_x, n_y, n_z]`. Optional extraction rotation uses the frame
`[parallel B, perpendicular bulk flow, B x perpendicular flow]` and produces
`[n_parallel, n_perp1, n_perp2]`. Hermite-enabled extraction uses the same
timestep worker count as raw-only extraction. Each physical VDF is extracted
once, and its raw row and Hermite cube use the same local and final row.
Rotation increases per-worker memory because interpolation arrays coexist
with one raw VDF and one coefficient cube.

This execution-only change leaves raw-only numerical output and Hermite
coefficient values unchanged. Throughput depends on both VLSV filesystem I/O
and the per-sample Hermite calculation, so worker counts should be increased
only after measuring the intended dataset on its allocated node.

Hermite datasets and Hermite-trained checkpoints created before this
physical-VDF convention must be regenerated and retrained. The active code
does not adapt or migrate the retired compact-log representation.

Neither representation appends labels, topology, coordinates, moments, or
other metadata.

| Workflow | Raw input | Hermite input |
|---|---|---|
| PCA | complete 3-D VDF, flattened in C order | complete coefficient cube, flattened in C order |
| CNN | complete 3-D VDF with `Conv3d` | complete coefficient cube with `Conv3d` |
| Autoencoder | complete 3-D VDF with `Conv3d` | complete coefficient cube with `Conv3d` |

## PCA analysis

```text
read-only raw or Hermite memory map
    -> batched float64 Chan/Welford statistics
    -> batched incremental PyTorch full or low-rank PCA updates
    -> batched projection onto the retained components
    -> internal KMeans and nearest-neighbor diagnostics on leading scores
    -> label-blind t-SNE on the same leading scores
    -> physical-class-only PCA and t-SNE figures
```

The saved array remains a read-only memory map and is processed three times:
once for float64 per-feature mean/M2 statistics, once for incremental PCA
basis updates, and once to project the final scores. The shipped
`pca.batch_size` is 16 and `--batch-size` overrides it; examples may use 8
for a smaller per-update working set. This replaces the
monolithic raw-PCA path that attempt to allocate too much memory.
No feature, standardized-feature, or component cache is written.

The optional `pca.multi_gpu` path is launched with one `torchrun` process
per GPU on one node. It divides the flattened feature dimension into
contiguous rank-local intervals, so each GPU owns only its local mean,
scale, feature-batch columns, and component-basis columns. NCCL sums the
small row-Gram matrix for each incremental update, and each rank reconstructs
only its own component columns from rank zero's broadcast small eigenpairs.
Final score contributions are reduced to rank zero; only rank zero runs
KMeans, nearest-neighbor metrics, t-SNE, plotting, and output writing. No
GPU stores or gathers the complete PCA basis. The existing CPU and
single-GPU paths remain available unchanged.

`full` uses exact reduced `torch.linalg.svd` inside each update. A one-update
fit is identical to the corresponding monolithic reduced SVD; with multiple
batches, retained-rank truncation makes the result incremental and not
generally identical to monolithic PCA. On one device, `lowrank` seeds
PyTorch once before its incremental randomized `torch.pca_lowrank` updates.
Distributed `lowrank` instead retains an oversampled feature-sharded basis
through exact small row-Gram eigensolves. PCA scores and explained-variance
ratios return to CPU NumPy arrays before the internal scikit-learn diagnostic
and t-SNE stages.

PCA, KMeans, nearest-neighbor search, and t-SNE fitting are label-blind.
KMeans uses the leading 32 retained scores by default and creates as many
diagnostic partitions as physical classes occur in `metadata.csv`; its IDs
are neither plotted nor saved. Beyond determining that partition count,
physical labels are introduced only after fitting for KMeans purity, class
concentration, local nearest-neighbor purity, sample counts, and plot styles.
When split metadata is present, nearest-neighbor purity uses training scores
as the reference for every split and removes each training query from its own
neighbor list. Without split metadata, all scores form the reference and each
query is likewise excluded from itself.

t-SNE also uses the leading 32 scores by default, with perplexity 50, PCA
initialization, automatic learning rate, early exaggeration 16, and 2000
iterations. Both figures use point area 3; PCA uses alpha 0.30 and t-SNE
uses alpha 0.28 to reduce overplotting without moving classes or fabricating
gaps. Continuous CID or timestep tracks and
overlapping physical classes may therefore remain, and distances between
separated t-SNE groups are not quantitative.

Each PCA run writes exactly:

```text
pca_physical_classes.png
tsne_physical_classes.png
metrics.txt
```

Both figures use only current physical classes for color, marker, and legend.
`metrics.txt` consolidates run settings, dataset and split/class counts,
explained variance, KMeans purity and concentration, silhouette score, and
nearest-neighbor physical-label purity. PCA scores, KMeans labels, t-SNE
coordinates, CSV tables, and an explained-variance figure remain in memory
only or are not produced.

## CNN

`VdfCNN` accepts one representation tensor and returns dataset-derived
physical-class logits, six auxiliary topology predictions, and an embedding.
The classification width and explicit output order come from the class IDs
present in dataset metadata; topology is an output, never an input.

Training uses complete-timestep train/validation/test partitions with gaps,
training-only input normalization and topology scaling, natural sample
frequency, AdamW, unweighted cross-entropy plus masked Smooth L1 loss, and
validation macro-F1 checkpoint selection. The CNN checkpoint stores the
representation, input shape and normalization, class mapping, topology
scaling and order, architecture, and training velocity grid. A training run
writes only `model.pt` and `metrics.txt`.

With `model_parallel_gpus` greater than one, one Python process places
consecutive encoder stages and the output stage on several visible CUDA
devices. Activations move between those devices while autograd carries
gradients across the transfers. Parameters, AdamW state, and saved
intermediate activations are therefore distributed instead of replicating
the complete model as DDP or `DataParallel` would. This layer-parallel path
does not spatially shard an individual `Conv3d`; each individual stage and
its activation must still fit on its assigned GPU. Runtime placement is not
stored in checkpoint tensor names, so the same checkpoint can be loaded with
one or several GPUs.

## Autoencoder

`VdfAutoencoder` is one Conv3d model for both representations. Raw batches
contain every `X.npy` voxel in `(batch, 1, vx, vy, vz)` order; Hermite batches
contain every signed coefficient in `(batch, 1, n1, n2, n3)` order. Its
encoder creates one latent vector that feeds both the Conv3d decoder and a
small auxiliary topology head. The head predicts, in order,
`distance_to_x_point_re`, `distance_to_o_point_re`,
`vdf_to_x_point_dx_re`, `vdf_to_x_point_dz_re`,
`vdf_to_o_point_dx_re`, and `vdf_to_o_point_dz_re`. These Earth-radius values
come from metadata, are scaled from valid training entries only, and remain
masked where missing. They supervise the latent space but never enter the
model input or decoder.

Autoencoder samples use on-demand memory-mapped loading. Each DataLoader
process opens `X.npy` or `X_hermite.npy` read-only on its first sample request,
reuses that mapping, and has `__getitem__` copy and preprocess only the
requested row. The complete prepared dataset is never materialized in RAM;
the model still receives an ordinary complete three-dimensional batch.

`model.bottleneck_shape` configures the maximum retained spatial cells along
the three model axes after encoding. The effective shape is capped by the
encoded volume on each axis, so adaptive pooling never expands a smaller
axis. Smaller values reduce the two dense bottleneck projections and increase
compression; larger values retain more spatial information but increase
parameters, checkpoint size, and bottleneck-stage memory and compute.

Training uses complete-timestep partitions, AdamW, and one combined objective:

```text
total_loss = full_volume_reconstruction_mse
           + topology_loss_weight * masked_topology_smooth_l1
```

Validation total loss controls early stopping and the selected checkpoint.
With `model_parallel_gpus` greater than one, one process can place consecutive
encoder, bottleneck, decoder, reconstruction, and topology stages across
several visible GPUs. Activations move only when stage ownership changes, and
one optimizer owns every parameter. The same placement path supports CPU and
one GPU. `data_loader.num_workers` controls loading and preprocessing only.
This is layer model parallelism, not DDP, model replication, or data
parallelism.

The autoencoder checkpoint stores architecture, complete input
shape, representation and topology scaling, selection metadata, and a CPU
state dictionary. Runtime CUDA IDs and stage mapping are not stored. Direct
representation and topology-order fields define model identity.
Existing two-dimensional reconstruction-only checkpoints require retraining.
A run writes `autoencoder.pt`, `metrics.txt`, `training_history.csv`, and
`reconstruction_examples.png`. Raw figures restore complete physical volumes,
then display one x-z plane through the original volume's three-dimensional
peak for original, reconstruction, and absolute error. Hermite figures retain
their signed coefficient-plane convention. Those planes are visualization
only. Figure preparation reconstructs one full raw pair at a time and retains
only its two fixed display planes, avoiding a collection of full `268^3`
examples in host memory. Full-resolution raw training is memory intensive;
batch size one is a practical starting point, every individual Conv3d stage
must fit on its owner, and activation transfers can make model parallelism
slower for small models.

## Prediction and figures

Coordinate prediction writes one `predictions_<timestep>.csv` row using the
same schema and writer as region prediction. It records both the requested
position in Earth radii and the actual nearest VDF-cell centre, writes the CSV
before optional rendering, and does not write the former JSON sidecar. Region
prediction continues to stream per-timestep CSV rows in stable cell order and
bounded batches. Both outputs retain checkpoint-ordered class probabilities,
`predicted_probability`, and all six inverse-scaled topology values.

Optional combined figures prefer a directly stored x velocity in this order:
`<population>/vg_v`, `<population>/V`, `vg_v`, and `V`. BCH-style sources
without a direct producer use the legacy `rho_v / rho` ratio. The result stays
in m/s on a symmetric red-white-blue scale with black magnetic streamlines and
the fixed `[-30, 30, -15, 15]` x-z Earth-radii view. A dedicated legend row
sits immediately below the map. Coordinate figures use a compact blue star
for the requested location and a compact predicted-class marker for the
selected VDF cell. They retain both scalar X/O distance circles and the
`cid`/class/probability title. Region figures use only compact predicted-class
markers for representatives with displayed VDF panels: class names remain in
the legend rather than beside individual markers, no distance circles are
drawn, and the spatial title is a concise region description. Arrows,
inferred endpoints, and connectors are not rendered in either mode, although
all six topology values remain in CSV and model results.

The active runtime assumes current datasets, configurations, checkpoints,
and source variables are correctly formed. It accesses expected fields
directly and lets Python and the underlying scientific libraries report
ordinary exceptions.

## Documentation

- [Example commands](docs/EXAMPLE_COMMANDS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Data, physics, and representations](docs/DATA_PHYSICS_REPRESENTATIONS.md)
- [PCA](docs/PCA.md)
- [CNN](docs/CNN.md)
- [Autoencoder](docs/AUTOENCODER.md)
- [Prediction and plotting](docs/PREDICTION_AND_PLOTTING.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Historical recovery](docs/LEGACY.md)

## Historical code

Legacy and deprecated implementations, commands, configurations, and old
checkpoint loaders are absent from `main`. The earlier implementation is
preserved on the separate `legacy` branch. Review or run an old workflow
only from a separate checkout; see
[Historical recovery](docs/LEGACY.md).

## License and citation

This checkout contains no project-specific license or citation file.
Consult the project maintainer before redistribution or citation. Analysator
has separate license and citation metadata in its own repository.
