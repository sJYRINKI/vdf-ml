# Architecture

## Overview

The project has one implementation package:

```text
src/
├── data/
├── physics/
├── representations/
├── analysis/
├── cnn/
├── autoencoder/
├── prediction/
├── plotting/
└── configuration/
```

Command-line entry points are thin modules under `scripts/`. Configuration
files are grouped by the same domains under `configs/`.

The main data flow is:

```text
Vlasiator VLSV + flux files
    |
    v
data and physics
    -> VDF-cell discovery
    -> X/O detection and physical labels
    -> raw VDF extraction
    -> optional Hermite coefficients
    |
    v
saved dataset
    |-----------------------|-------------------------|
    v                       v                         v
PCA analysis         CNN or autoencoder      optional plot frames
    |                       |                         |
    v                       v                         v
PCA figures + report   model checkpoints      optional animations
                            |
                            v
                    coordinate or region prediction
                            |
                            v
                    tables and combined figures
```

## Package responsibilities

### `src.data`

This package owns source-file resolution, VDF-cell discovery, velocity-grid
descriptors, sample planning, metadata, staged writing, and ordinary final
directory creation. Its linear creation path is named
`step_00_create_dataset.py` through `step_07_animate_dataset_frames.py`.
Stages 0 through 5 create the core dataset. Config-controlled Stage 6
renders per-timestep colormaps, per-cell-ID raw-VDF frames, and optional
signed per-cell-ID Hermite coefficient frames after the final directory
rename; Stage 7 creates animations only from the colormap and raw-VDF frame
directories.
Disabled postprocessing does not import or run either visual
stage. The package does not depend on analysis or model code.

Stage 0 streams the first nonempty timestep through the parent process to
discover the raw VDF shape. Stage 4 then submits one Joblib task per remaining
timestep when `extraction_n_jobs` is greater than one, for either raw-only or
paired raw-plus-Hermite extraction. A worker opens one source, reuses one VDF
extractor, and processes samples sequentially into worker-local memory maps.
The parent alone consumes results in submission order, copies raw and
optional Hermite rows into the identical next final range, and places
metadata at those indexes. Workers never receive or write the final staged
memory maps, and `extraction_n_jobs: 1` retains the serial path.

### `src.physics`

This package owns X/O point selection and topology, physical labels,
same-cell magnetic-field and velocity producers, VDF sparsity-threshold
resolution for physical plotting, and the Hermite transform. Threshold
resolution follows
Analysator-era order: `MinValue`, then `<population>/MinValue`, then
`<population>/EffectiveSparsityThreshold`, then
`<population>/vg_effectivesparsitythreshold`. Prediction resolves the name
once per open source and retains it with the prepared source. Physics
functions operate on explicit arrays, coordinates, and source readers. They
do not import PCA, CNN, autoencoder, prediction, or command-line code.

The Hermite transform projects the physical linear VDF on endpoint velocity
axes with a physically normalized physicists' basis and `dv**3` quadrature.
It is unrotated by default; optional rotation uses the magnetic-field and
perpendicular total-bulk-flow frame. `MinValue` is not a Hermite input.

### `src.representations`

This package is the shared feature boundary for PCA and CNN workflows.
`step_01_load_saved_representation.py` directly opens the selected saved
representation; stages 2 and 3 prepare the alternative raw and Hermite
inputs. The package creates either:

- a complete logarithmic raw VDF in `(vx, vy, vz)` order; or
- a complete dimensional `hermite` coefficient volume whose saved shape and
  optional rotation setting are propagated to consumers.

The source arrays remain read-only. The package does not own training splits
or model-specific normalization. Raw PCA transforms and flattens complete
sample batches on demand, while raw CNN training preserves all three
velocity axes and adds one channel for `Conv3d`.

### `src.analysis`

This package owns label-blind, three-pass PyTorch PCA; internal scikit-learn
KMeans, nearest-neighbor search, and t-SNE; post-fit physical diagnostics;
two physical-class figures; and one consolidated metrics report. The
selected representation stays memory mapped. Pass 1 combines float64
Chan/Welford mean and M2 statistics, pass 2 performs bounded incremental
full or low-rank basis updates on the selected device, and pass 3 projects
sample batches into CPU NumPy scores. No feature, standardized-feature,
component, score, KMeans-label, or t-SNE-coordinate cache is written. Its
`step_00_run_pca.py` entry calls stages 1 through 7.

The leading 32 PCA score columns are shared by KMeans, nearest-neighbor
purity, silhouette scoring, and t-SNE by default. KMeans receives no labels
and creates one diagnostic partition per physical class present in the
dataset. Physical classes are introduced only after the geometric fits for
reporting and for the colors and markers in `pca_physical_classes.png` and
`tsne_physical_classes.png`. The remaining output, `metrics.txt`, contains
run settings, dynamic sample counts, variance, KMeans diagnostics, and local
physical-label purity. The historical neighbor metric is preserved: one
model uses the training scores as its reference for every explicit split,
training queries exclude themselves, and the resulting label-agreement
fractions are reported by split and physical class. An unsplit dataset uses
all samples as the reference and excludes each query itself.

Full mode performs an exact reduced SVD for each update. Retaining only the
configured rank between several updates makes the overall result
incremental/truncated; it matches monolithic reduced SVD only when all rows
enter one update. Single-device low-rank mode seeds PyTorch once before its
incremental randomized updates; feature-sharded low-rank mode retains an
oversampled basis through small row-Gram eigensolves. The default sample
batch size is 16 and the command can override it with `--batch-size`.

### `src.cnn`

This package owns CNN data loading, timestep splitting, training-only
normalization, topology scaling, the raw/Hermite `Conv3d` model, loss,
evaluation, checkpoint, and the two training outputs. Its training path is
`step_00_train_cnn.py` through `step_08_save_cnn.py`;
`load_cnn_checkpoint.py` is the reusable inference loader.

### `src.autoencoder`

This package owns raw/Hermite reconstruction data, the 2-D/3-D model,
timestep-aware training, per-sample normalized-space reconstruction
evaluation, checkpoint, and the four training outputs. Its path is
`step_00_train_autoencoder.py` through
`step_07_save_autoencoder.py`; `load_autoencoder_checkpoint.py` reconstructs
the current saved model directly. The raw autoencoder keeps its existing
middle-`vy` two-dimensional input in an autoencoder-owned preparation path;
this plane does not define the raw PCA or CNN representation. Physical class
labels are joined only after final reconstruction to produce split and class
tables in `metrics.txt`; they never enter training or checkpoint selection.
For `reconstruction_examples.png`, inverse preprocessing restores the raw
plane to physical phase-space density and delegates its original and
reconstruction panels to the extraction Stage 6 VDF preparation and drawing
owners. The pair shares logarithmic limits, physical km/s axes, transparent
masks, and white axes. Signed Hermite coefficients retain their symmetric
display convention on white axes.

### `src.prediction`

This package owns checkpoint-driven coordinate and region prediction. It
resolves source data, builds the checkpoint representation, performs bounded
inference, inverse-scales topology outputs, and saves structured CSV results
through one common row builder and writer. Coordinate prediction writes its
one-row CSV before optional rendering; region prediction streams per-timestep
rows in stable CID order.
For a raw checkpoint it resamples the complete physical `(vx, vy, vz)` VDF
to the training velocity grid before the full-volume logarithmic transform
and `Conv3d` inference.
The public coordinate and region orchestrators call shared stages 1 through
5 for checkpoint loading, source loading, input preparation, inference, and
output saving.

### `src.plotting`

This package owns reusable current-format plotting. Dataset plotting
uses one renderer for extraction Stage 6 and the standalone command. It
combines an x-velocity VLSV background with black magnetic streamlines,
yellow candidate VDF cells, blue source coordinates, and red selected
dataset cells. Selection-area rendering reuses the exact current X and O
physical/manual geometry that owns sample selection. One peak-plane
extractor and renderer provide physical `xz`, `xy`, and `yz` cuts to the
standalone command, while Stage 6 always requests `xz`. Frame animation owns
only ffmpeg command construction and encoding of already rendered colormap
and raw-VDF PNGs. The separate Hermite coefficient renderer memory-maps the
saved cube, groups rows by CID and timestep, and plots the signed
`coefficients[:, 0, :]` plane without recomputation or animation.
Prediction plotting combines a fixed x-velocity spatial background,
predicted-class markers, and representative source VDFs without rerunning the
model. The shared renderer prefers `<population>/vg_v`, `<population>/V`,
`vg_v`, or `V`, then derives legacy velocity from `rho_v / rho`. Coordinate
maps distinguish the compact requested star from the compact selected VDF
cell, retain the CID/class/probability title, and draw scalar X/O distances as
unfilled circles. Region maps contain only compact predicted-class markers
for displayed VDF representatives: they omit distance circles, adjacent class
text, and per-representative title lines. A concise region title and a
dedicated class-legend axes remain below the map. Vector values stay in CSV
without arrows, endpoints, or connectors.

### `src.configuration`

The configuration boundary loads YAML. Each pipeline then accesses the
expected current settings it owns and passes them through its ordered
stages.

## Focused module boundaries

Large workflows retain one public orchestration entry point while their
stages live in modules that own one sequential operation:

- dataset extraction separates configuration resolution, sample planning,
  per-timestep extraction, metadata construction, velocity-grid operations,
  staging, final dataset writing, optional frame rendering, and optional
  animation encoding;
- point physics separates flux-grid input, critical-point selection,
  selection geometry and metadata, island contours, and topology values;
- the Hermite calculation separates rotation-frame construction, basis and
  coefficient projection, and transform orchestration;
- PCA stages open a memory-mapped representation, accumulate scaling
  statistics, update PCA incrementally, project scores, fit internal KMeans
  and nearest-neighbor searches, calculate physical metrics, fit label-blind
  t-SNE, and save two physical-class plots plus `metrics.txt`;
- CNN stages load data, split timesteps, scale inputs and topology, build
  the model, calculate loss, optimize, evaluate, and save;
- autoencoder stages load data, split timesteps, scale inputs, build,
  optimize, evaluate, and save;
- prediction stages load a current CNN, open source VDFs, prepare model
  input, run inference, and save coordinate or region outputs; and
- prediction figures separate the fixed x-velocity background,
  predicted-class markers and labels, peak-centred VDF cuts, and figure
  composition.

These are responsibility boundaries rather than compatibility layers.
Callers use the current module paths directly, and no implementation is
duplicated behind forwarding modules.

## Dependency direction

The intended direction is:

```text
scripts
    -> pipeline entry points
        -> representations / data / physics / plotting
            -> NumPy arrays and external readers
```

Additional constraints are:

- data and physics never import model packages;
- PCA does not depend on CNN or autoencoder code;
- training does not depend on prediction orchestration;
- prediction does not own training logic;
- scripts contain no reusable scientific or numerical implementation;

These constraints keep scientific calculations usable in isolation and
avoid reverse dependencies from extraction into machine learning.

## Scripts and configurations

The eight commands are grouped under:

```text
scripts/data/
scripts/data/plotting/
scripts/analysis/
scripts/models/
scripts/prediction/
```

They are invoked with `python -m`. A script parses arguments, loads at most
one configuration, calls one package entry point, and prints a concise
summary. There is no per-script path modification.

The six configurations remain:

```text
configs/data/extraction.yaml
configs/analysis/pca.yaml
configs/models/cnn.yaml
configs/models/autoencoder.yaml
configs/prediction/coordinate.yaml
configs/prediction/region.yaml
```

Each functionality guide documents the configuration it owns.

## Output ownership

Dataset writing uses a hidden same-parent staging directory. Parallel
timestep workers write only raw and optional Hermite memory maps beneath a
staging-local temporary directory; the parent merges ordered results with a
monotonically increasing row offset, writes metadata at the same indexes,
flushes and closes the final staged arrays, and uses an
ordinary directory rename. Other workflows create their output directories
and save the artifacts they own directly. PCA is report-only: it saves
`pca_physical_classes.png`,
`tsne_physical_classes.png`, and `metrics.txt`, with no PCA, KMeans, or
t-SNE arrays, CSV tables, or explained-variance figure.

Optional dataset frames and animations are created only after the core
dataset has been renamed into place. They live under `plots/` and
`animations/` and are not read by PCA, CNN, autoencoder, or prediction
workflows.

Checkpoints contain the preprocessing and model data used directly by
inference. CNN training saves only `model.pt` and its consolidated
`metrics.txt` report. Autoencoder training saves `autoencoder.pt`,
`metrics.txt`, `training_history.csv`, and one consolidated
`reconstruction_examples.png` generated from the restored best model.
