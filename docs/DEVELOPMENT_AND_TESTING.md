# Development and testing

## Development principles

The code should remain readable to a physicist following the numerical data
flow.

- Preserve units, axes, coordinate conventions, timestep logic, class
  meanings, target order, and sample order.
- Prefer bounded batches, memory maps, streaming, preallocation, and
  vectorized NumPy operations.
- Profile before adding parallelism or changing numerical algorithms.
- Keep scripts limited to argument parsing, configuration loading, one
  package call, and concise output.
- Keep physics independent from machine-learning packages.
- Do not duplicate representation, checkpoint, split, or staging logic.
- Use direct descriptive names rather than compatibility layers.
- Let ordinary Python and library exceptions propagate from runtime
  workflows without project-specific translation.

Every active Python file has one primary responsibility. Files below
`src`, `scripts`, and `tests` have a hard maximum of 500
implementation lines. Long functionality must be divided by a real
scientific stage, workflow stage, or owned artifact; arbitrary numbered
pieces and vague catch-all filenames are prohibited. Genuinely linear
workflows use zero-padded descriptive names such as
`step_03_scale_cnn_inputs.py`.

The source-structure checker parses Python with `ast`. It counts every
physical line, including imports, definitions, decorators, comments, blank
lines, constants, and disabled code. It subtracts only lines occupied by
recognized module, class, function, or asynchronous-function docstrings.
Large comments, unrelated string literals, and blank padding therefore do
not reduce the implementation count.

Active modules and public APIs use informative NumPy-style docstrings.
Parameter names must match signatures, meaningful returns must be described,
and scientific APIs must record relevant units, shapes, axes, ordering,
assumptions, and side effects. Write for an external developer who does not
know the refactoring history: explain what the symbol does, why it is a
separate workflow step, which step precedes and follows it, and what it
reads, returns, mutates, or writes. Avoid stock summaries such as “Load
data” or “Save results.”

Use simply “CNN” for the current model and workflow. Describe its storage
mechanism as sample-wise loading, on-demand sample reading, or memory-mapped
sample access when that implementation detail matters.

## Multi-user worktrees

Before editing:

```bash
git status --short
git log -3 --oneline
git diff --name-only
git diff --stat
```

Inspect existing diffs and preserve unrelated hunks. Do not reset, restore,
clean, or overwrite work owned by another user. Generated smoke output
belongs in a new unique `/tmp/vdf-ml-*` location.

## Full test suite

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
```

Report passed, failed, and skipped counts and explain every skip. Synthetic
results verify software behavior only.

## Focused checks

Useful bounded test groups include:

```bash
python -m pytest -q tests/test_dataset_loading.py

python -m pytest -q \
    tests/test_vdf_transform.py \
    tests/test_physical_context.py

python -m pytest -q \
    tests/test_pca_features.py \
    tests/test_pca_estimators.py \
    tests/test_pca_analysis.py \
    tests/test_pca_cli.py \
    tests/test_distributed_pca.py \
    tests/test_distributed_pca_cuda.py \
    tests/test_slurm_pca_jobs.py

python -m pytest -q \
    tests/test_cnn_data_and_scaling.py \
    tests/test_cnn_model_and_loss.py \
    tests/test_cnn_timestep_split.py

python -m pytest -q \
    tests/test_cnn_optimization_and_evaluation.py \
    tests/test_cnn_model_parallel.py \
    tests/test_cnn_training_workflow.py

python -m pytest -q \
    tests/test_autoencoder_model_and_data.py \
    tests/test_autoencoder_training_workflow.py \
    tests/test_autoencoder_checkpoint_loading.py

python -m pytest -q \
    tests/test_prediction_checkpoint_runtime.py \
    tests/test_prediction_preprocessing.py \
    tests/test_prediction_coordinate_workflow.py \
    tests/test_prediction_region_workflow.py \
    tests/test_prediction_plot_geometry.py \
    tests/test_prediction_vdf_cuts.py \
    tests/test_prediction_visualization.py
```

Extraction behavior should be checked with deterministic synthetic
writer/extraction fixtures rather than a production source run.

Optional extraction postprocessing has focused successful-path coverage:

```bash
python -m pytest -q \
    tests/test_point_selection.py \
    tests/test_dataset_creation_plots.py \
    tests/test_dataset_creation_animations.py \
    tests/test_dataset_vdf_frames.py \
    tests/test_point_selection_area_plotting.py
```

The tests keep ffmpeg subprocess execution mocked while checking exact
manual/physical/consensus/union selection, empty and cross-kind conflict
omission, deterministic frame order, per-cell-ID grouping, exact selection
artists, and animation command construction. When ffmpeg is installed, a bounded
synthetic animation may additionally be encoded under a unique `/tmp`
directory.

## Source structure and docstrings

Run the permanent architecture and documentation policy independently:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q tests/test_source_structure.py
```

The test reports physical, recognized-docstring, and implementation line
counts for any file above the limit. It also checks module and public API
docstrings, NumPy-style section underlines, signature parameter coverage,
meaningful returns, descriptive filenames, ordered `step_XX_` workflow
stages, rejection of arbitrary numbered splits, external-developer
explanations for nontrivial callables, vague summary phrases, and direct CNN
terminology.

Print the complete sorted count table without running other tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -c \
  "from tests.source_structure import print_source_structure_table; print_source_structure_table('.')"
```

## Python compilation

Compile project sources while directing bytecode outside the repository:

```bash
PYTHONPYCACHEPREFIX=/tmp/vdf-ml-pycache \
python -m py_compile $(find src scripts -name '*.py' \
    -not -path '*/__pycache__/*')
```

Run `pyflakes` or another static checker only when already installed. Do not
add a dependency solely for a final check.

## Command help

All public commands must provide help:

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

## Configuration checks

Load all six YAML files through the project loader:

```bash
python -c "from src.configuration import load_config; \
[load_config(path) for path in (
'configs/data/extraction.yaml',
'configs/analysis/pca.yaml',
'configs/models/cnn.yaml',
'configs/models/autoencoder.yaml',
'configs/prediction/coordinate.yaml',
'configs/prediction/region.yaml')]"
```

Pipeline-specific tests exercise successful current configuration values.

## Slurm jobs

The 30 site-specific jobs under `slurm/turso/` and `slurm/hile/` pass CLI
arguments directly to the corresponding current module. The four additional
GPU jobs provide feature-sharded PCA and model-parallel CNN launchers for
both sites while preserving every single-device wrapper.

Check every nested job file without invoking a workflow:

```bash
find slurm/turso slurm/hile -type f -name "*.sbatch" -print0 \
    | xargs -0 -n1 bash -n
```

Check command construction without calling `srun` or submitting a job:

```bash
VDF_ML_DRY_RUN=1 \
bash slurm/turso/cpu/train_cnn.sbatch \
    --config configs/models/cnn.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model \
    --representation raw
```

Repeat dry runs for both site trees and CPU/GPU variants. A GPU dry run
should show the wrapper-injected `--device cuda`; a CPU dry run should show
the wrapper-injected `--device cpu`.
The permanent Slurm tests exercise all 30 jobs:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q tests/test_slurm_scripts.py
```

See [Example commands](EXAMPLE_COMMANDS.md) for local and scheduler
invocations, site profiles, log locations, and operational commands for all
eight workflows. The wrappers do not create temporary directories.

## Saved dataset loading

Exercise direct loading with the focused dataset tests:

```bash
python -m pytest -q tests/test_dataset_loading.py
```

Normal PCA, CNN, autoencoder, prediction, and plotting paths should read only
the saved files needed by their operation.

## Cheapest subsystem smoke tests

- Extraction: deterministic synthetic source/writer fixtures.
- Dataset plotting: bounded colormap, per-cell-ID raw-VDF frames, and signed
  per-cell-ID Hermite frames under a new `/tmp` dataset, including the
  x-velocity background, role-colored points, exact X/O selection artists,
  and the saved `coefficients[:, 0, :]` plane.
- Dataset animation: mocked ffmpeg command construction, plus one tiny
  synthetic encoding when ffmpeg is available.
- PCA: raw and Hermite with full and low-rank methods on a small saved
  dataset; include CUDA when it is available. Exercise dynamic KMeans
  partition counts, physical-class purity/concentration, nearest-neighbor
  purity, label-blind t-SNE, and the exact three-file output contract.
- CNN: deterministic synthetic multi-timestep raw and Hermite training.
- Autoencoder: deterministic synthetic multi-timestep raw and Hermite
  reconstruction.
- Prediction: one coordinate and one-cell or very small region.
- Plotting: decode PNGs, then inspect representative images when image
  viewing is available.
- Hermite physics: compare endpoint axes, drift, thermal velocity, normalized
  physicists' basis, and `dv**3` coefficient projection against an
  independent literal physical-VDF implementation. Exercise unrotated and
  optionally rotated nondefault orders.

Do not run production extraction, full scientific training, large-region
prediction, or production-scale Hermite work as a routine smoke check.

## Timestep leakage checks

Training tests must verify:

- complete-timestep partitions;
- excluded gaps;
- no sample or timestep overlap;
- training-only feature normalization;
- training-only topology scaling for CNN;
- no test-set influence on model selection; and
- no sample-level fallback for insufficient timesteps.

The previously used BCH verification dataset contained one timestep.
Bounded forward execution is appropriate; scientific training is not.

## Memory checks

Use `/usr/bin/time -v` for bounded process measurements when practical.
Record representation, device, PCA batch size, model batch size,
normalization batch size, elapsed time, and maximum RSS. A previous
monolithic full-resolution raw-PCA attempt requested approximately 844 GiB;
the current PCA regression must demonstrate bounded batches rather than
attempting that production allocation.

Expected behavior:

- saved arrays remain memory mapped where supported;
- raw PCA batches retain and flatten every `(vx, vy, vz)` voxel;
- PCA reads three passes in stable sample order: float64 Chan/Welford
  statistics, incremental component updates, and final score projection;
- no complete feature or standardized-feature matrix is materialized and no
  feature, standardized-feature, or component cache is written;
- `pca.batch_size` defaults to 16 and `--batch-size` overrides it;
- full PCA matches monolithic reduced SVD when all rows enter one update,
  while multi-batch retained-rank updates follow the specified incremental
  truncation;
- single-device low-rank PCA seeds PyTorch once before its randomized
  incremental update pass, while distributed low-rank PCA retains the
  oversampled rank through exact row-Gram updates;
- feature-sharded PCA divides columns with
  `n_features * rank // world_size` boundaries, reduces row-Gram matrices
  and score batches, broadcasts the small Gram eigenpairs, exchanges only
  per-component orientation candidates, and leaves the complete basis
  unassembled;
- raw CNN batches retain `(batch, 1, vx, vy, vz)` for `Conv3d`;
- model-parallel CNN places consecutive stages across visible GPUs in one
  process and retains no replicated model copy;
- the raw autoencoder retains its existing two-dimensional reconstruction
  plane independently of the PCA/CNN input path;
- normalization uses bounded accumulation;
- the parent streams the first nonempty extraction timestep for shape
  discovery; `extraction_n_jobs: 1` remains serial, while larger values use
  timestep workers for remaining raw-only or paired raw-plus-Hermite output;
- each timestep worker reuses one reader and extractor, processes samples
  sequentially into worker-local memory maps, and never writes the final
  staged arrays; the parent merges raw, Hermite, and metadata rows at the
  same planned offsets regardless of completion order;
- serial and parallel saved raw, Hermite, metadata, and velocity-grid
  outputs are exactly equal for the same planned extraction;
- Hermite consumers derive complete volume shape and order from the saved
  array or checkpoint and handle one full sample at a time;
- region predictions stream rows;
- representative plotting rereads only bounded VDF data; and
- repeated samples or epochs do not cause retained-memory growth.

Missing CUDA is not a failure. Verify CPU behavior and report CUDA as
unavailable.

## Plot checks

Generated PNGs must decode, have nonzero dimensions and file size, and use
the intended coordinates and representation. Inspect:

- physical-class-only PCA and t-SNE plots, including the shared class colors
  and markers, size-3 rasterized points, PCA alpha 0.30, t-SNE alpha 0.28,
  and legends without KMeans IDs;
- standalone and extraction-time dataset colormaps;
- cell-ID-organized extraction VDF frames;
- cell-ID-organized signed Hermite frames from `X_hermite.npy`;
- single- and multi-sample VDF-slice figures; and
- raw/Hermite combined coordinate and region figures.

Prediction tests should require the exact threshold producer order
`MinValue`, `<population>/MinValue`,
`<population>/EffectiveSparsityThreshold`, then
`<population>/vg_effectivesparsitythreshold`, and confirm the selected name
is reused for several CIDs from one prepared source. Coordinate output tests
require one common-schema `predictions_<timestep>.csv` row before optional
plotting and no JSON sidecar. Region tests retain the per-timestep CSV
contract.

Combined-figure artist checks should require all four direct-velocity
candidates plus legacy `rho_v / rho`, the symmetric x-velocity background,
`[-30, 30, -15, 15]` x-z view, a dedicated legend axes below the map, the
coordinate-only requested star with area 12 and linewidth 0.4, and selected
prediction markers with area 4 and linewidth 0.3. Coordinate figures retain
their CID/class/probability title and scalar X/O circles. Region figures must
retain predicted-class marker shapes and colors but have no distance-circle
patches or legend labels, adjacent class text, representative CID/class/
probability title lines, or requested-coordinate star. No topology arrow,
inferred endpoint, or connector artist should remain. The class legend stays
below both maps, with independently readable legend symbols.

Raw autoencoder reconstruction checks should compare the original artist to
the shared extraction Stage 6 VDF preparation and drawing path. The raw
original/reconstruction pair must share physical km/s axes, threshold masks,
one `LogNorm`, unmodified `nipy_spectral`, transparent masks over white axes,
and the same fixed x-z plane; the error panel remains separate and
nonnegative. Hermite original/reconstruction panels retain their symmetric
signed coefficient scale, and every Hermite axes background remains white.
Extraction Stage 6 and standalone VDF plots must remain numerically and
visually unchanged.

Hermite Stage 6 checks must require one directory per numeric CID, stable
timestep/sample-index ordering, `coefficients[:, 0, :]`, symmetric `RdBu_r`
limits, white axes, a coefficient colorbar, and unrotated or rotated labels
from the saved convention. The plotting path memory-maps coefficients and
must not reopen VLSV sources, recompute the transform, normalize cubes, or
create Hermite animations.

For extraction sequences, also verify stable frame dimensions, numerical
timestep ordering, the single peak-`vy` `vx-vz` frame, one overlay per
accepted source point, the four exact X/O physical/manual area families when
their geometry exists, and the below-axes legend. Standalone VDF-slice
figures must default to all three peak-centred planes when `--plane` is
omitted, preserve explicit plane order, and render `--plane xz` with the same
plane data and style as Stage 6.

PCA output tests must require exactly `pca_physical_classes.png`,
`tsne_physical_classes.png`, and `metrics.txt`. The metrics report covers
sample counts, explained and cumulative variance, internal KMeans purity,
weighted purity, silhouette score, physical-class concentration, and
nearest-neighbor physical-label purity. Estimator spies should confirm that
KMeans, neighbor search, and t-SNE see only the configured leading PCA
scores, never physical labels. The production defaults use 32 leading
components, 25 neighbors, and t-SNE perplexity 50 with PCA initialization,
automatic learning rate, early exaggeration 16, and 2000 iterations.

Check the consolidated CNN and autoencoder `metrics.txt` reports and the
autoencoder CSV history numerically and as tables. Autoencoder tests should
verify per-sample normalized-space MSE before aggregation; split and dynamic
physical-class tables must contain count, mean, median, minimum, and maximum.
Batch spies must confirm that class labels never enter model forward or loss.
The minimal training workflows do not write figures.

If image viewing is unavailable, report only programmatic inspection.

## Static and artifact protection

Check new whitespace:

```bash
git diff --check
```

Separate inherited findings from new ones. At completion confirm:

- no dataset, model, checkpoint, or scientific cache was overwritten;
- no unexpected artifact appeared under project data/model directories;
- generated smoke output used a unique temporary location;
- no file was staged;
- no commit or push occurred; and
- unrelated worktree hunks remain unchanged.

## Python cache cleanup

Use `PYTHONDONTWRITEBYTECODE=1` for final verification. Remove project-owned
`__pycache__`, `*.pyc`, and `*.pyo` only below `src/`, `scripts/`, and
`tests/`, then verify:

```bash
find src scripts tests \
    \( -name '*.pyc' -o -name '*.pyo' -o -name '__pycache__' \) \
    -print
```

The command should produce no output. Do not remove virtual-environment,
external, vendored, system, or user-home caches.
