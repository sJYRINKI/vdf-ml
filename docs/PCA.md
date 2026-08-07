# PCA

## Purpose

The analysis compares label-blind structure in either current VDF
representation with saved physical classes. Labels are not features or
targets for scaling, PCA, KMeans, nearest-neighbor search, or t-SNE. The
number of physical classes determines only the requested KMeans partition
count.

```text
read-only raw or Hermite memory map
    -> pass 1: batched float64 mean and M2 accumulation
    -> pass 2: batched standardization and incremental PyTorch PCA updates
    -> pass 3: batched projection onto the retained components
    -> internal KMeans and nearest-neighbor diagnostics on leading scores
    -> label-blind t-SNE on the same leading scores
    -> physical-class-only PCA and t-SNE figures
```

KMeans partition IDs remain in memory as internal diagnostics. They are not
saved, plotted, or treated as another classification. Physical classes from
`metadata.csv` are the only plot colors, markers, and legend categories.

## Raw features

Raw analysis uses the complete `raw` VDF:

```text
X.npy sample [vx, vy, vz]
    -> apply the positive log_eps floor to every velocity-space cell
    -> log10
    -> stable C-order flattening of all vx * vy * vz cells
```

The source VDF remains unchanged. No velocity plane is sliced, projected,
averaged, cropped, or downsampled.

## Hermite features

Hermite analysis uses the complete saved dimensional coefficient volume.
Unrotated arrays have axes `[n_x, n_y, n_z]`; optionally rotated arrays have
`[n_parallel, n_perp1, n_perp2]`. PCA derives the actual order and feature
count from `X_hermite.npy`, retains every coefficient, and flattens the cube
in stable C order without logarithms, sparsity thresholds, slicing,
downsampling, or per-sample normalization.

The coefficients come from the physical linear VDF with endpoint velocity
coordinates, a physically normalized physicists' Hermite basis, and
`dv**3` quadrature. Hermite datasets created under the retired compact-log,
fixed-order convention must be regenerated before current PCA analysis.

## Numerical workflow

PyTorch is the only current scaling and PCA decomposition backend. The
saved representation stays in its read-only NumPy memory map, and only the
configured number of samples is transformed at a time. This avoids the
complete in-memory feature matrix that caused a production raw-PCA attempt
to request too much memory. The three passes preserve raw/Hermite
feature definitions, flattening, and saved sample order.

The device setting accepts `auto`, `cpu`, `cuda`, or an indexed CUDA device
such as `cuda:0`. `auto` selects CUDA when PyTorch reports it available and
otherwise selects CPU. Only standardization and PCA use this device.

`pca.batch_size` is the number of consecutive saved samples prepared as one
source batch. The shipped value is 16 and the command-line `--batch-size`
option overrides it; examples may use 8 for a smaller per-update working
set. The first component update may accumulate several
source batches until it has enough rows for the retained working rank.
Smaller batches reduce the ordinary per-batch working set but perform more
updates. Because each update retains only a bounded basis, changing the batch
size can change the incremental numerical result; it does not change which
physical VDF or Hermite values are used.

Batch size does not shrink the retained component basis. The current batch, update matrix,
and PyTorch decomposition workspaces require additional memory, so lowering
`--batch-size` reduces only the batch-dependent part of the peak.

### Feature-sharded multi-GPU execution

Set `pca.multi_gpu: true` or pass `--multi-gpu` only when launching the
command with one `torchrun` process per visible GPU on a single node. Rank
`r` of `w` owns the contiguous flattened interval

```text
feature_start = n_features * r // w
feature_stop  = n_features * (r + 1) // w
```

Every rank visits the same saved sample rows in the same order, but copies,
transforms, standardizes, and stores only those local feature columns. The
complete raw VDF and Hermite coefficient definitions are preserved by the
union of the non-overlapping intervals; no rank constructs a complete
feature row or component basis.

For each incremental update, a rank forms its local update columns and the
small row-Gram contribution `local_update @ local_update.T`. NCCL sums these
contributions with `all_reduce`, giving every rank the global row-Gram
matrix. Rank zero solves that replicated small symmetric eigenproblem and
broadcasts the small eigenpairs so every rank reconstructs its local
right-singular-vector columns from one consistent left basis. Component
signs are selected from the global largest loading using only small
per-component orientation values. During pass 3, rank-local score
contributions are reduced to rank zero. Only rank zero retains the final
score matrix and runs KMeans, nearest-neighbor reporting, t-SNE, plots,
metrics, and output writes.

Distributed `full` retains `n_components` after each exact row-Gram update.
Distributed `lowrank` uses the same exact row-Gram solve while retaining
`n_components + lowrank_oversampling` rows between updates and truncating at
the final boundary. The latter does not call `torch.pca_lowrank` and does not
use `lowrank_niter`; those remain settings for the unchanged single-device
randomized low-rank path. Both distributed methods remain incremental and
batch-size dependent because the retained basis is truncated between
updates. Multi-node execution is not supported.

### Pass 1: feature statistics

Stage 2 reads every sample batch and combines its feature-column statistics
with the running statistics using the Chan/Welford equations. The running
count, mean, and sum of squared deviations (`M2`) are float64 even when the
saved representation is float32. Population scaling is then:

```text
variance = M2 / n_samples
scale = sqrt(variance)
scale = 1 where scale is zero
```

Statistics are label-blind. Mean, M2, and scale are working values only;
they are not saved as scaler or cache artifacts.

### Pass 2: incremental component updates

Each source batch is transformed again, standardized with the pass-1
statistics, and moved to the selected device. After the first decomposition,
the next update matrix is formed from the retained weighted right singular
vectors and the next standardized sample batch:

```text
update = concatenate(S[:, None] * Vh, standardized_batch, axis=0)
```

Only this bounded update matrix and the retained decomposition need to be on
the selected device. The implementation does not save a feature matrix, a
standardized matrix, or a component cache.

### Full PCA

`method: full` applies exact reduced SVD to every incremental update:

```python
U, S, Vh = torch.linalg.svd(
    update,
    full_matrices=False,
)
```

For `n_components = k`, each update retains:

```text
S = S[:k]
Vh = Vh[:k]
```

The SVD within an update is exact and never constructs a feature-by-feature
covariance matrix. When every row fits in the first update, the result is the
ordinary exact reduced SVD of the complete standardized matrix. With
multiple updates, intermediate rank truncation makes the final result an
incremental retained-rank approximation; it is not generally identical to a
monolithic full-matrix PCA.

### Single-device low-rank PCA

On one device, `method: lowrank` uses randomized `torch.pca_lowrank` for the
same incremental updates. The working rank is derived rather than configured
directly:

```text
working_rank = n_components + lowrank_oversampling
q = min(working_rank, min(update.shape))
```

Initial sample batches are accumulated until enough rows are available for
the working rank. PyTorch's random seed is set once before the incremental
low-rank pass, rather than reset for every update. `lowrank_niter` sets the
number of power/subspace iterations:

```python
U, S, V = torch.pca_lowrank(
    update,
    q=q,
    center=False,
    niter=lowrank_niter,
)
```

Oversampling adds randomized working vectors beyond the saved component
count. Low-rank mode is approximate and incremental; it is not an automatic
fallback or selector.

### Variance and component orientation

Both methods calculate retained variance from the final singular values and
accumulate the complete standardized energy while visiting pass-2 batches:

```text
explained_variance = S[:k]^2 / (n_samples - 1)
total_variance = sum(standardized_batch.square()) / (n_samples - 1)
explained_variance_ratio = explained_variance / total_variance
```

Low-rank mode does not run a hidden full SVD to calculate the denominator.

Singular vectors have arbitrary signs. Both methods use the same
deterministic orientation: for each component, find the loading with the
largest absolute magnitude and choose the sign that makes that loading
nonnegative. An exactly zero selected loading uses positive one. The same
sign multiplies the component and its score column, so the PCA subspace is
unchanged while the in-memory scores have a stable orientation.

### Pass 3: scores

The source memory map is read a third time. Each batch receives the same
transform and pass-1 standardization, then is projected onto the final,
deterministically oriented components. The complete score array and
explained-variance ratios are CPU NumPy arrays retained only for downstream
analysis and reporting. KMeans, nearest-neighbor search, and t-SNE all use
the first `analysis_components` score columns, limited by the number of
retained components. The shipped downstream width is 32.

KMeans creates one diagnostic partition per physical class present in the
current metadata, but receives only PCA scores during fitting. Physical
labels are introduced afterward to calculate:

- composition counts for every physical class inside each partition;
- dominant-class purity for every partition and weighted purity overall;
- silhouette score in the same leading-PCA space;
- the partition containing the largest share of each physical class; and
- local physical-label agreement among the 25 nearest PCA-space neighbors.

Nearest-neighbor search excludes each sample itself. Labels are compared only
after score-space neighbors have been found, and the report groups the
per-sample agreement by metadata split and physical class. With explicit
splits, the historical metric fits its reference on training PCA scores and
queries training, validation, and test rows against that same reference;
training queries remove their own reference row. A dataset without split
metadata uses all score rows as the reference, assigns one in-memory `all`
reporting group, and removes each query itself. No split is invented or
written back.

t-SNE remains label-blind and is never used to choose KMeans partitions. The
large-dataset defaults are perplexity 50, PCA initialization, automatic
learning rate, early exaggeration 16, and 2000 iterations. Both figures use
point area 3; PCA alpha 0.30 and t-SNE alpha 0.28 reduce opaque overplotting.
They do not move classes, create
class-dependent offsets, or guarantee disconnected islands. Continuous CID
or timestep tracks may remain, and gaps or distances in t-SNE space are not
quantitative physical distances.

Physical-class tables and plot legends use the stable numeric IDs present in
`metadata.csv`; they do not hard-code a class count or sort names
alphabetically. The current project order is `lobe`, `current_layer`,
`o_point`, `x_point`, `solar_wind`, `magnetosheath`, and
`inner_magnetosphere`. These labels affect only post-fit descriptions and
colors.

## Ordered modules

The analysis package reads in execution order:

```text
step_00_run_pca.py
step_01_load_pca_features.py
step_02_scale_pca_features.py
step_03_fit_pca.py
step_04_fit_kmeans.py
step_05_calculate_physical_metrics.py
step_06_fit_tsne.py
step_07_save_pca_outputs.py
distributed_pca.py
```

The stage-0 entry owns orchestration only. Each later stage receives the
previous numerical result and passes its result to the next stage.
`distributed_pca.py` owns only the optional one-node process group,
feature-sharded statistics, row-Gram updates, orientation, and score
reduction; the ordinary ordered single-device stages remain unchanged.

## Configuration

`configs/analysis/pca.yaml` is owned by:

```text
python -m scripts.analysis.run_pca_analysis
```

Its sections are:

| Section | Current fields |
|---|---|
| `random_state` | reproducibility seed |
| `analysis_components` | leading retained PCA columns shared by KMeans, silhouette, nearest neighbors, and t-SNE |
| `nearest_neighbors` | score-space neighbors used for local physical-label purity |
| `raw` | positive `log_eps` used as the complete-volume lower bound |
| `pca` | `method`, `device`, `multi_gpu`, `batch_size`, `n_components`, `lowrank_oversampling`, `lowrank_niter` |
| `kmeans` | `n_init`, `max_iter`, and algorithm for the internal diagnostic partition |
| `tsne` | output width, perplexity, learning rate, early exaggeration, iterations, initialization, marker size, and alpha |
| `plot` | output DPI |

The supplied configuration uses:

- random seed 1234;
- low-rank PCA on the automatically selected device;
- feature-sharded multi-GPU execution disabled;
- 16 samples per PCA batch;
- 64 retained PCA components;
- 5 low-rank oversampling vectors and 4 low-rank subspace iterations when
  low-rank mode is selected;
- 32 leading PCA components shared by downstream diagnostics;
- 25 neighbors for local physical-label purity;
- one KMeans partition per physical class present, with `n_init=20`,
  `max_iter=300`, and Lloyd's algorithm;
- two-dimensional t-SNE on the same leading 32 PCA scores;
- perplexity 50, automatic learning rate, PCA initialization, early
  exaggeration 16, and 2000 iterations;
- point area 3, PCA alpha 0.30, and t-SNE alpha 0.28; and
- plot DPI 150.

`analysis_components` is one shared downstream width; KMeans,
nearest-neighbor search, silhouette scoring, and t-SNE do not have separate
component-count settings. These operations remain on CPU in scikit-learn.

Comments in `configs/analysis/pca.yaml` are the quickest reference for each
accepted setting and its numerical or output effect.

Example adjustment:

```yaml
pca:
  method: lowrank
  device: auto
  multi_gpu: false
  batch_size: 8
  n_components: 64
  lowrank_oversampling: 5
  lowrank_niter: 4
analysis_components: 32
nearest_neighbors: 25
tsne:
  perplexity: 50
  init: pca
  early_exaggeration: 16.0
  max_iter: 2000
```

The KMeans cluster count is derived directly from the number of physical
classes present. Physical labels do not enter KMeans features or targets.

## Commands

See [Example commands](EXAMPLE_COMMANDS.md#pca-analysis) for verified local
and Slurm variants.

Select either `raw` or `hermite` and use a separate output directory for
each representation.

The selected representation file and metadata are loaded directly. The
workflow assumes they follow the current dataset layout.

The command-line options `--pca-method {full,lowrank}`, `--device DEVICE`,
`--multi-gpu`, `--batch-size N`, `--analysis-components N`, and
`--nearest-neighbors N`
override their YAML settings without adding a second PCA command. Full and
low-rank modes accept either raw or Hermite features.

Performance depends on sample count, feature count, retained component
count, selected dtype, device-transfer cost, and GPU hardware. Low-rank PCA
is not always faster, and GPU execution is not always faster. Measure the
actual representation and machine before choosing a production profile.
Complete raw volumes can require substantially more host and accelerator
memory than Hermite coefficients. The three-pass design bounds work by
sample batch and retained rank, but one transformed sample, the float64
statistics, the retained basis, and an update workspace must still fit.
Start with a smaller `--batch-size` for full-resolution raw data and use
low-rank PCA when only a small leading subspace is required, without assuming
either choice will always fit or run faster.

## Output

Each result contains:

```text
pca_physical_classes.png
tsne_physical_classes.png
metrics.txt
```

The two figures color and mark samples only by their physical class. The PCA
figure shows PC1 versus PC2, including their explained-variance percentages.
The t-SNE figure shows the label-blind embedding with the same class styles.
Neither figure displays KMeans IDs.

`metrics.txt` contains, in order, the run settings, dataset summary, sample
counts by split and class, explained and cumulative variance, KMeans purity
against physical classes, physical-class concentration in KMeans partitions,
and nearest-neighbor label purity. PCA scores, explained-variance arrays,
KMeans labels, t-SNE coordinates, CSV tables, and an explained-variance plot
are not persisted. Full and low-rank PCA use this same three-file contract;
no PCA model, scaler, feature matrix, standardized matrix, component cache,
backend metadata, or method-specific artifact is added.

## Explicit exclusions

The PCA workflow has no:

- topology, spatial, or physical-moment features;
- class balancing, class-aware sampling, class weights, or filtering;
- class-aware PCA, nearest-neighbor, KMeans, or t-SNE fitting;
- SOM or DBSCAN;
- rotated public representation;
- PCA- or CNN-embedding sample removal;
- per-class PCA or t-SNE fit;
- automatic full/low-rank method selection; or
- an alternative PCA decomposition backend.
