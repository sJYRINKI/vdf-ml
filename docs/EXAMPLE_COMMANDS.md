# Example commands

This page is the detailed command reference for the eight current workflows.
Run every Python command from the repository root. Replace the generic paths
with locations available on the machine or cluster where the workflow runs.

## Environment

Create and activate the project environment as described in the
[root README](../README.md#installation):

```bash
cd /path/to/repository
source .venv/bin/activate
```

Extraction, prediction, and VLSV colormaps require Analysator. Install its
checked-out source into the same environment by following
[Install Analysator](../README.md#install-analysator). No project-specific
`PYTHONPATH` is required.

For noninteractive plotting, configure Matplotlib and Analysator before
starting Python:

```bash
export MPLBACKEND=Agg
export PTNOLATEX=1
export PTNONINTERACTIVE=1
```

The checked-in YAML files contain example source-file locations. Review those
paths before running VLSV-facing commands.

Each active YAML file now documents its settings directly above the
corresponding keys. Use those comments as the quickest reference for
accepted choices, physical units, coordinate order, Boolean effects, and
written files; use the subsystem guides for the scientific context.

## Slurm sites and profiles

Submit Turso files from a Turso login node and HILE files from a HILE login
node. The job files contain no Slurm cluster-routing directive, so a path
name does not transfer a job between schedulers.

None uses MPI or multiple Slurm tasks. Use application worker and batch settings to consume
only useful resources. A whole-node allocation does not guarantee a speedup,
and these workflows do not necessarily scale linearly with the core count.
Choose the CPU or GPU job according to the device on which the workflow
should run.

The complete path matrix is:

| Workflow | Turso CPU | Turso GPU | HILE CPU | HILE GPU |
|---|---|---|---|---|
| Dataset extraction | `slurm/turso/cpu/extract_dataset.sbatch` | — | `slurm/hile/cpu/extract_dataset.sbatch` | — |
| Dataset colormap | `slurm/turso/cpu/plot_colormap.sbatch` | — | `slurm/hile/cpu/plot_colormap.sbatch` | — |
| VDF slices | `slurm/turso/cpu/plot_vdf_slices.sbatch` | — | `slurm/hile/cpu/plot_vdf_slices.sbatch` | — |
| PCA | `slurm/turso/cpu/run_pca_analysis.sbatch` | `slurm/turso/gpu/run_pca_analysis.sbatch` | `slurm/hile/cpu/run_pca_analysis.sbatch` | `slurm/hile/gpu/run_pca_analysis.sbatch` |
| Feature-sharded PCA | — | `slurm/turso/gpu/run_pca_multi_gpu.sbatch` | — | `slurm/hile/gpu/run_pca_multi_gpu.sbatch` |
| CNN training | `slurm/turso/cpu/train_cnn.sbatch` | `slurm/turso/gpu/train_cnn.sbatch` | `slurm/hile/cpu/train_cnn.sbatch` | `slurm/hile/gpu/train_cnn.sbatch` |
| Model-parallel CNN | — | `slurm/turso/gpu/train_cnn_model_parallel.sbatch` | — | `slurm/hile/gpu/train_cnn_model_parallel.sbatch` |
| Autoencoder training | `slurm/turso/cpu/train_autoencoder.sbatch` | `slurm/turso/gpu/train_autoencoder.sbatch` | `slurm/hile/cpu/train_autoencoder.sbatch` | `slurm/hile/gpu/train_autoencoder.sbatch` |
| Model-parallel autoencoder | — | `slurm/turso/gpu/train_autoencoder_model_parallel.sbatch` | — | `slurm/hile/gpu/train_autoencoder_model_parallel.sbatch` |
| Coordinate prediction | `slurm/turso/cpu/predict_coordinate.sbatch` | `slurm/turso/gpu/predict_coordinate.sbatch` | `slurm/hile/cpu/predict_coordinate.sbatch` | `slurm/hile/gpu/predict_coordinate.sbatch` |
| Region prediction | `slurm/turso/cpu/predict_region.sbatch` | `slurm/turso/gpu/predict_region.sbatch` | `slurm/hile/cpu/predict_region.sbatch` | `slurm/hile/gpu/predict_region.sbatch` |

## Dataset extraction

The extraction command selects and labels VDF cells, writes full-resolution
raw arrays, and optionally writes Hermite coefficients. The output base
directory comes from `output_dirs` in `configs/data/extraction.yaml`.

Local command:

```bash
python -m scripts.data.extract_dataset \
    --config configs/data/extraction.yaml \
    --start-timestep 3072 \
    --n-timesteps 1 \
    --dataset-kind train
```

The example creates a directory named `timesteps_3072_1` below the configured
training output directory. It contains:

```text
X.npy
metadata.csv
velocity_grid.npz
X_hermite.npy       when Hermite output is enabled
```

To save the default order-22 physical-VDF Hermite representation and its
per-CID coefficient frames, use the live nested settings:

```yaml
creation:
  planning_n_jobs: 40
  extraction_n_jobs: 4

representations:
  hermite:
    enabled: true
    order: 22
    rotate: false

postprocessing:
  plot_vdfs: true
  plot_hermites: true
```

The transform projects the physical linear VDF with endpoint velocity axes,
a physically normalized physicists' Hermite basis, and `dv**3` quadrature.
It applies no logarithm or `MinValue` threshold. `planning_n_jobs` plans
independent timesteps, while `extraction_n_jobs` controls timestep workers
for raw-only or aligned raw-plus-Hermite output. The parent streams the first
nonempty timestep for shape discovery; each remaining worker owns one
timestep, reuses one reader and extractor, and processes samples
sequentially into paired temporary memory maps. The parent alone merges
those rows in planned order. Set `extraction_n_jobs: 1` for serial execution.
The value `4` is an example that must fit the allocated physical cores,
memory, and VLSV filesystem workload. A one-timestep run has no remaining
timestep to submit, so it cannot use more than one extraction worker.

Manual plasma-region coordinates in `class_coords_re` use
`[x_re, y_re, z_re]` order and Earth-radii units. Each entry maps to the
nearest VDF-carrying cell. The checked-in `solar_wind`, `magnetosheath`, and
`inner_magnetosphere` lists are intentionally empty until explicit
scientific locations are supplied; an empty list creates no samples.
`lobe` uses the same mechanism.

`current_layer` is not listed under `class_coords_re`. Stage 3 assigns it
after accepted X/O selection to remaining VDF cells in the inclusive
magnetotail sampling region `-30 <= x_re <= -10`, `|z_re| <= 1`, with no y
restriction. X/O and rejected conflict cells are excluded, and no
current-density detector participates in this assignment. The complete
class ID table is in
[Data, physics, and representations](DATA_PHYSICS_REPRESENTATIONS.md#physical-labels).

Five Boolean postprocessing settings control the derived artifacts. To
render static dataset frames during extraction, set:

```yaml
postprocessing:
  plot_colormaps: true
  plot_vdfs: true
  plot_hermites: true
  animate_colormaps: false
  animate_vdfs: false
  animation_fps: 4
  animation_format: mp4
```

Run the same local extraction command. The completed dataset then also
contains one colormap PNG per requested extraction timestep, one
peak-centred raw-VDF PNG per sample, and, when Hermite is enabled, one signed
coefficient PNG per sample, organized by cell ID:

```text
plots/
├── colormaps/
│   └── frame_<frame-index>_timestep_<timestep>.png
├── vdfs/
│   └── cid_<cell-id>/
│       └── frame_<frame-index>_timestep_<timestep>.png
└── hermite/
    └── cid_<cell-id>/
        └── frame_<frame-index>_timestep_<timestep>.png
```

Each integrated colormap uses x-directed bulk velocity with the symmetric
`bwr` range `[-1.5e6, +1.5e6] m/s`, black magnetic-field streamlines,
yellow VDF-carrying cells, blue configured or detected source coordinates,
and red selected dataset cells. Exact contributing X/O search areas follow
the shared visibility rule: overlapping union sets show only the physical
outline, while disjoint union sets show both. The compact legend is below the
spatial axes. Each per-cell-ID VDF PNG is one square LogNorm
`vx-vz` cut through that sample's own peak `vy`, with velocity axes in km/s
and a vertical `f(v)` colorbar. Its historical display preparation resolves
the source sparsity-threshold producer in Analysator-era order (`MinValue`,
the population-scoped `MinValue`, then the population-scoped
`EffectiveSparsityThreshold`, then the population-scoped
`vg_effectivesparsitythreshold`), uses
`dv = (vymax - vymin) / nvy`, multiplies a float32 plane copy by `dv`, zeros
values strictly below `threshold * dv`, and masks nonpositive values. The
unmodified `nipy_spectral` colormap uses `LogNorm` over visible support, with
the default transparent bad mask over white axes rather than forced black.
Saved VDF values are unchanged.

Each Hermite frame reads the saved cube rather than recomputing it and plots
`coefficients[:, 0, :]` with signed symmetric `RdBu_r` limits, white axes,
and a coefficient colorbar. Unrotated frames label `n_x` and `n_z` with
`n_y=0`; rotated frames label `n_parallel` and `n_perp2` with `n_perp1=0`.
There is no Hermite animation switch.

To enable the optional physical velocity-space rotation before moments and
projection, change only:

```yaml
representations:
  hermite:
    enabled: true
    order: 22
    rotate: true
```

Downstream CNN and autoencoder checkpoints store the actual saved order,
volume shape, and rotation setting. Hermite datasets and checkpoints created
under the retired compact-log, fixed-order convention must be regenerated
and retrained.

To create static frames and animations in the same run, set:

```yaml
postprocessing:
  plot_colormaps: true
  plot_vdfs: true
  plot_hermites: true
  animate_colormaps: true
  animate_vdfs: true
  animation_fps: 4
  animation_format: mp4
```

This adds:

```text
animations/
├── colormap_evolution.mp4
└── vdfs/
    └── cid_<cell-id>.mp4
```

Animation consumes the corresponding colormap or raw-VDF frame directories,
so enable
`plot_colormaps` with `animate_colormaps` and `plot_vdfs` with
`animate_vdfs` in the same run. Static plotting does not require ffmpeg;
MP4 animation does.

The optional `--x-selection-method` and `--o-selection-method` arguments
override the configured point-cell selection method. Supported values are
`manual`, `physical`, `consensus`, and `union`. Manual uses only the
configured box, physical uses the X Hessian box or O flux contour, consensus
uses their intersection, and union uses either set. Empty results are not
accepted. Any X and O records sharing a final selected VDF cell are both
omitted completely.

This display preference does not change union membership: all manual-only,
physical-only, and shared cells remain selected. Manual, physical, and
consensus area display is unchanged.

Turso CPU submission:

```bash
sbatch slurm/turso/cpu/extract_dataset.sbatch \
    --config configs/data/extraction.yaml \
    --start-timestep 3072 \
    --n-timesteps 1 \
    --dataset-kind train
```

HILE CPU submission:

```bash
sbatch --partition=hile slurm/hile/cpu/extract_dataset.sbatch \
    --config configs/data/extraction.yaml \
    --start-timestep 3072 \
    --n-timesteps 1 \
    --dataset-kind train
```

Extraction and Hermite projection can require substantially more memory than
a one-timestep example. Choose worker counts for the selected files,
velocity mesh, sample count, full-node CPU allocation, and enabled
postprocessing. More workers increase the number of concurrent raw VDFs and
Hermite calculations. Rotated 268 x 268 x 268 VDFs also require interpolation
arrays, so begin with two to four extraction workers and increase only after
measuring memory and VLSV I/O; unrotated extraction may support more. Do not
set `extraction_n_jobs` above the allocated physical CPU cores. Both site
wrappers launch one Slurm task, and the configured Joblib workers remain
local to that node. Throughput depends on VLSV filesystem I/O and the
per-sample Hermite calculation. The selected extraction job runs optional
Stage 6 and Stage 7 according to the YAML; no separate plotting or animation
job is needed.

## Dataset colormap

The standalone command uses the same renderer as extraction Stage 6: an
x-directed bulk-velocity background, black magnetic streamlines, yellow
VDF-carrying cells, blue configured or detected source coordinates, red
selected dataset cells, and exact current X/O geometry using the same union
visibility rule as Stage 6.
Each blue coordinate and its red selected VDF cells use the same small marker
shape, which carries class or point identity. Empty point selections and
complete X/O records ejected for sharing a selected VDF cell are absent. The
compact paired legend remains below the spatial axes; `current_layer` appears
only as a red selected-cell marker because it has no source coordinate.

Local command with an explicit source:

```bash
python -m scripts.data.plotting.plot_colormap \
    --config configs/data/extraction.yaml \
    --dataset-dir /path/to/dataset \
    --timestep 3072 \
    --source-vlsv /path/to/vlasiator/data/bulk.0003072.vlsv \
    --spatial-box-re -20 -10 -5 5 \
    --output /path/to/output/dataset-colormap.png
```

`--source-vlsv` takes precedence over paths in dataset metadata and the
extraction configuration. For a three-dimensional source,
`--spatial-slice-y-re` selects the y cut in Earth radii. Boolean options have
matching negative forms, including `--no-show-all-vdf-cells`,
`--no-show-labeled-cells`, `--no-show-class-legend`, and
`--no-show-magnetic-streamlines`. With no visibility overrides, the output
matches the Stage 6 visual conventions. The background is fixed to x velocity,
and one invocation writes the PNG supplied through `--output`.

Turso CPU submission:

```bash
sbatch slurm/turso/cpu/plot_colormap.sbatch \
    --config configs/data/extraction.yaml \
    --dataset-dir /path/to/dataset \
    --source-vlsv /path/to/vlasiator/data/bulk.0003072.vlsv \
    --output /path/to/output/dataset-colormap.png
```

HILE CPU submission:

```bash
sbatch --partition=hile slurm/hile/cpu/plot_colormap.sbatch \
    --config configs/data/extraction.yaml \
    --dataset-dir /path/to/dataset \
    --source-vlsv /path/to/vlasiator/data/bulk.0003072.vlsv \
    --output /path/to/output/dataset-colormap.png
```

## VDF plane plotting

The command reads full-resolution `X.npy` samples and uses the same
peak-plane extractor and renderer as Stage 6. Stage 6 always requests `xz`;
the standalone command accepts repeatable `--plane {xz,xy,yz}` options.
Omitting `--plane` draws all three in `xz`, `xy`, `yz` order. Every plane
crosses the sample's own three-dimensional VDF peak, and one invocation
writes one combined PNG. All velocity axes are km/s. Preparation reads the
resolved same-cell VLSV sparsity threshold, forms float32 `plane * dv` with
`dv = (vymax - vymin) / nvy`, zeros values strictly below
`threshold * dv`, and masks nonpositive values. The renderer uses unmodified
`nipy_spectral` with visible-support `LogNorm`, as Stage 6 does. Masked
regions retain the default transparency over white axes. Preparation affects
only the display copy, not `X.npy` or any machine-learning input.

Draw one `xz` plane matching the Stage 6 style:

```bash
python -m scripts.data.plotting.plot_vdf_slices \
    --dataset-dir /path/to/dataset \
    --sample-index 0 \
    --plane xz \
    --output /path/to/output/vdf-xz.png
```

Draw one `xy` plane:

```bash
python -m scripts.data.plotting.plot_vdf_slices \
    --dataset-dir /path/to/dataset \
    --sample-index 0 \
    --plane xy \
    --output /path/to/output/vdf-xy.png
```

Draw one `yz` plane:

```bash
python -m scripts.data.plotting.plot_vdf_slices \
    --dataset-dir /path/to/dataset \
    --sample-index 0 \
    --plane yz \
    --output /path/to/output/vdf-yz.png
```

Draw all three planes explicitly:

```bash
python -m scripts.data.plotting.plot_vdf_slices \
    --dataset-dir /path/to/dataset \
    --sample-index 0 \
    --plane xz \
    --plane xy \
    --plane yz \
    --output /path/to/output/vdf-all-planes.png
```

Select a deterministic number of point-class samples and two planes:

```bash
python -m scripts.data.plotting.plot_vdf_slices \
    --dataset-dir /path/to/dataset \
    --class-name x_point \
    --class-name o_point \
    --number-of-samples 4 \
    --plane xz \
    --plane yz \
    --velocity-limit-ms 2000000 \
    --output /path/to/output/vdf-point-classes.png
```

Sample-index, CID, and class-name selection are mutually exclusive. Repeat a
selector of one kind to request several values. Repeated `--plane` options
preserve their requested order. With several samples, the combined figure has
one row per sample and one column per requested plane.

Turso CPU submission:

```bash
sbatch slurm/turso/cpu/plot_vdf_slices.sbatch \
    --dataset-dir /path/to/dataset \
    --sample-index 0 \
    --plane xz \
    --plane yz \
    --output /path/to/output/vdf-slices.png
```

HILE CPU submission:

```bash
sbatch --partition=hile slurm/hile/cpu/plot_vdf_slices.sbatch \
    --dataset-dir /path/to/dataset \
    --sample-index 0 \
    --plane xz \
    --plane yz \
    --output /path/to/output/vdf-slices.png
```

The Slurm wrappers pass repeated plane arguments through unchanged.

## PCA analysis

PCA uses `configs/analysis/pca.yaml`. The source representation remains a
read-only memory map while three sample-batched passes calculate float64
Chan/Welford statistics, update the retained PyTorch PCA basis, and project
the final scores. The shipped `pca.batch_size` is 16; the examples below
use `--batch-size 8` as a lower-memory override. Raw PCA still transforms and
flattens every `(vx, vy, vz)` voxel. The batched design avoids the monolithic
allocation that requeres too much memory, but one batch, the
statistics, retained basis, and decomposition workspace must still fit.
Smaller batches lower peak memory and perform more incremental updates.

KMeans, nearest-neighbor search, and t-SNE use the leading 32 PCA score
columns by default and remain label blind. KMeans creates as many internal
diagnostic partitions as there are physical classes in the current dataset;
its labels are not saved or plotted. Physical labels are introduced only for
the two scatter plots, KMeans purity and concentration, local
nearest-neighbor purity, and sample-count reporting. Physical class is the
only color, marker, and legend category in both figures, and the default
neighbor count is 25. The default t-SNE uses perplexity 50, PCA
initialization, automatic learning rate, early exaggeration 16, and 2000
iterations. Both figures use marker size 3; PCA uses alpha 0.30 and t-SNE
uses alpha 0.28. These settings reduce overplotting without forcing physical
classes apart.

Full raw PCA on CPU:

```bash
python -m scripts.analysis.run_pca_analysis \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output \
    --representation raw \
    --pca-method full \
    --batch-size 8 \
    --device cpu
```

Low-rank raw PCA on CPU:

```bash
python -m scripts.analysis.run_pca_analysis \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output \
    --representation raw \
    --pca-method lowrank \
    --batch-size 8 \
    --device cpu
```

Low-rank Hermite PCA on CPU:

```bash
python -m scripts.analysis.run_pca_analysis \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output \
    --representation hermite \
    --pca-method lowrank \
    --batch-size 8 \
    --device cpu
```

Full raw PCA on CUDA:

```bash
python -m scripts.analysis.run_pca_analysis \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output \
    --representation raw \
    --pca-method full \
    --batch-size 8 \
    --device cuda
```

Low-rank raw PCA on CUDA:

```bash
python -m scripts.analysis.run_pca_analysis \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output \
    --representation raw \
    --pca-method lowrank \
    --batch-size 8 \
    --device cuda
```

Feature-sharded low-rank raw PCA on four GPUs in one node:

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    --module scripts.analysis.run_pca_analysis \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/pca \
    --representation raw \
    --pca-method lowrank \
    --device cuda \
    --multi-gpu \
    --batch-size 8
```

Each process owns a different contiguous feature interval and component
shard. NCCL combines small row-Gram matrices, broadcasts small Gram
eigenpairs, exchanges small orientation candidates, and reduces score
batches; only rank zero runs downstream analysis and writes the three output
files. This is one-node feature sharding, not sample data parallelism or MPI.

Local `--device` also accepts `auto` and indexed CUDA devices such as
`cuda:0`. `auto` uses CUDA when PyTorch reports it available and CPU
otherwise.

Turso CPU full-PCA submission:

```bash
sbatch slurm/turso/cpu/run_pca_analysis.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output/pca-turso-cpu-full \
    --representation raw \
    --pca-method full \
    --batch-size 8
```

Turso CPU low-rank submission:

```bash
sbatch slurm/turso/cpu/run_pca_analysis.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output/pca-turso-cpu-lowrank \
    --representation hermite \
    --pca-method lowrank \
    --batch-size 8
```

Turso GPU full-PCA submission:

```bash
sbatch slurm/turso/gpu/run_pca_analysis.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output/pca-turso-gpu-full \
    --representation raw \
    --pca-method full \
    --batch-size 8
```

Turso GPU low-rank submission:

```bash
sbatch slurm/turso/gpu/run_pca_analysis.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output/pca-turso-gpu-lowrank \
    --representation raw \
    --pca-method lowrank \
    --batch-size 8
```

HILE CPU full-PCA submission:

```bash
sbatch --partition=hile slurm/hile/cpu/run_pca_analysis.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output/pca-hile-cpu-full \
    --representation raw \
    --pca-method full \
    --batch-size 8
```

HILE CPU low-rank submission:

```bash
sbatch --partition=hile slurm/hile/cpu/run_pca_analysis.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output/pca-hile-cpu-lowrank \
    --representation hermite \
    --pca-method lowrank \
    --batch-size 8
```

HILE GPU full-PCA submission:

```bash
sbatch --partition=hile slurm/hile/gpu/run_pca_analysis.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output/pca-hile-gpu-full \
    --representation raw \
    --pca-method full \
    --batch-size 8
```

HILE GPU low-rank submission:

```bash
sbatch --partition=hile slurm/hile/gpu/run_pca_analysis.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output/pca-hile-gpu-lowrank \
    --representation raw \
    --pca-method lowrank \
    --batch-size 8
```

Turso feature-sharded multi-GPU PCA submission:

```bash
sbatch slurm/turso/gpu/run_pca_multi_gpu.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/pca \
    --representation raw \
    --pca-method lowrank \
    --batch-size 8
```

HILE feature-sharded multi-GPU PCA submission:

```bash
sbatch --partition=hile slurm/hile/gpu/run_pca_multi_gpu.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/pca \
    --representation raw \
    --pca-method lowrank \
    --batch-size 8
```

The CPU wrappers force `--device cpu`, and the GPU wrappers force
`--device cuda`. The two multi-GPU PCA wrappers additionally pass
`--multi-gpu` and launch one torchrun rank per allocated GPU. None hardcodes
a PCA method or batch size; `--pca-method` and `--batch-size` pass through
unchanged, while YAML defaults apply when the options are omitted.

The output directory receives:

```text
pca_physical_classes.png
tsne_physical_classes.png
metrics.txt
```

Both PCA methods write this identical three-file set and no feature,
standardized-feature, component, PCA-score, KMeans-label, or t-SNE-coordinate
cache. `metrics.txt` contains run and dataset settings, dynamic sample counts,
explained variance, internal KMeans purity and physical-class concentration,
the silhouette score, and nearest-neighbor physical-label purity. Full mode
uses exact reduced SVD inside each update; with several batches,
retained-rank truncation makes the result incremental rather than generally
identical to monolithic PCA. A single update is identical to the
corresponding monolithic reduced SVD. Low-rank mode seeds PyTorch once before
its single-device incremental randomized updates. Distributed low-rank mode
instead retains an oversampled feature-sharded basis through exact small
row-Gram eigensolves. Relative performance depends on sample count, feature
count, batch size, retained component count, dtype, transfer cost, and GPU
hardware; neither low-rank nor GPU execution is always faster.

The t-SNE figure shows the actual label-blind embedding. Physical classes
can overlap or follow continuous CID/timestep trajectories, and apparent
gaps are not quantitative distances. Perplexity values such as 30, 50, or
100 can be useful comparison experiments, but production creates one
configured embedding rather than selecting the most separated-looking plot.

## CNN training

CNN training uses `configs/models/cnn.yaml`. The dataset must contain enough
timesteps for the configured train, validation, gap, and test partitions.
The CNN derives classification width and output order from the physical
class IDs represented in `metadata.csv`; there is no class-count command
option. Raw training sends the complete `(vx, vy, vz)` VDF through
`Conv3d`. For a full-resolution run, first set `batch_size` and
`normalization_batch_size` to one in the CNN config and use a full-node GPU
profile when available. Actual memory use and runtime depend on the data
shape, architecture, and device.

Raw CNN on CPU:

```bash
python -m scripts.models.train_cnn \
    --config configs/models/cnn.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/cnn-raw \
    --representation raw \
    --device cpu
```

Hermite CNN on CUDA:

```bash
python -m scripts.models.train_cnn \
    --config configs/models/cnn.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/cnn-hermite \
    --representation hermite \
    --device cuda
```

Raw CNN with four consecutive model stages across visible GPUs:

```bash
python -m scripts.models.train_cnn \
    --config configs/models/cnn.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model \
    --representation raw \
    --device cuda \
    --model-parallel-gpus 4
```

This command runs one Python process. It distributes consecutive model
stages, optimizer state, and saved intermediate activations; it does not use
DDP and does not spatially shard an individual `Conv3d` operation.

The CLI also accepts `auto` and indexed devices such as `cuda:0`.

Turso CPU submission:

```bash
sbatch slurm/turso/cpu/train_cnn.sbatch \
    --config configs/models/cnn.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/cnn-raw \
    --representation raw
```

Turso GPU submission:

```bash
sbatch slurm/turso/gpu/train_cnn.sbatch \
    --config configs/models/cnn.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/cnn-raw \
    --representation raw
```

HILE CPU submission:

```bash
sbatch --partition=hile slurm/hile/cpu/train_cnn.sbatch \
    --config configs/models/cnn.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/cnn-raw \
    --representation raw
```

HILE GPU submission:

```bash
sbatch --partition=hile slurm/hile/gpu/train_cnn.sbatch \
    --config configs/models/cnn.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/cnn-raw \
    --representation raw
```

Turso model-parallel CNN submission:

```bash
sbatch slurm/turso/gpu/train_cnn_model_parallel.sbatch \
    --config configs/models/cnn.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model \
    --representation raw
```

HILE model-parallel CNN submission:

```bash
sbatch --partition=hile slurm/hile/gpu/train_cnn_model_parallel.sbatch \
    --config configs/models/cnn.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model \
    --representation raw
```

The output directory contains:

```text
model.pt
metrics.txt
```

## Autoencoder training

Autoencoder training uses `configs/models/autoencoder.yaml`. Raw input is the
complete `X.npy` volume in `(vx, vy, vz)` order; Hermite input is the complete
signed `X_hermite.npy` cube. Both use the same Conv3d model, complete-timestep
split policy, auxiliary six-target topology task, and combined validation
objective. `model.bottleneck_shape` sets the maximum retained spatial cells
along the three encoded axes; smaller values reduce dense bottleneck capacity
and memory. There is no raw slice, raw downsampling, or two-dimensional mode.

Raw autoencoder on CPU:

```bash
python -m scripts.models.train_autoencoder \
    --config configs/models/autoencoder.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/autoencoder-raw \
    --representation raw \
    --device cpu \
    --model-parallel-gpus 1
```

One-GPU raw autoencoder:

```bash
python -m scripts.models.train_autoencoder \
    --config configs/models/autoencoder.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/autoencoder-raw \
    --representation raw \
    --device cuda \
    --model-parallel-gpus 1
```

Local three-GPU Hermite autoencoder:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
python -m scripts.models.train_autoencoder \
    --config configs/models/autoencoder.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/autoencoder-hermite \
    --representation hermite \
    --device cuda \
    --model-parallel-gpus 3
```

The CLI also accepts `auto` and indexed devices such as `cuda:0`.
`--model-parallel-gpus` overrides the YAML value. It controls consecutive
model stages in one Python process; `data_loader.num_workers` controls only
sample loading and preprocessing.

Turso CPU submission:

```bash
sbatch slurm/turso/cpu/train_autoencoder.sbatch \
    --config configs/models/autoencoder.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/autoencoder-raw \
    --representation raw
```

Turso GPU submission:

```bash
sbatch slurm/turso/gpu/train_autoencoder.sbatch \
    --config configs/models/autoencoder.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/autoencoder-raw \
    --representation raw
```

HILE CPU submission:

```bash
sbatch --partition=hile slurm/hile/cpu/train_autoencoder.sbatch \
    --config configs/models/autoencoder.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/autoencoder-raw \
    --representation raw
```

Turso one-process model-parallel submission:

```bash
sbatch slurm/turso/gpu/train_autoencoder_model_parallel.sbatch \
    --config configs/models/autoencoder.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/autoencoder-raw \
    --representation raw
```

HILE one-process model-parallel submission:

```bash
sbatch slurm/hile/gpu/train_autoencoder_model_parallel.sbatch \
    --config configs/models/autoencoder.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/autoencoder-hermite \
    --representation hermite
```

The output directory contains:

```text
autoencoder.pt
metrics.txt
training_history.csv
reconstruction_examples.png
```

`metrics.txt` describes the complete input shape, Conv3d model, configured and
effective bottleneck shapes, latent size, requested and effective stage
placement, topology scaling and order, and reconstruction, topology, and total
losses by split. It retains reconstruction MSE by current physical class and
reports each topology target's valid count, MAE, and RMSE in Earth radii. Class
labels are joined only after reconstruction for reporting; they are not model
inputs, loss targets, or checkpoint-selection data. Topology metadata is a
masked auxiliary target and is never a model input.
The consolidated PNG uses the restored best model and shows deterministic
original, reconstructed, and absolute-error examples for the active raw or
Hermite representation. Raw originals and reconstructions reuse the
extraction Stage 6 physical VDF preparation and drawing functions with
physical km/s axes, transparent threshold masks over white axes,
`nipy_spectral`, and one shared `LogNorm`. The displayed x-z plane crosses the
original complete volume's three-dimensional peak and is visualization only.
Hermite examples retain their signed coefficient scale over white axes.

Full `268^3` raw training has high memory cost. Begin with batch size one;
model parallelism does not split one Conv3d, so every stage and its backward
activation must fit on its owning GPU. For small models, activation transfers
may make several GPUs slower than one.

## Coordinate prediction

Coordinate prediction selects the source VDF cell nearest a requested
`[x, y, z]` coordinate in Earth radii.

Local command:

```bash
python -m scripts.prediction.predict_coordinate \
    --config configs/prediction/coordinate.yaml \
    --checkpoint /path/to/model/model.pt \
    --output-dir /path/to/output/coordinate-prediction \
    --timestep 3072 \
    --coord-re -11.74 0.01 7.08 \
    --device cpu \
    --plot
```

The checkpoint determines the numerical representation used for inference.
For a raw checkpoint, prediction resamples and transforms the complete
three-dimensional source VDF before `Conv3d` inference; any VDF planes in
the optional combined PNG are visualization only. Those raw physical panels
reuse the shared historical preparation and drawing where applicable, with
km/s axes and transparent masked regions over white axes; they do not alter
the saved VDF or model input.
`--file-source` selects a key from the configured source templates. Use
`--no-plot` to disable a figure enabled in the YAML.

Turso CPU submission:

```bash
sbatch slurm/turso/cpu/predict_coordinate.sbatch \
    --config configs/prediction/coordinate.yaml \
    --checkpoint /path/to/model/model.pt \
    --output-dir /path/to/output/coordinate-turso-cpu \
    --timestep 3072 \
    --coord-re -11.74 0.01 7.08 \
    --plot
```

Turso GPU submission:

```bash
sbatch slurm/turso/gpu/predict_coordinate.sbatch \
    --config configs/prediction/coordinate.yaml \
    --checkpoint /path/to/model/model.pt \
    --output-dir /path/to/output/coordinate-turso-gpu \
    --timestep 3072 \
    --coord-re -11.74 0.01 7.08 \
    --plot
```

HILE CPU submission:

```bash
sbatch --partition=hile slurm/hile/cpu/predict_coordinate.sbatch \
    --config configs/prediction/coordinate.yaml \
    --checkpoint /path/to/model/model.pt \
    --output-dir /path/to/output/coordinate-hile-cpu \
    --timestep 3072 \
    --coord-re -11.74 0.01 7.08 \
    --plot
```

HILE GPU submission:

```bash
sbatch --partition=hile slurm/hile/gpu/predict_coordinate.sbatch \
    --config configs/prediction/coordinate.yaml \
    --checkpoint /path/to/model/model.pt \
    --output-dir /path/to/output/coordinate-hile-gpu \
    --timestep 3072 \
    --coord-re -11.74 0.01 7.08 \
    --plot
```

The output directory contains:

```text
predictions_<timestep>.csv
prediction_combined.png       when plotting is enabled
```

The one-row CSV uses the same schema and writer as region prediction. It is
written before optional figure rendering and contains requested and selected
coordinates, `predicted_probability`, dynamic class probabilities, and all
six topology values. The CSV is the sole serialized coordinate prediction.

The combined map prefers direct x velocity from `<population>/vg_v`,
`<population>/V`, `vg_v`, or `V`, then uses legacy `rho_v / rho`. It keeps
black magnetic streamlines over the default x-z box
`[-30, 30, -15, 15]` R_E. The requested coordinate is a compact blue star,
the selected VDF uses a compact predicted-class marker, scalar X/O distances
remain circles around that VDF cell, and the legend occupies a dedicated row
below the colormap. The title continues to report CID, class, and probability.
Vector arrows, inferred endpoints, and connectors are not rendered.

## Region prediction

Region prediction selects cells with the `region_re` bounds in
`configs/prediction/region.yaml`. Spatial bounds are configured in YAML;
there is no separate region-bounds CLI option.

Local command:

```bash
python -m scripts.prediction.predict_region \
    --config configs/prediction/region.yaml \
    --checkpoint /path/to/model/model.pt \
    --output-dir /path/to/output/region-prediction \
    --start-timestep 3072 \
    --n-timesteps 3 \
    --device cpu \
    --plot
```

As with coordinate prediction, `--file-source`, `--plot`, and `--no-plot`
provide direct workflow overrides. The checkpoint determines the
representation. A raw checkpoint uses complete three-dimensional VDFs;
two-dimensional cuts in the combined figure are visualization only. Those
raw physical panels reuse the shared historical preparation and drawing
where applicable, with km/s axes and transparent masked regions over white
axes; they do not alter saved VDFs or model inputs.

Turso CPU submission:

```bash
sbatch slurm/turso/cpu/predict_region.sbatch \
    --config configs/prediction/region.yaml \
    --checkpoint /path/to/model/model.pt \
    --output-dir /path/to/output/region-turso-cpu \
    --start-timestep 3072 \
    --n-timesteps 3 \
    --plot
```

Turso GPU submission:

```bash
sbatch slurm/turso/gpu/predict_region.sbatch \
    --config configs/prediction/region.yaml \
    --checkpoint /path/to/model/model.pt \
    --output-dir /path/to/output/region-turso-gpu \
    --start-timestep 3072 \
    --n-timesteps 3 \
    --plot
```

HILE CPU submission:

```bash
sbatch --partition=hile slurm/hile/cpu/predict_region.sbatch \
    --config configs/prediction/region.yaml \
    --checkpoint /path/to/model/model.pt \
    --output-dir /path/to/output/region-hile-cpu \
    --start-timestep 3072 \
    --n-timesteps 3 \
    --plot
```

HILE GPU submission:

```bash
sbatch --partition=hile slurm/hile/gpu/predict_region.sbatch \
    --config configs/prediction/region.yaml \
    --checkpoint /path/to/model/model.pt \
    --output-dir /path/to/output/region-hile-gpu \
    --start-timestep 3072 \
    --n-timesteps 3 \
    --plot
```

For each timestep, the output directory contains:

```text
predictions_<timestep>.csv
predictions_<timestep>_combined.png       when plotting is enabled
```

Region CSV rows retain predicted class probabilities and all six topology
values. The optional map uses the same fixed x-velocity background and view
as coordinate prediction but shows only compact predicted-class markers for
the representatives whose VDF panels are displayed. It has no
requested-coordinate star, distance circles, adjacent class-name text,
representative CID/class/probability title lines, vector arrows, inferred
endpoints, or connectors. A concise region title remains, and the dedicated
legend row below the map identifies the marker classes.

Prediction VDF panels automatically resolve the first source threshold
producer available in this order: `MinValue`,
`<population>/MinValue`,
`<population>/EffectiveSparsityThreshold`, and
`<population>/vg_effectivesparsitythreshold`. The name is selected once per
open source and reused for every displayed CID; no threshold-name command
option is required.

## Common Slurm operation

The site and CPU/GPU directory selects the checked-in resource profile. The
Python argument surface is identical across profiles, and every argument
after the job path reaches the module unchanged. CPU and GPU variants inject
`--device cpu` and `--device cuda`, respectively, for the five
device-selectable workflows. Slurm submissions therefore omit `--device`;
do not override a GPU profile with `--device cpu` merely to obtain its
allocation, and do not request CUDA from a CPU profile.

Select the repository and virtual environment without editing a job file:

```bash
VDF_ML_REPO_ROOT=/path/to/repository \
VDF_ML_VENV=/path/to/venv \
sbatch slurm/turso/cpu/run_pca_analysis.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/output/pca \
    --representation raw \
    --batch-size 8
```

The wrappers do not set `TMPDIR` or create and remove temporary directories.
They use `VDF_ML_REPO_ROOT` when supplied, otherwise Slurm's
`SLURM_SUBMIT_DIR`, and use their nested source location only for local
execution outside Slurm. They optionally activate `VDF_ML_VENV` and preserve
the submitted environment. Submit from the repository root so
`SLURM_SUBMIT_DIR` and relative CLI paths resolve to the project.

Test CPU command construction locally without invoking `srun`:

```bash
VDF_ML_DRY_RUN=1 \
bash slurm/turso/cpu/train_cnn.sbatch \
    --config configs/models/cnn.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/cnn \
    --representation raw
```

Test the corresponding GPU command:

```bash
VDF_ML_DRY_RUN=1 \
bash slurm/turso/gpu/train_cnn.sbatch \
    --config configs/models/cnn.yaml \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/model/cnn \
    --representation raw
```

Dry-run mode prints shell-quoted Python arguments and exits without calling
`srun` or contacting a scheduler. It does not verify partition availability,
site policy, VLSV access, CUDA, or scientific inputs.

For a multi-GPU wrapper dry run outside Slurm, set the local process/device
count explicitly:

```bash
VDF_ML_DRY_RUN=1 \
VDF_ML_GPU_COUNT=4 \
bash slurm/turso/gpu/run_pca_multi_gpu.sbatch \
    --config configs/analysis/pca.yaml \
    --dataset-dir "/path/with spaces/dataset" \
    --output-dir /path/to/pca \
    --representation raw \
    --pca-method lowrank \
    --batch-size 8
```

`VDF_ML_GPU_COUNT` controls only local command construction or the number of
processes/stages launched inside an existing allocation. It does not change
an `sbatch` GPU request; use an `sbatch --gres=gpu:N` override when the
scheduler allocation itself must differ from the checked-in HILE profile.

Submitted standard output and error use these checked-in patterns:

```text
slurm/outputs/slurm-%x-%j.out
slurm/outputs/slurm-%x-%j.err
```

Slurm replaces `%x` with the job name and `%j` with the job ID. The tracked
`slurm/outputs/` directory exists before Slurm opens these files. Common
operational commands are:

```bash
squeue --me
scontrol show jobid -dd <jobid>
seff <jobid>
scancel <jobid>
```

Use `seff` after the job completes. Smaller exploratory jobs can override
resource directives on the `sbatch` command line or use an interactive
allocation instead of the checked-in maximum-resource profiles.

The files request one node and one task and launch one `srun` process.
Increasing a Python worker count does not turn a run into MPI or a Slurm
multi-task job. Because `--exclusive` and `--mem=0` reserve the full node,
short plotting jobs may queue for substantially more resources than they
consume; use local execution when it is appropriate and source data is
accessible.

## Output summary

| Functionality | Python command | Main outputs |
|---|---|---|
| Dataset extraction | `scripts.data.extract_dataset` | core dataset files; optional per-timestep colormaps, per-cell-ID VDF frames, and animations |
| Dataset colormap | `scripts.data.plotting.plot_colormap` | one requested PNG |
| VDF slices | `scripts.data.plotting.plot_vdf_slices` | one requested combined PNG |
| PCA | `scripts.analysis.run_pca_analysis` | two physical-class PNGs and `metrics.txt` |
| CNN training | `scripts.models.train_cnn` | `model.pt`, `metrics.txt` |
| Autoencoder training | `scripts.models.train_autoencoder` | `autoencoder.pt`, `metrics.txt`, `training_history.csv`, `reconstruction_examples.png` |
| Coordinate prediction | `scripts.prediction.predict_coordinate` | one-row `predictions_<timestep>.csv`, optional `prediction_combined.png` |
| Region prediction | `scripts.prediction.predict_region` | per-timestep prediction CSV and optional combined PNG |
