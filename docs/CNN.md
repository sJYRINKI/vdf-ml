# CNN

## Purpose

The CNN predicts a dataset physical class and six auxiliary topology values
from either the raw or Hermite representation. Its classification width is
derived from the stable class IDs present in training metadata rather than a
hard-coded class count.

The model receives exactly one tensor:

```text
representation tensor
    -> class logits
    -> topology predictions
    -> learned embedding
```

Labels, topology targets, masks, coordinates, magnetic field, fluid
velocity, VDF sparsity thresholds, moments, and requested prediction
coordinates are never
model inputs.

## Input representations

### Raw

Raw input uses the complete `raw` VDF:

```text
X.npy [vx, vy, vz]
    -> positive floor applied to every velocity-space cell
    -> log10
    -> one-channel [1, vx, vy, vz] tensor
```

The saved velocity-axis order is preserved. No axis is sliced, projected,
averaged, cropped, or downsampled before the model.

### Hermite

Hermite input is the complete saved physical-VDF coefficient volume:

```text
X_hermite.npy
    -> derive actual (order, order, order) shape
    -> preserve [n_x, n_y, n_z] when unrotated or
       [n_parallel, n_perp1, n_perp2] when rotated
    -> one-channel 3-D tensor
```

It is not flattened, sliced, transposed, logged, thresholded, or normalized
per sample. The dimensional coefficients were projected directly from the
physical VDF; the default extraction order is 22, but model construction
uses the saved shape rather than a global order constant.

Training opens the selected saved representation as a read-only memory map.
Each dataset item reads and prepares only its requested row, avoiding
materialization of the complete processed tensor.

## Model

The raw encoder repeats:

```text
Conv3d(kernel=3, padding=1) -> ReLU -> AvgPool3d
```

The unchanged Hermite encoder uses corresponding `Conv3d` operations on
the complete coefficient cube. Pooling is omitted when it would make an
axis smaller than the configured adaptive-pooling target. Three-dimensional
adaptive average pooling gives a fixed encoder output without assuming a
hardcoded input size.

Both encoders feed a shared hidden layer. Separate linear heads produce:

- one class logit per mapped dataset class in explicit checkpoint order; and
- six scaled topology values.

The returned embedding is diagnostic. It is not used to filter samples.

### Layer-parallel CUDA placement

`model_parallel_gpus: 1` preserves the ordinary CPU or single-CUDA-device
model. A larger value uses one Python process to place consecutive existing
encoder blocks and the shared output stage on consecutive visible CUDA
devices. The input-normalization buffers and first encoder stage use the
first device. Each activation moves only when the next consecutive stage is
on another device, and the shared embedding, class head, and topology head
remain on the final occupied device. PyTorch autograd propagates gradients
through these transfers.

This layout distributes layer parameters, AdamW optimizer state, and saved
intermediate activations across GPU memories. It is model parallelism, not
DDP, `DataParallel`, FSDP, or batch splitting, and the complete model is not
replicated per GPU. It also does not spatially shard an individual `Conv3d`
or one convolution activation. Consequently, each individual stage and its
activation must still fit on the device that owns it, and extra transfers
can make model-parallel execution slower than one GPU.

## Class mapping

Project class IDs are mapped explicitly to contiguous model-output indices:

| Project ID | Class |
|---:|---|
| 0 | `lobe` |
| 1 | `current_layer` |
| 2 | `o_point` |
| 3 | `x_point` |
| 4 | `solar_wind` |
| 5 | `magnetosheath` |
| 6 | `inner_magnetosphere` |

The training loader reads classes represented by `metadata.csv`, orders
their project IDs numerically, and constructs a contiguous output mapping.
Thus a dataset containing the complete configured taxonomy has seven
classification logits, while the implementation remains driven by dataset
metadata. The checkpoint stores project IDs, names, and output order. Code
must not infer order alphabetically.

## Input normalization

Mean and population standard deviation are accumulated in float64 for every
raw voxel or Hermite coefficient using training indices only:

```text
normalized = (input - input_mean) / input_scale
```

Near-zero scales become one. The model stores mean and scale as buffers and
applies normalization exactly once. Validation and test samples do not
influence the fit. These values are independent from PCA's label-blind
PyTorch feature standardization.

## Topology targets

Topology schema has this exact order:

1. `distance_to_x_point_re`
2. `distance_to_o_point_re`
3. `vdf_to_x_point_dx_re`
4. `vdf_to_x_point_dz_re`
5. `vdf_to_o_point_dx_re`
6. `vdf_to_o_point_dz_re`

The target scaler fits each column using valid training values only. It
uses the topology mask so missing values contribute nothing to scaling or
loss.

The fixed target order remains in `src.data.metadata_columns`. Generic target
extraction, finite masks, training-only scaling, inverse scaling, and globally
masked Smooth L1 live in `src.learning.topology_supervision` and are shared
unchanged with the autoencoder; CNN architecture and numerical behavior
remain owned here.

Evaluation and prediction inverse-transform outputs to Earth-radii physical
units.

## Loss

The training objective is:

```text
cross_entropy(class_logits, class_target)
    + topology_loss_weight
      * masked_smooth_l1(topology_prediction, topology_target)
```

Smooth L1 is summed over valid scalar targets and divided by the number of
valid values. A batch with no valid topology values receives a
graph-connected zero topology term.

There are no class weights, focal terms, balanced samplers, replacement
sampling, over/undersampling, or class-aware filters.

## Timestep-aware split

Samples are partitioned by complete, chronologically ordered timesteps:

```text
train -> excluded gap -> validation -> excluded gap -> test
```

The configured split produces:

- nonempty train, validation, and test partitions;
- no sample or timestep overlap;
- no gap timestep in a partition;
- stable row order within each partition.

Input normalization and topology scaling use training indices only. Test
results do not influence checkpoint selection.

## Ordered modules

The training package reads in execution order:

```text
step_00_train_cnn.py
step_01_load_cnn_data.py
step_02_split_cnn_timesteps.py
step_03_scale_cnn_inputs.py
step_04_build_cnn.py
step_05_calculate_cnn_loss.py
step_06_optimize_cnn.py
step_07_evaluate_cnn.py
step_08_save_cnn.py
```

`load_cnn_checkpoint.py` is reusable by evaluation and prediction rather
than a numbered training stage.

## Optimization and model selection

Training uses:

- an ordinary shuffled training DataLoader;
- stable validation and test loaders;
- AdamW;
- optional configured gradient clipping;
- validation after every epoch;
- validation macro F1 as the primary selection metric;
- validation total loss as a deterministic tie-breaker;
- patience-based early stopping; and
- either one CPU/CUDA device or consecutive CUDA model stages controlled by
  one Python process, without DDP or `DataParallel`.

Final train, validation, and test metrics come from the reloaded best
checkpoint.

## Evaluation

Evaluation calculates:

- accuracy;
- macro and weighted F1;
- dynamically named, fixed-order classification reports;
- fixed-order confusion matrices; and
- per-target valid count, MAE, and RMSE in `R_E`.

Classes absent from a partition remain in the fixed output order with zero
support. The consolidated `metrics.txt` records run and configuration
information, dataset identity, chronological split ranges, class order,
training duration, selected epoch, best validation macro F1, final training
loss, final partition losses and scores, classification reports, confusion
matrices, and aggregate and per-target topology errors. Class names always
come from the current run's explicit class mapping; the report does not
embed a historical example taxonomy.

The error summaries use these established definitions:

```text
error = 1 - accuracy
generalization gap = test error - train error
bias proxy = train error
variance proxy = max(0, generalization gap)
```

`Final training loss` is the optimizer loss from the last completed epoch
before the validation-selected state is restored. Final train, validation,
and test losses are separate evaluation-mode measurements of the restored
model.

## Checkpoint

`model.pt` is the CNN checkpoint and stores:

- representation and representation name;
- complete input shape;
- complete raw training velocity-grid shape and extent or Hermite actual
  volume shape, order, and optional rotation setting;
- input mean, scale, and epsilon;
- class IDs, names, and output mapping;
- topology order and scaler;
- model architecture;
- topology loss weight;
- random seed.

The direct representation name, preprocessing values, architecture, and
exact topology target order define the checkpoint.

The loader accesses these fields directly, reconstructs the model and
scalers, loads the state dictionary, and selects evaluation mode.
Checkpoints are expected to come from the current workflow and use `raw` or
`hermite`. Current loading does not adapt earlier sliced raw checkpoints.
It also does not adapt Hermite checkpoints trained on the retired compact-log
or fixed-order convention; regenerate the Hermite dataset and retrain the
model.
Runtime CUDA device IDs and the requested model-parallel GPU count are not
part of the model architecture or state-dictionary names. A checkpoint
trained with several stage devices can therefore be reconstructed on one
GPU, another supported multi-GPU layout, or CPU inference.

## Configuration

`configs/models/cnn.yaml` owns the default seed, device,
`model_parallel_gpus`, model, topology loss, timestep split, loader, AdamW,
and training settings. The command selects the dataset, output directory,
and raw or Hermite representation.

The important sections are:

| Section | Purpose |
|---|---|
| `raw` | positive `log_eps` used as the complete-volume lower bound |
| `model` | raw/Hermite Conv3d channels, three-dimensional adaptive pool shapes, shared hidden width, dropout |
| `topology` | fixed Smooth L1 loss, loss weight, scaler epsilon |
| `split` | train and validation fractions plus gap timesteps |
| `data_loader` | batch sizes, workers, normalization batch size, pinning |
| `optimizer` | AdamW rate, decay, betas, epsilon, optional clipping |
| `training` | maximum epochs, patience, minimum improvement |
The configuration is expected to contain values accepted by the successful
PyTorch and scikit-learn path and to describe a usable whole-timestep
split.

The configuration contains no class weights, samplers, balancing,
over/undersampling, or feature filtering. Devices are `cpu`, `cuda`,
`cuda:N`, and `auto`. The supplied compact raw channels are `[4, 8, 16]`,
and the raw adaptive-pooling shape is `[2, 2, 2]`. Complete raw volumes
require substantial activation memory in both the CNN and the fully
three-dimensional autoencoder. The shared checked-in batch values remain
eight to preserve Hermite defaults; set them to one for an initial
full-resolution raw run and profile memory before increasing batch sizes,
channels, or worker counts.

Comments immediately above each YAML key are the quickest reference for
accepted choices, training effects, and memory-sensitive settings.

## Command

See [Example commands](EXAMPLE_COMMANDS.md#cnn-training) for raw, Hermite,
CPU, CUDA, and Slurm variants.

Select `raw` for the complete physical VDF or `hermite` for the complete
coefficient cube. Both branches use `Conv3d`; they do not share weights.

Training uses only whole-timestep partitions and never falls back to
sample-level random partitioning.

`--model-parallel-gpus N` overrides the runtime stage count for training.
Coordinate and region prediction expose the same override and load the
checkpoint onto that runtime layout without changing prediction inputs or
outputs. Model-parallel training and prediction use one Python process; they
are not launched with `torchrun`.

## Artifacts

A successful run writes:

```text
model.pt
metrics.txt
```

Files are saved directly. `metrics.txt` is the single human-readable run and
evaluation report. CNN training does not write JSON, CSV, prediction, or
separate confusion-matrix artifacts, and it does not persist the complete
per-epoch history.
