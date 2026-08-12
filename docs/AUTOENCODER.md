# Autoencoder

## Purpose

The autoencoder learns one compact combined latent representation from either
complete three-dimensional VDF representation and aligned plasma
context. It reconstructs the normalized VDF input and uses six masked topology
targets to encourage that representation to retain physical X/O geometry.

It is:

- deterministic and non-variational;
- one Conv3d encoder/decoder model for both representations;
- trained on complete volumes without raw slicing or downsampling;
- topology-aware without using topology as an input;
- conditioned on aligned 16-value plasma context;
- optionally layer-model-parallel in one Python process.

It is not adversarial, a physical-class predictor, or currently a generative
model. It does not use DDP, `DataParallel`, FSDP, model replication, or
sample-level training pools.

## Inputs

Every raw or Hermite sample is paired with one float32
`plasma_context.npy` row of shape `(16,)`. Its exact SI order is
`magnetic_field_x_t`, `magnetic_field_y_t`, `magnetic_field_z_t`,
`electric_field_x_vm`, `electric_field_y_vm`, `electric_field_z_vm`,
`bulk_velocity_x_ms`, `bulk_velocity_y_ms`, `bulk_velocity_z_ms`,
`number_density_m3`, `pressure_xx_pa`, `pressure_yy_pa`, `pressure_zz_pa`,
`pressure_xy_pa`, `pressure_xz_pa`, and `pressure_yz_pa`. B uses tesla, E
uses volts per metre, V uses metres per second, density uses particles per
cubic metre, and pressure uses pascals. The components preserve direction and
magnitude, so the model does not receive redundant vector magnitudes.
Topology remains supervised metadata rather than context.

### Raw

Raw reconstruction uses every saved VDF voxel:

```text
X.npy sample [vx, vy, vz]
    -> configured positive floor
    -> log10
    -> training-derived per-voxel centring and scaling
    -> model tensor [1, vx, vy, vz]
```

The saved array has shape `(n_samples, vx, vy, vz)`, and a model batch has
shape `(batch_size, 1, vx, vy, vz)`. Axis order remains `[vx, vy, vz]`.
Training never selects a fixed velocity-space plane; it does not make a
projection, concatenate planes, crop the grid, or downsample the VDF. The
source `X.npy` values remain unchanged and no transformed raw cache is saved.

### Hermite

Hermite reconstruction uses every signed saved coefficient:

```text
X_hermite.npy sample [n1, n2, n3]
    -> training-derived per-coefficient centring and scaling
    -> model tensor [1, n1, n2, n3]
```

The model derives `(n1, n2, n3)` directly from `X_hermite.npy.shape[1:]`.
Unrotated axes are `[n_x, n_y, n_z]`; optionally rotated axes are
`[n_parallel, n_perp1, n_perp2]`. The default extraction order is 22, but the
autoencoder does not hardcode an order. It does not select a coefficient
plane, truncate or downsample the cube, take absolute values, remove negative
coefficients, or apply the raw-VDF logarithm.

## On-demand memory-mapped loading and normalization

Each DataLoader process opens the representation and plasma-context files as
read-only NumPy memory maps on its first sample request and reuses those
mappings. A partition dataset retains the paths, selected sample indices,
small topology target
and mask arrays, normalization values, and reporting identity. Each
`__getitem__` call reads, copies, and preprocesses only the requested saved
row. This is on-demand memory-mapped sample loading: the workflow never
materializes a complete `(n_samples, 1, vx, vy, vz)` raw tensor in RAM.

The same training indices fit the context mean and population standard
deviation in float64. Validation, test, plotting, and checkpoint reload apply
those training values to aligned context rows without changing the physical
saved array.

Input normalization uses the established bounded float64 accumulation:

- only training indices fit the per-voxel or per-coefficient mean and
  population standard deviation;
- near-constant scales become one;
- validation and test use the training values; and
- the float32 mean and scale are model buffers and checkpoint data.

Raw normalization follows the positive-floor and `log10` transform before
training-derived feature scaling. Hermite normalization acts directly on the
complete signed coefficient cube. Neither representation is normalized per
sample.

`loader.num_workers` controls only PyTorch worker processes that load and
preprocess samples. It does not place or replicate model stages, and extraction
worker settings do not control autoencoder training.

## Auxiliary topology targets

The autoencoder predicts exactly these metadata values, in this order:

1. `distance_to_x_point_re`
2. `distance_to_o_point_re`
3. `vdf_to_x_point_dx_re`
4. `vdf_to_x_point_dz_re`
5. `vdf_to_o_point_dx_re`
6. `vdf_to_o_point_dz_re`

All six values are in Earth radii. Optional y-displacement metadata is not a
target. Each sample has a target vector and Boolean mask with shape `(6,)`.
The mask is true independently for each finite value, so a sample with only
some available quantities still contributes those quantities. Missing values
do not remove a sample from reconstruction training, affect timestep splits,
or change reconstruction weighting.

Topology scaling is independent from representation normalization. Each
column's mean and population standard deviation use finite training entries
only. Validation and test targets use those training-derived values, and
evaluation applies the inverse transform before reporting errors in Earth
radii.

Topology is an auxiliary target because its role is to encourage the latent
space to retain physical X/O geometry. It is never concatenated to raw voxels,
Hermite coefficients, encoder activations, the latent vector, or decoder
input. Topology predictions are not fed back into reconstruction.

## Architecture

`VdfAutoencoder` is the sole active model class. Both representations use the
same dimensional convention and direct three-dimensional equivalents of the
established architecture:

```text
complete one-channel volume
    -> Conv3d / ReLU / conditional AvgPool3d encoder blocks
    -> flattened spatially reduced bottleneck
    -> Linear VDF latent projection
16-value context -> Linear / ReLU context embedding
    -> concatenate on bottleneck device -> combined_latent
       |-> Linear / ReLU / Linear topology head -> 6 predictions
       `-> Linear decoder projection
           -> trilinear interpolation / Conv3d decoder blocks
           -> one-channel complete-volume reconstruction
```

Pooling reduces spatial dimensions before the latent linear layer. The model
records encoder geometry and uses temporary internal interpolation so the
decoder restores the exact input shape without cropping or permanently
padding saved data. The final reconstruction projection remains linear in
normalized representation space.

`model.bottleneck_shape` supplies a maximum for the three spatial dimensions
retained by `AdaptiveAvgPool3d`. Raw values follow `(vx, vy, vz)`; Hermite
values follow the saved unrotated `(n_x, n_y, n_z)` or rotated
`(n_parallel, n_perp1, n_perp2)` coefficient order. The effective bottleneck
shape is the per-axis minimum of the configured maximum and encoded volume,
so pooling never expands a smaller encoded axis. The direct constructor
default `[4, 4, 4]`
therefore preserves the existing behavior for encoded volumes smaller than
four cells along any axis.

The two latent projection parameter counts scale with the product of the
effective bottleneck dimensions. Smaller settings reduce dense parameters,
checkpoint size, bottleneck-stage memory, and compute at the cost of stronger
spatial compression. Larger settings retain more encoded spatial information;
the complete bottleneck stage must still fit on one owning device.

The forward method requires `vdf_input` and aligned `plasma_context` and returns
this mapping:

```text
reconstruction
topology_predictions
combined_latent
```

`reconstruction` has the same shape as `vdf_input`, topology predictions have
shape `(batch_size, 6)`, and the non-detached combined latent has width
`model.latent_size + plasma_context.hidden_size`. Both losses reach the
context branch and shared VDF encoder through the combined latent. The
decoder reconstructs only the normalized VDF and has no context output or
context reconstruction loss.

## One-process model parallelism

`model_parallel_gpus` selects ordered layer model parallelism inside one
Python process. Meaningful consecutive stages comprise input normalization
and encoder blocks, the bottleneck, decoder blocks, final reconstruction
projection, and topology output. Adjacent stages remain adjacent after
placement. With `S` meaningful stages and `D` effective devices, stage `i`
uses device index:

```text
i * D // S
```

The effective count is capped by the requested count, visible CUDA devices,
and meaningful stage count. CPU and one-GPU runs use the same placement path
with every stage on one device. Runtime attributes describe the input device,
output device, and stage devices.

The VDF input batch moves to the input device, normalization runs there, and
an activation moves only when the next stage has another owner. The small
scaled context batch and its dense branch move to the bottleneck device,
where fusion occurs. Reconstruction,
topology predictions, and the exposed latent vector finish on the output
device. PyTorch autograd follows those transfers during one backward pass.
One AdamW optimizer owns every encoder, VDF latent, context branch, decoder,
reconstruction, and topology parameter exactly once, including when parameter
groups reside on several devices.

This design does not schedule microbatches, replicate the model, all-reduce
gradients, or spatially shard one convolution. Each individual Conv3d stage
and its local activation must fit on its owning device. Activation transfers
can make several GPUs slower than one GPU for small volumes or compact models.

## Training and selection

The autoencoder preserves chronological complete-timestep train, validation,
and test partitions with configured gap timesteps. Topology availability does
not influence those partitions or row order.

The combined loss is:

```text
reconstruction_loss = mse(
    complete_normalized_reconstruction,
    complete_normalized_input,
)

topology_loss = smooth_l1(
    topology_predictions[topology_mask],
    scaled_topology_targets[topology_mask],
)

total_loss = reconstruction_loss
           + topology.loss_weight * topology_loss
```

The Smooth L1 reduction is global over every valid scalar in the batch. A
batch without a valid topology entry receives the established graph-connected
zero topology term. Its samples still contribute full-volume reconstruction
MSE.

AdamW updates the one model with one `total_loss.backward()` call per batch.
Validation total loss, not reconstruction loss alone, controls the best epoch,
early stopping, and the state restored for final evaluation. Test data is
evaluation-only.

## Evaluation

The restored validation-selected state is evaluated on train, validation, and
test. Each split reports:

- complete-volume normalized reconstruction MSE;
- globally masked scaled topology Smooth L1;
- the combined total objective; and
- reconstruction MSE grouped by physical class after inference.

Topology predictions and targets are inverse-scaled into Earth radii. Each
split and target reports valid count, MAE, and RMSE using valid entries only.
Physical class names remain reporting metadata; they are not inputs, targets,
sampler weights, or model-selection data. Per-sample topology predictions are
not saved.

## Ordered modules

The training package reads in execution order:

```text
step_00_train_autoencoder.py
step_01_load_autoencoder_data.py
step_02_split_autoencoder_timesteps.py
step_03_scale_autoencoder_inputs.py
step_04_build_autoencoder.py
step_05_optimize_autoencoder.py
step_06_evaluate_autoencoder.py
step_07_save_autoencoder.py
```

`autoencoder_loss.py` owns the combined objective without becoming an
artificial workflow stage. `load_autoencoder_checkpoint.py` is the reusable
checkpoint loader. Generic six-target extraction, scaling, and masked loss
mathematics live in `src/learning/topology_supervision.py` and are shared
with the CNN.

## Checkpoint

`autoencoder.pt` stores:

- representation and complete three-dimensional input shape;
- encoder and decoder channels, configured bottleneck maximum, effective
  bottleneck shape, latent size, and pooling architecture;
- training-derived representation mean, scale, and epsilon;
- exact 16 plasma-context names, training mean and scale, dense-branch
  hidden width, VDF latent width, and combined latent width;
- exact topology target order, scaler mean and scale, hidden size, and loss
  weight;
- random seed;
- best epoch and best validation total loss; and
- device-independent CPU model parameters under the direct `state_dict`
  field.

Runtime CUDA IDs, stage mapping, requested GPU count, Slurm state, and
DataLoader worker process IDs are not stored. Direct representation and
target-order fields define the checkpoint. Loading reconstructs stage
ownership from the saved architecture, requested runtime GPU count, and
currently visible devices. The same checkpoint therefore loads on CPU, one
GPU, or several GPUs. Retired two-dimensional or reconstruction-only
checkpoints are not migrated or repaired; retrain them with the current
architecture.
Earlier context-layout and context-free checkpoints also require retraining.
The current direct artifact remains `autoencoder.pt`; no versioned checkpoint
name or migration path is used.

## Configuration

`configs/models/autoencoder.yaml` owns the default seed, device, model,
plasma-context branch, topology task, timestep split, loader,
AdamW, training, and plot settings. The command separately selects the
dataset, output directory, and representation.

The important settings are:

| Setting | Purpose |
|---|---|
| `raw.log_eps` | positive physical VDF floor before `log10` |
| `model.channels` | ordered Conv3d encoder/decoder widths |
| `model.bottleneck_shape` | maximum retained spatial cells on each encoded axis |
| `model.latent_size` | VDF latent width before plasma-context fusion |
| `model.pooling` | conditional three-dimensional pooling convention |
| `model_parallel_gpus` | visible devices used by consecutive model stages |
| `plasma_context.hidden_size` | dense 16-feature branch width before latent fusion |
| `topology.hidden_size` | hidden width of the auxiliary topology head |
| `topology.loss_weight` | multiplier for globally masked Smooth L1 |
| `split` | train and validation fractions plus gap timesteps |
| `loader` | sample batches, worker count, normalization batch size, pinning |
| `optimizer` | AdamW learning rate, weight decay, betas, and epsilon |
| `training` | maximum epochs, patience, and minimum total-loss improvement |

The checked-in configuration uses `model.bottleneck_shape: [10, 10, 10]`,
`model_parallel_gpus: 1`, `plasma_context.hidden_size: 32`,
`topology.hidden_size: 64`, and `topology.loss_weight: 0.1`.
`loader.num_workers` affects loading and preprocessing only; it is
independent from stage placement. Dataset `creation.extraction_n_jobs` is a
separate timestep-extraction process count and does not control training. The
configuration has no 2-D, raw-slice,
downsampling, convolution-dimensionality, topology-input, class-objective,
DDP, world-size, or local-rank option.

## Command

See [Example commands](EXAMPLE_COMMANDS.md#autoencoder-training) for CPU,
one-GPU, local multi-GPU, Turso, and HILE examples. The command accepts
`--model-parallel-gpus N`, which overrides the YAML value. Select `raw` for
the complete `X.npy` VDF or `hermite` for the complete `X_hermite.npy` cube;
three-dimensional and topology-aware behavior are not optional modes.

## Artifacts

A successful run writes exactly the established four artifacts:

```text
autoencoder.pt
metrics.txt
training_history.csv
reconstruction_examples.png
```

`training_history.csv` keeps one row per completed epoch and includes:

```text
epoch
learning_rate
train_reconstruction_loss
train_topology_loss
train_total_loss
validation_reconstruction_loss
validation_topology_loss
validation_total_loss
epoch_seconds
```

`metrics.txt` preserves run information, architecture, dataset partitions,
reconstruction loss by split, reconstruction loss by physical class, and the
training-history summary. It additionally reports the complete
three-dimensional representation, model-parallel stage placement, auxiliary
topology task, combined objective by split, and per-target topology error.
Tables are aggregate only; no JSON report or per-sample topology file is
created.

`reconstruction_examples.png` uses deterministic examples from the restored
best model and preserves original, reconstruction, and absolute-error columns.
Raw examples are first restored as complete physical `(vx, vy, vz)` volumes.
The original volume's full three-dimensional maximum fixes one `vy` index;
the same x-z plane is then displayed for original, reconstruction, and error
through the current physical VDF renderer. It retains physical km/s axes,
same-cell thresholding, one shared `LogNorm`, unmodified `nipy_spectral`, a
grey grid, transparent masks, and white axes. Hermite examples display one
signed coefficient plane with a symmetric `RdBu_r` scale and a nonnegative
error panel. Raw rows are reconstructed sequentially and reduced to copied
display planes before the next row, so the figure path holds one complete raw
pair rather than every selected pair. Plotting never changes full-volume
model values or training loss.

## Memory considerations

A full `268^3` raw sample is far larger than the retired plane input.
Begin raw training and normalization with batch size one and increase only
after measuring host and accelerator memory. Model parallelism distributes
consecutive parameters and activations, but it does not make one Conv3d fit
across devices: every stage and its local backward activations must fit on its
owner. For small Hermite cubes, cross-device activation transfers may cost
more time than they save.
