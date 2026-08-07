# Autoencoder

## Purpose

The autoencoder learns a compact latent representation by reconstructing the
selected VDF representation. It supports raw and Hermite inputs.

It is:

- deterministic;
- convolutional;
- trained with mean squared reconstruction error; and
- intended for representation diagnostics.

It is not variational, adversarial, or currently a generative model.

## Inputs

### Raw

Raw reconstruction intentionally retains the autoencoder-specific
two-dimensional plane:

```text
X.npy
    -> full middle-vy [vx, vz] plane
    -> positive floor
    -> log10
    -> one-channel 2-D tensor
```

Dimensions are inferred from the velocity grid. There is no downsampling.
This existing autoencoder behavior is isolated from the shared raw
PCA/CNN path: raw PCA and CNN use the complete three-dimensional VDF.
Reconstruction plotting does not change that target plane. It reverses the
training transform to physical phase-space density and passes the plane to
the same physical preparation and drawing functions used by extraction Stage
6, rather than using an autoencoder-specific heatmap.

### Hermite

Hermite reconstruction uses the complete coefficient volume:

```text
X_hermite.npy
    -> derive actual (order, order, order) shape
    -> [n_x, n_y, n_z] when unrotated or
       [n_parallel, n_perp1, n_perp2] when rotated
    -> one-channel 3-D tensor
```

The volume is not flattened before the encoder, sliced, transposed,
logged, thresholded, normalized per sample, or augmented. Its dimensional
coefficients come directly from the physical VDF. The default extraction
order is 22, while the autoencoder derives its input and reconstruction
shape from `X_hermite.npy`.

The model receives no label, topology, coordinate, magnetic-field, velocity,
VDF sparsity-threshold, moment, or diagnostic input. It reconstructs only
the chosen representation.

## Normalization

The same streaming feature-normalization implementation used by CNN training
fits one mean and scale per raw pixel or Hermite coefficient:

- accumulation is float64;
- training indices are the only fit input;
- near-zero variance uses scale one;
- statistics are stored in the checkpoint; and
- normalization is applied exactly once.

PCA standardization statistics are not reused or saved as model artifacts.

## Architecture

Raw uses Conv2d/ReLU stages and Hermite uses corresponding Conv3d/ReLU
stages. Conditional average pooling reduces spatial dimensions without
assuming a fixed source mesh. The encoder output is projected to a configured
latent vector.

The decoder projects the latent vector back to the encoded shape, uses
bilinear or trilinear interpolation to reverse pooled stages, and restores
one channel at exactly the original input shape.

The default configuration uses channels `[8, 16, 32]`, latent dimension 64,
and average pooling.

The forward result contains:

```text
reconstruction
latent
```

## Training

The autoencoder uses the same chronological timestep split as CNN
training.

Training uses ordinary shuffled batches, stable validation/test order,
AdamW, one device, and early stopping. The sole objective is unweighted mean
squared error in normalized representation space. Validation MSE selects the
best epoch. After restoring that selected model, evaluation calculates one
MSE per sample by averaging squared reconstruction error over every non-batch
representation dimension. Those values are aggregated by chronological split
and by the dynamic physical classes in `metadata.csv`.

Physical class identity is reporting-only. It is not yielded in model
batches and never enters the autoencoder input, loss, optimizer, early
stopping, latent representation, or checkpoint selection.

The previously used one-timestep BCH verification dataset can support a
bounded forward reconstruction but not a leakage-safe training split.
Use a genuine multi-timestep dataset; no random sample-level fallback
exists.

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

`load_autoencoder_checkpoint.py` is a reusable reconstruction loader rather
than a numbered training stage.

## Checkpoint

`autoencoder.pt` is the autoencoder checkpoint. It stores:

- representation and representation name;
- input shape and, for Hermite, actual order and optional rotation setting;
- input mean, scale, and epsilon;
- channel, pooling, and latent architecture;
- model state;
- random seed.

The autoencoder checkpoint is independent from the CNN checkpoint. The command
selectors remain `raw` and `hermite`; the raw autoencoder checkpoint records
its autoencoder-owned plane convention. Loading accesses the current
checkpoint fields directly and reconstructs the model and feature scaler.
Hermite checkpoints trained on the retired compact-log or fixed-order
representation are not adapted; regenerate the dataset and retrain.

## Configuration

`configs/models/autoencoder.yaml` owns the default seed, device, model,
timestep split, loader, AdamW, and training settings. The command selects
the dataset, output directory, and raw or Hermite representation.

The important sections are:

| Section | Purpose |
|---|---|
| `raw` | positive `log_eps` for nonpositive raw working values |
| `model` | raw/Hermite channels, latent dimension, average pooling |
| `split` | train and validation fractions plus gap timesteps |
| `data_loader` | batch sizes, workers, normalization batch size, pinning |
| `optimizer` | AdamW learning rate, weight decay, betas, epsilon |
| `training` | maximum epochs, patience, minimum validation-MSE improvement |
The supplied model uses raw and Hermite channels `[8, 16, 32]`, latent
dimension 64, and average pooling. Loader batch size defaults to one and
normalization batch size to two.

The configuration has no class, topology, mass, peak, high-signal,
variational, adversarial, balancing, sampler, or filtering options. Devices
are `cpu`, `cuda`, `cuda:N`, and `auto`.

Comments immediately above each YAML key are the quickest reference for
accepted choices, reconstruction effects, and memory-sensitive settings.

## Command

See [Example commands](EXAMPLE_COMMANDS.md#autoencoder-training) for raw,
Hermite, CPU, CUDA, and Slurm variants.

Select `raw` for the 2-D model or `hermite` for the 3-D model.

Training uses only whole-timestep partitions and never falls back to
sample-level random partitioning.

## Artifacts

A successful run writes:

```text
autoencoder.pt
metrics.txt
training_history.csv
reconstruction_examples.png
```

`metrics.txt` is a plain UTF-8 report with these sections:

1. autoencoder training run;
2. model architecture;
3. dataset partitions;
4. reconstruction loss by split;
5. reconstruction loss by physical class; and
6. training history summary.

The split and class tables contain sample count, mean MSE, median MSE,
minimum MSE, and maximum MSE. All losses use the unchanged normalized-space
objective and the restored validation-selected model. The report does not
save per-sample losses. `training_history.csv` retains one row per completed
epoch with train/validation losses and the selected-epoch flag. Files are
saved directly without a post-save inspection pass.

`reconstruction_examples.png` is created from the already restored
validation-selected model. Deterministic examples retain the historical
split/class ordering: early training rows and the largest reconstruction
errors from validation and test are considered first, then each alphabetic
physical class contributes up to `plot.max_per_class` rows. Each selected row
shows the original representation, reconstruction with matching color limits,
and absolute error.

For raw examples, inverse preprocessing restores physical phase-space density
from the normalized log-space model input. Both original and reconstruction
then use the extraction Stage 6 VDF preparation and drawing functions with
the same sample threshold, velocity-cell width, physical km/s axes, grey
grid, `LogNorm`, and unmodified `nipy_spectral` colormap. The original fixes
the displayed x-z plane and logarithmic limits for its reconstruction;
below-threshold and nonpositive values remain transparent over white axes.
The current raw model input is already a two-dimensional middle-`vy` plane,
so plotting uses it directly instead of inventing a different slice. The raw
absolute-error panel remains nonnegative and also uses a white axes
background.

Hermite examples continue to display a signed central third-axis coefficient
plane with a symmetric `RdBu_r` scale shared by original and reconstruction.
The displayed axes are `(n_x, n_y)` when unrotated or
`(n_parallel, n_perp1)` when rotated, and the selected index follows the
actual saved shape. Their nonnegative absolute-error panels and all
surrounding axes use white backgrounds; physical VDF `LogNorm` is not
applied to signed coefficients. Physical class names appear only in titles
and selection metadata, never in inference, loss, or checkpoint selection.
No reconstruction arrays or per-sample files are saved.
