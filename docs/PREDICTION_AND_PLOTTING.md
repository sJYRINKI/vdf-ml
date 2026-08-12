# Prediction and plotting

## Overview

Coordinate and region prediction load a CNN checkpoint, construct
the recorded representation and same-cell plasma context from a source
VDF, run bounded two-input
inference, inverse-scale topology outputs, and save structured results.

The model receives the representation tensor plus 16 scaled context values:
`(Bx, By, Bz)`, `(Ex, Ey, Ez)`, `(Vx, Vy, Vz)`, number density, and
`(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz)`. The vector components preserve both
direction and magnitude; no separate magnitude values are added. Requested
and selected coordinates, the resolved VDF sparsity threshold, labels, and
topology are provenance or outputs, never inputs.

## Checkpoint contents

Prediction directly reads:

- checkpoint metadata;
- `raw` or `hermite` representation;
- representation name and input shape;
- input normalization;
- exact plasma-context feature order and training mean and scale;
- class mapping and output order;
- exact six-target order and topology scaler;
- architecture;
- raw training velocity grid or Hermite actual shape, order, and optional
  rotation setting.

The checkpoint is expected to have been written by the current training
workflow and to use `raw` or `hermite`.

## Ordered shared stages

Coordinate and region orchestration share:

```text
step_01_load_cnn_model.py
step_02_load_source_vdf.py
step_03_prepare_prediction_input.py
step_04_run_cnn_prediction.py
step_05_save_prediction_outputs.py
```

`coordinate_prediction.py` and `region_prediction.py` call these stages for
their respective spatial selection and output loops.

## Raw prediction

Raw preprocessing follows:

```text
source physical VDF
    -> inspect source velocity grid
    -> trilinear resampling to the training grid when required
    -> complete [vx, vy, vz] physical VDF
    -> positive floor applied to every velocity-space cell
    -> log10
same-CID B/E/V/density/pressure -> 16 context values
    -> [1, 1, vx, vy, vz] VDF batch plus [1, 16] context batch
    -> checkpoint normalization
    -> Conv3d CNN
```

Interpolation occurs on physical VDF values before logarithms. Grid
coordinates are cell-centred, and values outside the source grid are zero.
The source VDF is not mutated. No middle plane, projection, crop, average,
or downsample is used as the raw model input.

## Hermite prediction

Hermite preprocessing follows:

```text
source physical VDF
    -> original [vx, vy, vz] grid, or optional checkpoint-recorded rotation
       from transient same-CID B and total bulk velocity
    -> endpoint velocity axes and physical-VDF drift/thermal moments
    -> physical-velocity physicists' Hermite basis
    -> complete checkpoint-order dimensional coefficients using dv**3
same-CID B/E/V/density/pressure -> 16 context values
    -> one-channel tensor plus [batch, 16] context
    -> checkpoint normalization
    -> CNN
```

No logarithm, `MinValue` threshold, clipping, or coefficient normalization
is applied. Prediction always reads magnetic field and fluid velocity for
the context, but an unrotated Hermite transform does not otherwise use those
vectors. When rotation is true, it reuses the same vectors without
a second source read and reproduces the checkpoint's
`(parallel, perpendicular bulk flow, cross-product)` frame before calculating
the VDF moments and coefficients. Prediction does not resample an already
generated coefficient cube or create a rotated cache. Region prediction
requires one full Hermite transform at a time.

Producer names are resolved once for each open prediction VLSV reader.
SpatialGrid values use the selected CID; field-grid B/E values are centred and
sampled at that CID's cell centre. Analysator's complete source-grid
allocation is temporary: prediction retains only selected vectors, releases
each grid before reading the next producer, and discards each selected vector
after its sample is prepared. Region prediction stacks
context rows in the same order and batches as VDF tensors. It never reads a dataset
`plasma_context.npy`, and context values are not appended to the prediction
CSV or figures.

The producer priorities match extraction: magnetic field uses
`vg_b_vol`, `B_vol`, `fg_b_vol`, `fg_b`, `B`; electric field uses
`vg_e_vol`, `E_vol`, `fg_e_vol`, `fg_e`, `E`; configured-population velocity
uses direct `vg_v`/`V` families before historical `rho_v / rho`; density uses
configured-population `vg_rho`/`rho` before unprefixed `rho`; and pressure
uses a complete total tensor or the supported thermal/nonthermal or
backstream/nonbackstream sum. This keeps training and live prediction
scientifically identical without storing runtime producer names in the
checkpoint.

Hermite checkpoints created under the retired compact-log, fixed-order
convention are not compatible with current prediction. Regenerate the
Hermite dataset and retrain the CNN instead of reusing those checkpoints.

## Prediction values

Each result contains:

- predicted class ID and name;
- maximum class probability;
- one probability per checkpoint class;
- `predicted_distance_to_x_point_re`;
- `predicted_distance_to_o_point_re`;
- `predicted_vdf_to_x_point_dx_re`;
- `predicted_vdf_to_x_point_dz_re`;
- `predicted_vdf_to_o_point_dx_re`; and
- `predicted_vdf_to_o_point_dz_re`.

For a checkpoint trained with the complete current taxonomy, the probability
fields are:

```text
probability_lobe
probability_current_layer
probability_o_point
probability_x_point
probability_solar_wind
probability_magnetosheath
probability_inner_magnetosphere
```

These columns are generated from the checkpoint mapping rather than a
prediction-side class list. A checkpoint containing a dataset subset
therefore exposes exactly its recorded classes in recorded output order.

Topology predictions are inverse-transformed to Earth-radii physical units.
Scaled training-space values are not presented as physical output.

## Coordinate prediction

See [Example commands](EXAMPLE_COMMANDS.md#coordinate-prediction) for local
and Slurm invocations.

The coordinate workflow:

1. resolves one VLSV source;
2. accepts a requested coordinate in `R_E`;
3. selects the nearest VDF-carrying cell;
4. records both requested and selected cell-centre coordinates;
5. constructs the checkpoint representation;
6. runs inference; and
7. writes one common-schema CSV row; then
8. optionally renders the combined figure after the CSV exists.

The checkpoint determines the representation. `--plot` and `--no-plot`
override configured plotting.

Output:

```text
predictions_<timestep>.csv
prediction_combined.png        when plotting is enabled
```

The CSV has the same row schema and writer as region prediction. It includes
the requested x/y/z position, selected VDF-cell centre, timestep, CID,
predicted class ID/name, `predicted_probability`, every class probability,
and all six topology outputs. The requested coordinate is never substituted
for the selected-cell coordinate. No duplicate JSON output is written.

CSV serialization precedes optional plotting, so a later ordinary rendering
exception does not erase the successful numerical prediction.

## Region prediction

See [Example commands](EXAMPLE_COMMANDS.md#region-prediction) for local and
Slurm invocations.

The region workflow selects VDF cells inside configured `R_E` bounds,
preserves stable CID order, processes bounded batches, and streams one CSV
per timestep. It does not retain all region VDFs or Hermite volumes in
memory.

Output:

```text
predictions_<timestep>.csv
predictions_<timestep>_combined.png   when enabled
```

Rows include timestep, source, CID, selected-cell coordinates,
representation, class prediction, probabilities, and six topology
predictions. The common schema also contains requested x/y/z columns; they
hold the coordinate request in a coordinate run and remain empty for region
rows. Normal prediction does not calculate ground-truth topology.

## Combined figures

Where a prediction figure draws raw physical VDF panels, it reuses the
shared historical plane preparation and drawing path. Vlasiator generations
name the same physical sparsity threshold differently, so source loading
selects the first available producer in this Analysator order:

```text
MinValue
<population>/MinValue
<population>/EffectiveSparsityThreshold
<population>/vg_effectivesparsitythreshold
```

The selected name is retained once per open VLSV source and reused for every
displayed CID. Its value remains in physical VDF units of `s^3 m^-6`; there
is no numerical threshold fallback. For each cut the
renderer calculates `dv = (vymax - vymin) / nvy`, multiplies a float32 plane
copy by `dv`, sets values strictly below `threshold * dv` to zero, and masks
nonpositive values. It uses unmodified `nipy_spectral` and `LogNorm` over the
visible support. The default bad mask stays transparent over white axes; no
black bad or under-color is forced. Velocity axes are km/s, the grid and
colorbar remain visible, and neutral/default spines replace the previous
colored panel boundaries. This plotting copy is separate from raw resampling
and model preprocessing, so the complete saved VDF and raw or Hermite
machine-learning input are unchanged.

Both spatial panels show x-directed bulk velocity in m/s. The shared resolver
prefers `<population>/vg_v`, `<population>/V`, `vg_v`, `V`, then `fg_v`; only older
files without direct velocity use `rho_v / rho`. EGI therefore uses
`proton/vg_v`, BFA-style files use `proton/V` when available, and BCH-style
files use the legacy ratio. Thermal, nonthermal, backstream, and drift
velocities are not total-flow substitutes. The linear `bwr` range is fixed
symmetrically at `[-1.5e6, +1.5e6]` m/s, and black magnetic streamlines
remain. The default view is `[-30, 30, -15, 15]` in x-z Earth radii. That
plot box is independent of region prediction bounds. A compact legend has
its own frameless row immediately below the colormap.

### Coordinate figure

The top panel contains:

- the shared x-velocity background and black streamlines;
- a compact blue star at the user-requested coordinate (scatter area 12);
- the selected VDF cell in its compact predicted-class marker and color
  (scatter area 8);
- X/O scalar distance circles centred on the selected VDF cell;
- a dedicated legend below the colormap; and
- a title formatted as `cid=<cid> | class=<name> | p=<probability>` with
  probability shown to three decimals.

No connector, topology arrow, inferred point endpoint, or vector-component
artist is drawn.

The lower row contains three raw source-VDF cuts through that VDF’s own peak:

- `vx-vy`;
- `vx-vz`; and
- `vy-vz`.

Hermite checkpoint plots still show the physical source VDF.
These two-dimensional cuts are scientific visualizations only. They do not
define or modify the complete three-dimensional raw CNN input.

### Region figure

Only the bounded deterministic representatives whose VDF panels are shown
are marked on the spatial map. Each uses its stable predicted-class marker
and color with compact scatter area 8. No requested-coordinate star,
distance circle, adjacent class-name text, connector, topology arrow,
inferred endpoint, or vector-component artist is drawn. The concise spatial
title identifies the figure as region predictions without listing each
representative's CID, class, or probability. The dedicated legend below the
colormap remains the only spatial-map key for class marker shapes and colors;
legend symbols may be larger than the underlying map markers for readability.

Representatives are selected from the already written prediction rows. The
default selects the highest-confidence sample per predicted class in
checkpoint order.

Lower rows show source VDF cuts for those exact representative CIDs. Plotting
does not rerun the model and rereads only bounded representative VDF data.

## Topology distances

`predicted_distance_to_x_point_re` and
`predicted_distance_to_o_point_re` are inverse-scaled physical predictions
retained in CSV and model results. Each positive distance is rendered as an
unfilled circle in Earth radii around the selected VDF cell in coordinate
figures. The X circle is red and dashed; the O circle is blue and dotted.
Region maps retain these numerical columns in CSV but do not render either
circle.

## Topology values and rendering

Both CSV workflows retain the two predicted distances and four predicted
x-z displacement components in Earth radii. These values remain part of CNN
output and numerical prediction results. Coordinate figures use only the two
scalar distances for direction-free circles; region figures do not render
topology geometry. Neither mode draws displacement arrows, inferred
endpoints, or source-to-cell connectors. This rendering does not alter,
normalize, clamp, or discard topology predictions.

## Class colors and representatives

Colors and markers follow checkpoint physical-class order and remain
identical in:

- spatial markers;
- legends;
- coordinate figures; and
- region figures.

Extraction-time Stage 6 and the standalone dataset-colormap command use one
shared renderer with role colors instead of the prediction palette: all
VDF-carrying cells are yellow, configured manual and detected X/O source
coordinates are blue, and selected dataset cells are red. Each accepted
coordinate and its selected VDF cells use the same deliberately small marker
shape, so shape identifies the current class or point kind and color identifies
the plotting role. `current_layer` is red-only because it has no configured
source coordinate. The spatial background is x-directed bulk velocity in m/s
on the symmetric `bwr` interval
`[-1.5e6, +1.5e6]`, with thin black magnetic-field streamlines and a compact
multi-column legend below the axes. Standalone visibility flags may hide
individual layers, but their defaults reproduce the Stage 6 figure.

Only accepted point records reach either renderer. A point with an empty final
manual, physical, consensus, or union selection contributes no marker, search
area, legend entry, sample, or animation frame. If an X and O record share any
final selected VDF cell, both complete records are omitted, including their
otherwise unique cells. No X/O priority or overlap class resolves the conflict.

Both callers use the exact physical X, manual X, physical O, and manual O
geometry from planning. For union selection, any shared manual/physical VDF
cell makes the physical area the only displayed outline; manual-only selected
cells remain red and in the dataset. Disjoint union sets display both
nonempty areas. Manual, physical, and consensus display behavior is unchanged.
Neither caller approximates or recomputes selection membership.

Per-cell-ID Stage 6 VDF frames use one square logarithmic `vx-vz` panel at
the sample's own peak `vy` index. The velocity axes are km/s, the Cartesian
grid is grey, and the vertical colorbar is labelled `f(v)`. The standalone
VDF command reuses the same historical preparation and renderer. With no
`--plane` option it draws `xz`, `xy`, and `yz`, each crossing the sample's
full three-dimensional peak, in that order; repeated `--plane` options select
one or more planes in the requested order. Stage 6 remains fixed to the
`xz`/peak-`vy` plane, while prediction figures reuse the physical-panel path
where applicable within their separate plotting composition.

Representatives are stable, unique by CID, bounded by configuration, and
selected from existing rows. Supported strategies include highest confidence
per class, highest confidence overall, first stable sample per class, and
explicit configured CIDs.

## Device and memory behavior

Devices are `cpu`, `cuda`, `cuda:N`, or `auto`. The default uses one device
without `DataParallel`. `--model-parallel-gpus N` reconstructs the same
checkpoint with consecutive CNN stages on several visible GPUs while still
using one outer Python process. Only activations cross stage boundaries;
final class and topology arrays return to CPU through the unchanged
prediction path.

Complete raw volumes and Hermite transforms are memory-sensitive. Use small
region batches—starting with one complete VDF at a time—and a new output
directory. Actual memory and runtime depend on the source grid, checkpoint
architecture, device, and requested region; bounded verification does not
establish a production resource requirement.

## Runtime assumptions

Prediction assumes the configuration, current checkpoint, source
VLSV variables, model outputs, and destination are usable. It loads expected
fields and saves CSV and PNG files directly. Ordinary exceptions from
PyTorch, NumPy, Analysator, Matplotlib, and the operating system propagate
without project-specific translation.

## Configuration

`configs/prediction/coordinate.yaml` and
`configs/prediction/region.yaml` own their respective workflows.

Both configurations select a source template and population plus
combined-figure settings. Region configuration additionally owns the
inclusive Earth-radii selection bounds and inference batch size. The command
supplies the current checkpoint, output location, timestep or timestep
range, requested coordinate for coordinate prediction, and device. The
checkpoint owns `raw` or `hermite`, physical-class output order,
representation normalization, exact plasma-context feature order and training
mean and scale, and the raw training velocity grid or actual Hermite
shape/order/rotation. Raw resampling is trilinear on physical VDF values and
uses zero outside the source grid.

Prediction figures have one fixed x-velocity background, plot box, title,
marker, circle, and below-map legend behavior rather than configuration
switches for alternate density backgrounds, topology vectors, or circle
styles. The plotting section
retains enablement, streamline, DPI, VDF velocity-limit, and bounded region
representative settings. The coordinate workflow uses one representative.
Region selection uses its separate stable configured `R_E` x, y, and z
bounds; its batch size and outer-process count select execution sizes.
Hermite region prediction uses batch size one, and CUDA prediction uses one
outer process.

CLI `--file-source` may override source selection. `--plot` and `--no-plot`
override only configured plotting enablement. The checkpoint remains
authoritative for representation, model, representation and plasma-context
scaling, class-output order, and topology scaling. Comments in both YAML files are the quickest reference
for coordinate units, plotting switches, and command-owned values.
