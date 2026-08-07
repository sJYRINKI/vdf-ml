# Historical workflows

## Status

Executable legacy and deprecated implementations were removed from the
current source tree. The current checkout supports only `src` and the
eight module commands documented in the [root README](../README.md).

## Branch layout

The current working implementation is published on the `main` branch. The
previous historical implementation is preserved separately on the `legacy`
branch. `main` is the supported version for current development and
scientific workflows, while `legacy` retains the earlier code for read-only
reference and historical reproduction.

This branch separation does not make legacy modules part of the active
`main` runtime. Current code must continue to use `src` and must not
import implementation from the `legacy` branch.

## Command-line interface

CLI means command-line interface: the terminal commands used to start the
project workflows. On `main`, the supported CLI entry points live under
`scripts/` and call the active implementation in `src`. They parse
command-line arguments, load the selected configuration, and start the
corresponding extraction, analysis, training, prediction, or plotting
workflow. Historical CLI commands remain part of the `legacy` branch and
are not supported by `main`.

## Concise refactor history

The lasting changes were:

- dataset rows, metadata, labels, topology, and velocity-grid ownership were
  made explicit;
- raw VDF storage was separated from full-resolution `raw_xz_full` feature
  construction;
- the `hermite` transform was frozen around cell-centred magnetic field,
  cleaned-VDF first-moment drift, exact same-cell `MinValue`, and bounded
  density error;
- unsupervised analysis was narrowed to StandardScaler, PCA, fixed KMeans,
  physical-class purity/composition, and t-SNE;
- one raw/Hermite CNN was retained, with topology as six auxiliary outputs
  rather than inputs;
- timestep-aware CNN training, a CNN checkpoint, coordinate prediction,
  and streamed region prediction were added;
- a deterministic raw/Hermite reconstruction autoencoder with its own
  checkpoint was added;
- prediction figures combined spatial context, class predictions, X/O
  circles and vectors, and peak-centred source-VDF cuts;
- active implementation was isolated under `src`;
- raw downsampling and executable historical compatibility were removed;
- public names were simplified to `raw_xz_full`, `hermite`, and
  `legacy_b_polar_2d`; and
- saved-dataset audit and validation machinery was removed in favor of
  trusted current files, direct loading, minimal VLSV producer checks, and
  transactional output safety.

The current project also restores focused dataset colormap and physical
VDF-slice plotting by porting only useful visual behavior into
`src`. Git history remains the full record of intermediate reports
and removed implementations.
