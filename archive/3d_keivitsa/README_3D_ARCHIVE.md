# Frozen: 3-D Keivitsa prior line

This directory is a **read-only snapshot** of the 3-D EM/IP → resistivity U-Net
pipeline (`UnifiedPriorUNet3D`, tiled inference, GTK Voxelizer, Nagy gravity /
magnetic L-BFGS). It is **not** on the active scientific path.

Do not run these files from `archive/`. Restore them to `src/` first (below).

## Why it was frozen

The project goal is a **2-D MT prior**: station × period apparent resistivity
and phase → `log10(ρ)` grid → `MTGeophysics.jl` VFSA2DMT.

The 3-D Keivitsa line does something else: fuse surface EM/IP + boreholes onto
a 3-D voxel grid and infer a resistivity volume (`m0`, `m_min`, `m_max`). It
never fed COMMEMI / VFSA2DMT and is not integrated with
`train_mt_resistivity.jl`.

Active work is beating the COMMEMI short-VFSA baseline (v7 `commemi_rms`
checkpoint works, but homogeneous and v5 priors are still better). Keeping a
second, unused U-Net as `main.jl` made the default entry point contradict the
README.

`SmartPriorNet3D` (`PriorNet3D.jl`) was already unused (superseded by
`UnifiedPriorUNet3D`). Gravity/magnetic `physics_inversion/` was never called
from the 2-D or 3-D U-Net trainers.

## What moved here

| Path in this archive | Original path |
|---|---|
| `main.jl` | repo-root 3-D orchestrator |
| `src/networks/prior_unet3d.jl` | `UnifiedPriorUNet3D` |
| `src/training/train_prior.jl` | 3-D patch training |
| `src/training/PriorTrainingLoss.jl` | 3-D well / TV / bounds loss |
| `src/fusion/Voxelizer.jl` | GTK XYZ + borehole fusion |
| `src/fusion/GridSpec.jl` | 3-D voxel geometry |
| `src/data/preprocess.jl` | Keivitsa preprocess → HDF5 |
| `src/data/patch_loader.jl` | 3-D patch sampler |
| `src/neural_prior/CBAM3D.jl` | 3-D CBAM (used by the U-Net) |
| `src/neural_prior/PriorNet3D.jl` | unused `SmartPriorNet3D` |
| `src/physics_inversion/*` | Nagy gravity / magnetic + L-BFGS |

Tree layout under `src/` is unchanged so `@__DIR__`-relative includes between
archived files still match after restore.

`PriorTrainingLoss.jl` includes `src/neural_prior/Losses.jl`, which **stayed**
in the live tree. That include works only after restore.

## What stayed in the live tree (do not move)

- `src/analysis/keivitsa_stats.jl` — 2-D synthetic resistivity calibration
- `scripts/build_keivitsa_priors.jl` — writes `config/keivitsa_priors.*`
- `src/neural_prior/Losses.jl` — TV / masked-loss helpers (no 3-D network)
- `src/training/train_mt_resistivity.jl`, `commemi_probe.jl`, `models/`, `results/`

## How to re-open (restore)

From the repository root, with a clean working tree:

```bash
# 1. Put archived modules back next to the 2-D code
git mv archive/3d_keivitsa/src/networks/prior_unet3d.jl src/networks/
git mv archive/3d_keivitsa/src/training/train_prior.jl src/training/
git mv archive/3d_keivitsa/src/training/PriorTrainingLoss.jl src/training/
mkdir -p src/fusion src/data src/physics_inversion
git mv archive/3d_keivitsa/src/fusion/Voxelizer.jl src/fusion/
git mv archive/3d_keivitsa/src/fusion/GridSpec.jl src/fusion/
git mv archive/3d_keivitsa/src/data/preprocess.jl src/data/
git mv archive/3d_keivitsa/src/data/patch_loader.jl src/data/
git mv archive/3d_keivitsa/src/neural_prior/CBAM3D.jl src/neural_prior/
git mv archive/3d_keivitsa/src/neural_prior/PriorNet3D.jl src/neural_prior/
git mv archive/3d_keivitsa/src/physics_inversion/* src/physics_inversion/

# 2. Replace the 2-D wrapper with the 3-D orchestrator
git mv main.jl main_2d_wrapper.jl   # or delete if you no longer need it
git mv archive/3d_keivitsa/main.jl main.jl
```

Alternatively revert the archive commit on `phase2-archive-3d` in one step:

```bash
git revert <archive-commit-sha>
# or, if this branch was never merged:
git checkout master
```

After restore, 3-D training / inference:

```bash
julia --project=. src/training/train_prior.jl
julia --project=. main.jl
```

## Active 2-D entry (current)

```bash
julia --project=. main.jl
# equivalent:
julia --project=. examples/run_mt_prior_inversion.jl
```
