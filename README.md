# PriorModel

MT apparent-resistivity / phase curves → 2-D log₁₀ρ prior, then VFSA2DMT
(`MTGeophysics.jl`) inversion. Lux.jl U-Net (`MTResistivityUNet2D`) is trained
on synthetic TE/TM pairs generated on [`UNET_MESH`](src/synthetic/MeshParams.jl).

`main.jl` is a thin wrapper around
[`examples/run_mt_prior_inversion.jl`](examples/run_mt_prior_inversion.jl)
(neural prior → VFSA2DMT vs homogeneous halfspace). The 3-D Keivitsa U-Net
line is frozen under [`archive/3d_keivitsa/`](archive/3d_keivitsa/README_3D_ARCHIVE.md).

## Pipeline

1. `scripts/build_train_pairs.jl` — synthetic resistivity + forward MT → HDF5
2. `src/training/train_mt_resistivity.jl` — train U-Net (80/20 split, cosine LR,
   grad clip). Checkpoint is selected by periodic COMMEMI short-VFSA
   `commemi_rms` (synthetic `val_loss` is fallback).
3. `src/inference/export_prior_to_mtgeophysics.jl` — COMMEMI `.obs` → prior `.ini`
4. `scripts/evaluate_mid_scale_v*.jl` — VFSA2DMT vs homogeneous / previous priors
5. `julia --project=. main.jl` — E2E: prior `.ini` + VFSA (delegates to
   `examples/run_mt_prior_inversion.jl`; same CLI flags)

Canonical survey (v4+): 30 stations × 20 periods, `T ∈ [10⁻³, 10³]` s,
profile 120 × 160 m (19.2 km) so COMMEMI `±8000` m is in-distribution.

## Known limitations

See [`results/KNOWN_LIMITATIONS.md`](results/KNOWN_LIMITATIONS.md) for the v4
wide-window failure, the v5→v6 val-loss / COMMEMI-RMS mismatch, and the v7
response (`n=1000`, COMMEMI-family scenarios, `commemi_rms` checkpoint).

## Running on Colab

Open [`notebooks/colab_setup.ipynb`](notebooks/colab_setup.ipynb) and follow the
cells in order. The same steps from a Colab Python runtime:

### 1. GPU runtime

`Runtime → Change runtime type → Hardware accelerator → GPU` (T4 is enough).
Then install Julia **1.12.4** (Colab's built-in Julia runtime is 1.10 LTS and
does not match this project):

```bash
JULIA_VERSION="1.12.4"
JULIA_VER="${JULIA_VERSION%.*}"
wget -q "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VER}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
  -O /tmp/julia.tar.gz
tar -xzf /tmp/julia.tar.gz -C /usr/local --strip-components=1
rm /tmp/julia.tar.gz
julia --version
```

### 2. Clone and Drive mount

```bash
export GIT_TERMINAL_PROMPT=0
git -c credential.helper= clone --depth 1 --single-branch --branch master \
  https://github.com/hayrunnisayildiz/PriorModel.git /content/PriorModel
```

Do not let `git` prompt for a username: a public HTTPS clone waiting on credentials is the usual reason the Colab cell never finishes. If clone still stalls, download the public zip from the same URL (`…/archive/refs/heads/master.zip`). The notebook cell does that automatically after 90 s.

Mount Drive from a Python cell (`from google.colab import drive; drive.mount('/content/drive')`)
and keep artifacts under `/content/drive/MyDrive/PriorModel/` with the same
layout as the repo (`data/synthetic/*.h5`, `models/*.jld2`).

**Never use `--project=.` on Colab.** The notebook cwd is `/content`. That flag
creates a leftover empty `/content/Project.toml` which then hides this repo's
environment, so `using HDF5` / `using LuxCUDA` fail. Always pass the clone path:

```bash
julia --project=/content/PriorModel -e 'using Pkg; Pkg.instantiate()'
```

If `/content/Project.toml` already exists, delete it.

### 3. Data and checkpoints

`data/synthetic/*.h5` and `models/*.jld2` are gitignored (they are large). Copy
them from Drive with the notebook Python cell (not Julia). On Colab do **not**
synthesize with `--n 50`: that loads MTGeophysics → GLMakie and the cell hangs.

Drive layout on this account (nested):

`MyDrive/PriorModelData/PriorModel/PriorModel/data/synthetic/`

Copy `train_pairs_v7.h5` for training. The 5 MB `train_pairs.h5` is only a smoke test.

### 4. Train (headless)

`MTGeophysics.jl` v0.4.2 lists **GLMakie as a hard dependency** and runs
`using GLMakie` at package load. That cannot be patched from this repo; on a
headless Colab VM the import hangs or crashes. Disable the COMMEMI VFSA probe
and skip the training-curve PNG:

```bash
julia --project=/content/PriorModel \
  /content/PriorModel/src/training/train_mt_resistivity.jl \
  --dataset /content/PriorModel/data/synthetic/train_pairs_v7.h5 \
  --commemi-every 0 --no-plot
```

`--commemi-every` defaults to 10 locally (probe every 10 epochs). The probe
itself uses the plot-free `run_mt2d_vfsa` API (CairoMakie figures are only
written by the `VFSA2DMT()` wrapper). The Colab failure is the GLMakie import
inside MTGeophysics, not our plotting. Keep `--commemi-every 0` on Colab.

Scripts share one bootstrap (`src/pkg_setup.jl`) that activates `ROOT` no matter
which directory you start Julia from, as long as `--project` points at the clone.
