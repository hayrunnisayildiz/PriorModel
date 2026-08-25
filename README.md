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

Open [`notebooks/colab_setup.ipynb`](notebooks/colab_setup.ipynb) and run the
cells in order. Current default: TE+TM
`train_pairs_v8_tetm_n200.h5` → `models/prior_v8_tetm_n200.jld2`. Keep the
kernel as **Python 3**; Julia is invoked with
`julia --project=/content/PriorModel …`. VFSA / `main.jl` stay local
(MTGeophysics → GLMakie hangs a headless VM).

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
and keep artifacts under the same layout as the repo (`data/synthetic/*.h5`,
`models/*.jld2`). The notebook searches both
`/content/drive/MyDrive/PriorModel/` and the nested path
`MyDrive/PriorModelData/PriorModel/PriorModel/`.

**`cd /content/PriorModel` first, then `--project=.`.** Running `--project=.`
from `/content` creates an empty `/content/Project.toml` which hides this
repo. Delete leftover `/content/Project.toml` if it exists.

```bash
cd /content/PriorModel
julia --project=. -e 'include("src/pkg_setup.jl")'
```

### 3. Data and checkpoints

`data/synthetic/*.h5` and `models/*.jld2` are gitignored. Do **not** reuse
`train_pairs_v8_tetm_n200.h5` (that file is the discarded dx=80 run). Build:

```bash
cd /content/PriorModel
xvfb-run -a julia --project=. scripts/build_train_pairs.jl --n 200 --seed 42 --tetm \
  --out data/synthetic/train_pairs_v8_tetm_n200_dx160.h5
```

The production log must show `UNET_MESH: nx=120  dx=160.0 m` (v4–v7 contract).
A raw `julia … build_train_pairs.jl` on Colab loads MTGeophysics → GLMakie
and hangs unless wrapped in `xvfb-run`.

### 4. Train (COMMEMI probe on)

`train_mt_resistivity.jl` stubs GLMakie so `--commemi-every 5` is safe on
headless Colab. Do **not** pass `--commemi-every 0` (that was the discarded
run: val_loss fallback, Phase 1 probe disabled).

```bash
cd /content/PriorModel
julia --project=. src/training/train_mt_resistivity.jl \
  --dataset data/synthetic/train_pairs_v8_tetm_n200_dx160.h5 \
  --epochs 15 --commemi-every 5 \
  --output models/prior_v8_tetm_n200_dx160.jld2 \
  --training-log results/training_log_v8_tetm_n200_dx160.csv \
  --split-json results/train_val_split_v8_tetm_n200_dx160.json \
  --curve-png results/training_curve_v8_tetm_n200_dx160.png
```

First epoch compiles CUDA/Zygote; 10–20 min with no output is normal.

Scripts share one bootstrap (`src/pkg_setup.jl`) that activates `ROOT` after
`cd /content/PriorModel`.
