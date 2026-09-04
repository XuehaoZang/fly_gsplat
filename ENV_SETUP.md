# Environment Setup Guide

This document walks an external collaborator through setting up a working environment for `fly_gsplat` from a clean machine. It supersedes, for the `fly_gsplat` environment specifically, the informal notes in `.legacy/env_setup/env_setup.md`.

`fly_gsplat` reconstructs 3D Gaussian splats of fruit flies from 4-camera lab video using [Nerfstudio](https://docs.nerf.studio/) (`splatfacto` method) and [gsplat](https://github.com/nerfstudio-project/gsplat) as the CUDA rasterizer backend.

---

## 1. System Requirements

**3D Gaussian Splatting is sensitive to the exact PyTorch/CUDA build combination**, because `gsplat` ships prebuilt CUDA kernels tagged to a specific (Python, PyTorch, CUDA) triple. Do not casually substitute different versions of PyTorch, CUDA, or Python without also finding a matching `gsplat` wheel (or building it from source).

| Component | Required / Verified | Notes |
|---|---|---|
| OS | Linux (Ubuntu 22.04/24.04 verified). WSL2 on Windows also verified and fully supported. | The project was developed on WSL2 (Ubuntu 24.04.3) with a Windows host, but nothing in the codebase is WSL-specific. Native Linux should work identically. |
| Python | **3.10** (verified: 3.10.20) | `pyproject.toml` requires `>=3.10`. Do not use 3.8/3.9 — some dependencies (e.g. `fpsample`, `jaxtyping`) target 3.10+. |
| PyTorch | **2.1.2**, CUDA 11.8 build (`torch==2.1.2+cu118`) | See §3 below re: a known but currently-harmless version mismatch between this and the `gsplat` wheel. |
| torchvision | **0.16.2+cu118** | Must match the PyTorch build exactly (see PyTorch's compatibility matrix if you change the torch version). |
| CUDA Toolkit | **11.8** | Installed *inside the conda environment* (`conda install nvidia/label/cuda-11.8.0::cuda-toolkit`), not system-wide. This isolates the build toolchain from whatever CUDA version is installed system-wide (e.g. this repo's dev machine has CUDA 12.6 at `/usr/local/cuda`, untouched, because `CUDA_HOME` is never set to point at it). |
| NVIDIA driver | Must support CUDA 11.8 or newer (driver's CUDA support is backward compatible). Verified working with driver 595.71 (which supports up to CUDA 13.2). | Run `nvidia-smi` to check your driver's supported CUDA version — it must be ≥ 11.8. |
| gsplat | **1.4.0** (wheel tag `+pt20cu118`) | See §3 — this wheel is built against PyTorch 2.0, but is used here with PyTorch 2.1.2. This is the officially available wheel for gsplat 1.4.0 + CUDA 11.8; it has worked without observed runtime failures in this project's usage, but is a documented point of fragility. |
| nerfstudio | **1.1.5** | Installed via plain `pip install nerfstudio` — do not install a newer major version without checking `models/splatfacto_checkpoint.py` still subclasses cleanly against nerfstudio's `SplatfactoModel` API. |
| GPU / VRAM | Verified on NVIDIA RTX A5000 (24 GB VRAM). **Minimum VRAM has not been characterized** — [TODO: author to confirm minimum VRAM needed for a single-frame `splatfacto` training run; per-frame scenes are small (single fly, ~4 cameras, visual-hull init ~1M points before pruning) so it likely needs well under 24GB, but this has not been measured on smaller hardware.] | Training is per-frame (one fly pose at a time), not one giant scene, so memory needs are expected to be modest relative to typical NeRF/3DGS scene reconstruction — but treat this as unverified until confirmed. |
| Disk / build tools | `g++`, `ninja-build` (system packages) | Required to JIT/AOT-compile `gsplat`'s CUDA kernels and other C++ extensions at install time. |

If you have multiple GPUs, note that `gsplat`/`nerfstudio` will use whichever GPU is visible as `cuda:0` by default; use `CUDA_VISIBLE_DEVICES` to pin a specific device.

---

## 2. Step-by-Step Installation

These steps assume a clean Ubuntu machine (native or WSL2) with an NVIDIA GPU and driver already installed (`nvidia-smi` should already work at the OS level before you start).

### 2.1 System packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y g++ ninja-build git
```

### 2.2 Install Conda (if not already present)

```bash
wget https://repo.continuum.io/archive/Anaconda3-2025.06-1-Linux-x86_64.sh -O ~/anaconda_installer.sh
bash ~/anaconda_installer.sh
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda init
# restart your shell (or `source ~/.bashrc`) after this
```

Miniconda works equally well if you prefer a lighter install; swap the installer URL accordingly.

### 2.3 Create the conda environment

```bash
conda create -n fly_gsplat python=3.10 -y
conda activate fly_gsplat
```

### 2.4 Install CUDA 11.8 toolkit *inside* the environment

This gives you a self-contained `nvcc` 11.8 for building CUDA extensions, independent of whatever CUDA version (if any) is installed system-wide.

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
```

Verify it took precedence over any system CUDA:

```bash
which nvcc          # should print .../envs/fly_gsplat/bin/nvcc
nvcc --version       # should report release 11.8
```

If `which nvcc` points somewhere else (e.g. `/usr/local/cuda/bin/nvcc`), your `PATH` has a system CUDA installation ahead of the conda environment's `bin/` — fix your `PATH` ordering (conda environments should prepend themselves automatically on `activate`; check `echo $PATH` if this doesn't happen).

### 2.5 Install PyTorch (CUDA 11.8 build)

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expected: 2.1.2+cu118 11.8 True
```

If `torch.cuda.is_available()` prints `False`, stop here and fix your driver/CUDA setup before continuing — nothing downstream will work correctly on CPU-only fallback (gsplat's rasterizer is CUDA-only).

### 2.6 Install gsplat (prebuilt CUDA 11.8 wheel)

```bash
pip install ninja numpy jaxtyping rich
pip install gsplat==1.4.0 --index-url https://docs.gsplat.studio/whl/pt20cu118
```

Note the wheel index path says `pt20cu118` (PyTorch 2.0 + CUDA 11.8) even though you installed PyTorch 2.1.2 above — this is the correct, currently-only-available combination for `gsplat==1.4.0`+CUDA 11.8. See the Troubleshooting section (§5) for what to do if this causes a CUDA kernel error for you.

Verify:

```bash
python -c "import gsplat; from gsplat import csrc; print('gsplat OK:', gsplat.__version__)"
```

### 2.7 Install Nerfstudio and remaining pure-Python dependencies

```bash
# fpsample must be downgraded for compatibility with this dependency set
pip install fpsample==0.2.0
python -m pip install --upgrade setuptools==69.5.1

pip install nerfstudio==1.1.5
```

Verify:

```bash
python -c "import nerfstudio; print('nerfstudio OK:', nerfstudio.__file__)"
```

### 2.8 Clone the repo and install `fly_gsplat` itself

```bash
git clone <YOUR_REPO_URL> <YOUR_PROJECT_ROOT>
cd <YOUR_PROJECT_ROOT>
pip install -e .
```

This installs the remaining pure-Python dependencies pinned in `pyproject.toml` (numpy<2.0, opencv-python, open3d, viser, scikit-image/-learn, scikit-spatial, etc.), **and** registers this repo's custom `splatfacto-checkpoint` Nerfstudio method as a plugin entry point (`[project.entry-points."nerfstudio.method_configs"]` in `pyproject.toml`). Without this step, `ns-train splatfacto-checkpoint` will not be recognized by Nerfstudio's CLI.

Verify the entry point registered correctly:

```bash
ns-train --help | grep splatfacto-checkpoint
```

You should see `splatfacto-checkpoint` listed alongside the built-in `splatfacto` method.

---

## 3. Known Version Mismatch: gsplat wheel vs. PyTorch

The `gsplat==1.4.0` wheel installed above is tagged `+pt20cu118` — i.e. it was compiled against **PyTorch 2.0**, but this project runs it under **PyTorch 2.1.2** (also CUDA 11.8). In this project's actual usage (`torch.cuda.is_available()` returns `True`, GPUs enumerate correctly, training and rasterization work), this mismatch has **not** produced any observed runtime failure. It is flagged here as a fragility point, not a known bug:

- If you hit an unexplained CUDA kernel-level error (segfault, illegal memory access, symbol mismatch) specifically inside `gsplat` operations, revisit this mismatch first.
- If a `gsplat` wheel built against `pt21cu118` becomes available in the future, prefer it.
- Do not "fix" this preemptively by downgrading PyTorch to 2.0.1 — nerfstudio 1.1.5 and other dependencies in this project were validated against 2.1.2, not 2.0.1.

---

## 4. Verification Step

After completing installation, run this end-to-end smoke test. It does **not** require any real data — it builds a synthetic camera + point cloud, and runs a single `splatfacto` forward/backward pass to confirm the whole stack (torch → CUDA → gsplat → nerfstudio) is wired together correctly.

```bash
conda activate fly_gsplat
python - <<'EOF'
import torch
import gsplat
import nerfstudio

print("=== Environment Smoke Test ===")
print("torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available to PyTorch — fix driver/CUDA install first"
device = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(device))

print("gsplat:", gsplat.__version__)
print("nerfstudio path:", nerfstudio.__file__)

# Minimal synthetic rasterization call to confirm gsplat's CUDA kernels actually run.
N = 100
means = torch.randn(N, 3, device=device, requires_grad=True)
scales = torch.rand(N, 3, device=device) * 0.05 + 0.01
quats = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1)
opacities = torch.rand(N, device=device)
colors = torch.rand(N, 3, device=device)

viewmat = torch.eye(4, device=device)[None]  # identity camera at origin looking down -Z
viewmat[0, 2, 3] = 3.0  # push camera back so points are in view
K = torch.tensor([[300.0, 0, 32], [0, 300.0, 32], [0, 0, 1]], device=device)[None]

out, alpha, meta = gsplat.rasterization(
    means, quats, scales, opacities, colors, viewmat, K, width=64, height=64,
)
out.sum().backward()  # confirm autograd works through the CUDA kernel

print("Rendered image shape:", tuple(out.shape))
print("=== ALL CHECKS PASSED ===")
EOF
```

Then confirm the custom Nerfstudio method registered from this repo (run from inside `<YOUR_PROJECT_ROOT>`, after `pip install -e .`):

```bash
ns-train --help | grep splatfacto-checkpoint
```

If both commands succeed, your environment is ready to run the actual pipeline (`generate_dataset.py`, `generate_hull.py`, `ns-train splatfacto ...` — see the main `README.md` for pipeline usage with real data).

### 4.1 Optional: Full Pipeline Test with a Public Dataset (`campanile`)

The synthetic test above only exercises `gsplat`'s CUDA kernels directly. To also confirm Nerfstudio's own data-loading, training loop, and CLI (`ns-train`) work end-to-end — using a public dataset, no project-specific data required — download and train on Nerfstudio's small `campanile` example scene:

```bash
conda activate fly_gsplat
mkdir -p <YOUR_TEST_DIR> && cd <YOUR_TEST_DIR>

ns-download-data nerfstudio --capture-name=campanile
ns-train splatfacto --data data/nerfstudio/campanile --max-num-iterations 100 --viewer.quit-on-train-completion True
```

A successful run will print decreasing loss values and exit cleanly after 100 iterations, writing a checkpoint under `outputs/campanile/splatfacto/<timestamp>/`. This confirms the full `torch → gsplat → nerfstudio → ns-train CLI` chain works, not just the `gsplat` kernels in isolation. This has been run successfully before in this project's history (see `.legacy/env_setup/test_env/outputs/campanile/` for prior run artifacts) — it is not part of the `fly_gsplat` production pipeline itself (which uses real fly camera data, not `campanile`), just a convenient known-good sanity check.

---

## 5. Common Issues / Troubleshooting

**`torch.cuda.is_available()` returns `False`**
- Confirm the OS-level driver sees the GPU: `nvidia-smi` should list your GPU before you even touch conda/pip.
- Confirm you installed the `+cu118` build of torch, not the CPU-only or a mismatched CUDA build (`pip show torch` and check the version string ends in `+cu118`).
- On WSL2 specifically: confirm you're using a CUDA-capable WSL2 driver on the Windows host (not just a Linux-native driver) — see NVIDIA's WSL2 CUDA setup docs.

**`pip install gsplat==1.4.0 --index-url https://docs.gsplat.studio/whl/pt20cu118` fails to find a matching wheel**
- Double check your Python version is 3.10 — the wheel index is keyed by Python ABI as well as torch/CUDA version, and a 3.8/3.9/3.11 interpreter may not find a matching file.
- If no compatible prebuilt wheel exists for your exact platform, you'll need to build `gsplat` from source (`pip install ninja` then `pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0` with `nvcc` from the conda CUDA toolkit on your `PATH`) — this requires the CUDA toolkit's `nvcc` to succeed, which needs `g++`/`ninja-build` from §2.1.

**CUDA kernel errors specifically inside `gsplat` calls (illegal memory access, symbol lookup errors)**
- See §3 — this is the known `pt20cu118` wheel vs `torch==2.1.2` mismatch. Try reverting to whatever combination is confirmed working (torch 2.1.2 + gsplat 1.4.0+pt20cu118, as pinned above) if you changed either version.

**System-wide CUDA (e.g. CUDA 12.x under `/usr/local/cuda`) gets picked up instead of the conda environment's CUDA 11.8**
- This happens if `CUDA_HOME` is exported (e.g. in your `.bashrc`) pointing at `/usr/local/cuda`. Unset it, or make sure it's unset in any shell that will run this project: `unset CUDA_HOME`. The conda environment's own `bin/nvcc` (activated via `conda activate fly_gsplat`) should be the only one on `PATH`; confirm with `which nvcc`.

**`ns-train` doesn't recognize `splatfacto-checkpoint`**
- You skipped or need to re-run `pip install -e .` from the repo root (§2.8) — this is what registers the entry point. Re-activate the conda environment after installing if `ns-train --help` still doesn't show it.

**`fpsample` or `setuptools` version conflicts during `pip install nerfstudio`**
- Install `fpsample==0.2.0` and `setuptools==69.5.1` explicitly *before* `pip install nerfstudio` (as in §2.7) — installing in the other order has caused resolver conflicts in this project's setup history.

**WSL2: training seems slower than expected / intermittent I/O stalls**
- [TODO: author to confirm — this repo's own internal audit (`gpu/ENV.md`) found that storing datasets on a Windows drive mounted via `drvfs` (e.g. `/mnt/c`, `/mnt/x`) is ~5x slower than native WSL2 ext4 storage, and WSL2 memory can be capped by a host-side `.wslconfig` file. Neither is a fly_gsplat-specific issue, but worth checking if you're on WSL2 and see unexplained slowness.]

**Multi-GPU: wrong GPU gets used, or GPUs contend for VRAM**
- Set `CUDA_VISIBLE_DEVICES=<index>` before running training to pin a specific GPU. [TODO: author to confirm whether the scheduler in `gpu/schedule/` has its own GPU-selection mechanism that should be preferred over this env var.]

---

## 6. Notes on Placeholders Used in This Doc

- `<YOUR_REPO_URL>` — this repo's git remote URL (not hardcoded here since it may differ per collaborator/fork).
- `<YOUR_PROJECT_ROOT>` — wherever you clone the repo locally. The original internal docs hardcoded paths like `/home/computer0/fly_project/fly_gsplat` and `/home/abby/FlyProject/gaussian_reconstruction` — these are specific to individual authors' machines and should **not** be assumed; use your own path.
