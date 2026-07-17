# AMD gsplat — ROCm 7.2 / gfx1201 / PyTorch 2.13

This fork extends AMD-Ecosystem's ROCm port of gsplat 1.5.3b2 to AMD RDNA 4
hardware using Wave32 execution. It was built and validated on an AMD Radeon
AI PRO R9700 (`gfx1201`) with ROCm 7.2 and PyTorch 2.13.

The upstream AMD release targets Instinct MI300X-class hardware and Wave64.
This fork keeps that history and adds a separately tested `gfx1201` path; it
does not claim validation on MI300X or other architectures.

## Validated platform

| Component | Validated version |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700 |
| GPU architecture | `gfx1201` (RDNA 4, Wave32) |
| ROCm | 7.2 |
| PyTorch | 2.13.0+rocm7.2 |
| Python | 3.12 |
| Operating system | Ubuntu Linux |
| Upstream base | AMD-Ecosystem/gsplat `release/1.5.3b2`, commit `b01acd4` |

Other AMD GPUs and software versions may work but have not been validated by
this project.

## Porting changes

The port contains the following compatibility work:

- replaces the removed private PyTorch symbol
  `c10::hip::HIPCachingAllocator` with the ROCm-compatible
  `c10::cuda::CUDACachingAllocator` API exposed by PyTorch 2.13;
- makes warp reductions, masks, loop bounds and lane handling use Wave32 on
  `gfx1201` instead of assuming Wave64;
- routes the former 64-thread/Wave64-specialized backward rasterizer case
  through the generic kernel on Wave32;
- updates the affected 3DGS and 2DGS projection/backward kernels;
- fixes packed rasterizer background shapes by deriving image dimensions from
  intersection offsets instead of packed `[nnz, 2]` means;
- restores the documented 2DGS color broadcasting and packed gathering for
  both `[N, D]` and `[C, N, D]` input layouts.

## Clone

```bash
git clone --recurse-submodules \
  https://github.com/Painter3000/amd-gsplat-rocm72-gfx1201.git

cd amd-gsplat-rocm72-gfx1201
```

The recursive clone is required because gsplat uses GLM as a submodule.

## Prerequisites

Use a ROCm-enabled PyTorch environment matching the validated platform. Before
building, verify that PyTorch can see the AMD GPU:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Install the lightweight Python/build dependencies without replacing the
existing ROCm PyTorch wheel:

```bash
python -m pip install ninja numpy rich jaxtyping
```

## Build GLM

The AMD build configuration searches for GLM beneath `$HOME/.local/include`.
Build and install the checked-out GLM submodule first:

```bash
cd gsplat/cuda/csrc/third_party/glm

cmake \
  -DGLM_BUILD_TESTS=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local" \
  -B build \
  .

cmake --build build --parallel
cmake --install build

cd ../../../../..
```

## Build and install gsplat

```bash
PYTORCH_ROCM_ARCH=gfx1201 \
MAX_JOBS=4 \
python -m pip install \
  --no-build-isolation \
  --no-deps \
  --no-cache-dir \
  -v .

python -m pip check
```

`--no-build-isolation` lets the build use the already installed ROCm PyTorch
wheel. `--no-deps` prevents pip from silently replacing it. Install any
missing packages explicitly and repeat `python -m pip check` until it reports
no broken requirements.

## Import check

Run import checks from outside the repository root. Otherwise Python may load
the source directory instead of the installed extension.

```bash
cd ..

python - <<'PY'
import torch
import gsplat
from gsplat import csrc

print("gsplat:", gsplat.__file__)
print("Extension:", csrc.__file__)
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0))
print("GSPLAT_IMPORT: PASS")
PY
```

The printed `gsplat` and extension paths should point into the active Python
environment's `site-packages` directory.

## Reproduce the validation

The validation scripts live under `tests/rocm/`. Run them from the parent of
the repository so they exercise the installed wheel:

```bash
python amd-gsplat-rocm72-gfx1201/tests/rocm/gsplat_remaining_paths_validation.py

python amd-gsplat-rocm72-gfx1201/tests/rocm/gsplat_multicamera_packed_validation.py
```

### Validation results

| Area | Result |
| --- | --- |
| Basic covariance/precision forward and backward | PASS |
| 3DGS rasterization forward and backward | PASS |
| Tile sizes 8 and 16 | PASS |
| Packed versus unpacked 3DGS | PASS |
| Render modes `RGB`, `D`, `ED`, `RGB+D`, `RGB+ED` | PASS |
| 2DGS expected/median depth and distortion outputs | PASS |
| Packed versus unpacked 2DGS | PASS |
| Two cameras with partial visibility | PASS |
| Per-Gaussian colors `[N,3]` | PASS |
| Per-view colors `[C,N,3]` | PASS |

The main matrix completed 15/15 forward/backward runs. The targeted
multi-camera matrix completed 8/8 runs with Gaussians visible in both cameras,
only camera 0 and only camera 1. Packed and unpacked full output tensors were
identical; reconstructed radii and visibility masks matched exactly. The
largest observed gradient difference in the multi-camera comparison was about
`3.73e-9`.

These are full-tensor internal-consistency tests on ROCm. They are not a
CUDA-versus-ROCm numerical parity measurement.

## Known limitations

The following paths have not yet been validated on this fork:

- spherical harmonics rendering;
- antialiased rasterization mode;
- sparse gradients;
- camera distortion and rolling shutter;
- distributed or multi-GPU rendering;
- full application training and benchmark parity against CUDA;
- architectures other than `gfx1201`.

Treat support for those paths as experimental until separately tested.

## Upstream and license

This project is based on
[AMD-Ecosystem/gsplat](https://github.com/AMD-Ecosystem/gsplat), which in turn
ports the original [gsplat](https://github.com/nerfstudio-project/gsplat)
project. Retain and follow the repository's existing license and attribution
requirements.
