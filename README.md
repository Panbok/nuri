# nuri

This project uses [LightweightVK](https://github.com/corporateshark/lightweightvk) (LVK) + GLFW on Vulkan. LVK is included as a git submodule at `external/lightweightvk` and is always built as part of the project. GLFW and LVK are hidden behind the platform layer (`lib/nuri/platform`) so app code and public headers stay LVK/GLFW-free.

## Prerequisites

- Git
- Python 3 (`python3` or `py -3`)
- Vulkan SDK (so CMake can find `vulkan-1.lib` / `libvulkan.so`)
- CMake 3.27+
- Ninja
- Clang (recommended; the provided scripts use `clang`/`clang++`)
- vcpkg (used for `glm` and `assimp`)

## Windows (PowerShell)

```powershell
# Clone + submodule
git clone --recurse-submodules <your-repo-url> nuri
cd nuri
git submodule update --init --recursive   # safe to re-run

# Point to your vcpkg installation
$env:VCPKG_ROOT = "E:\install\vcpkg"   # adjust

# Build + run (bootstraps LVK deps automatically)
.\scripts\run_debug.bat
```

Release build:

```powershell
.\scripts\build_release.bat
```

Tests:

```powershell
.\scripts\run_tests.bat
```

`run_tests` enables the manifest `tests` feature automatically. If you consume `nuri` as a vcpkg port instead of using this repo in manifest mode, install `nuri[tests]` before running the test targets.

## Linux/macOS (bash)

```bash
git clone --recurse-submodules <your-repo-url> nuri
cd nuri
git submodule update --init --recursive   # safe to re-run

export VCPKG_ROOT="$HOME/vcpkg"   # adjust
./scripts/build_debug.sh
```

Tests:

```bash
./scripts/run_tests.sh
```

`run_tests` enables the manifest `tests` feature automatically. If you consume `nuri` as a vcpkg port instead of using this repo in manifest mode, install `nuri[tests]` before running the test targets.

## Notes

- LVK’s bootstrap downloads and builds third-party deps into `external/lightweightvk/third-party/deps` (first run can take a while).
- The warning about Python packages `paramiko`/`scp` can be ignored; they’re only needed for optional bootstrap features.
- Regular `build_debug`/`build_release` runs leave `NURI_BUILD_TESTS=OFF`; enable tests explicitly when you need them.

## Project layout (high level)

- `app/` sample application using engine APIs
- `lib/nuri/platform/` platform-facing abstractions (Window, GPUDevice)
- `lib/nuri/gfx/` renderer and pipeline/shader front-end (LVK-free)
- `lib/nuri/resources/` CPU/GPU asset types
- `external/lightweightvk/` LVK submodule
