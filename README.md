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

# Build + run app (debug by default; bootstraps LVK deps automatically)
.\scripts\run_app.bat

# Build + run editor
.\scripts\run_editor.bat
```

Release build:

```powershell
.\scripts\build_app.bat release
.\scripts\build_editor.bat release
.\scripts\build_lib.bat release
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
./scripts/build_app.sh
./scripts/build_editor.sh
./scripts/build_lib.sh
./scripts/run_app.sh
./scripts/run_editor.sh
```

Tests:

```bash
./scripts/run_tests.sh
```

`run_tests` enables the manifest `tests` feature automatically. If you consume `nuri` as a vcpkg port instead of using this repo in manifest mode, install `nuri[tests]` before running the test targets.

## Notes

- LVK’s bootstrap downloads and builds third-party deps into `external/lightweightvk/third-party/deps` (first run can take a while).
- The warning about Python packages `paramiko`/`scp` can be ignored; they’re only needed for optional bootstrap features.
- Target-specific scripts configure a minimal build tree per mode and target set. Bash scripts use `build/<mode>/<target>/`; Windows batch scripts use `build/debug/<target>/` for Debug and `build_release/<target>/` for Release.
- `build_debug`/`build_release` remain as compatibility wrappers and default to the `app` profile.

## Project layout (high level)

- `app/` sample application using engine APIs
- `lib/nuri/platform/` platform-facing abstractions (Window, GPUDevice)
- `lib/nuri/gfx/` renderer and pipeline/shader front-end (LVK-free)
- `lib/nuri/resources/` CPU/GPU asset types
- `external/lightweightvk/` LVK submodule
