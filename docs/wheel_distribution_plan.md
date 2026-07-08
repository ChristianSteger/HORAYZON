# Wheel Distribution Plan

## Goal

Make HORAYZON installable with `pip install horayzon` from binary wheels that
do not require users to install Embree or TBB locally.

## First Target

Build and test wheels on the fork before enabling publishing:

- Linux x86_64, CPython 3.10-3.13
- macOS x86_64, CPython 3.10-3.13
- macOS arm64, CPython 3.10-3.13

Windows x86_64 is feasible, but should follow after the Unix-like wheel path is
stable because the current build has not been exercised on Windows.

## Current Implementation

- `setup.py` discovers Embree and TBB from `HORAYZON_EMBREE_DIR`,
  `HORAYZON_TBB_DIR`, `HORAYZON_NATIVE_DIR`, `CONDA_PREFIX`, or common system
  prefixes.
- Runtime Python dependencies are declared in package metadata.
- `cibuildwheel` configuration builds and tests wheels without publishing.
- GitHub Actions stores wheel artifacts for review and manual installation
  tests.
- Linux wheel builds compile oneTBB and Embree in the manylinux container so
  `auditwheel` can bundle the shared libraries.
- macOS wheel builds use Homebrew Embree/TBB and rely on `delocate` through
  `cibuildwheel` to bundle the dynamic libraries.
- macOS deployment targets are pinned per runner to match Homebrew's bundled
  native libraries: 13.0 for x86_64 and 14.0 for arm64.
- Wheel artifacts include third-party redistribution notices for the bundled
  Embree and oneTBB runtime libraries.
- CI inspects repaired wheel dependencies and smoke-tests the installed wheel
  from a temporary directory.
- `cibuildwheel` runs the native ray-tracing smoke tests against every repaired
  wheel. The full pytest suite remains a local/pre-PR check because it touches
  optional helpers such as ocean masking.
- The release workflow builds artifacts for fork validation but only publishes
  from `ChristianSteger/HORAYZON` on GitHub release events after PyPI Trusted
  Publishing has been configured by the upstream maintainer.

## Before Publishing

- Confirm wheel artifacts install and pass smoke tests on clean Linux and macOS
  machines.
- Inspect wheel contents and linked libraries with `auditwheel show`,
  `delocate-listdeps`, and `otool -L`.
- Add Windows x86_64 once the build script and DLL repair path are verified.
- Follow `docs/release_checklist.md` to configure PyPI Trusted Publishing and
  publish from upstream.
