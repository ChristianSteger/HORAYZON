# Release Checklist

HORAYZON should be published to PyPI by the upstream maintainer from
`ChristianSteger/HORAYZON` after the packaging pull requests have been merged.
Fork workflows may build and test release artifacts, but must not publish them.

## Before the first PyPI release

1. Merge the prerequisite pull requests in order:
   - testing,
   - linting and formatting,
   - code fixes,
   - dependency-bundled wheel packaging.
2. Decide the public release version and update the package metadata before
   tagging. Do not publish the current placeholder version without checking it
   against the latest upstream release.
3. Confirm the release workflow builds all intended artifacts:
   - Linux x86_64 wheels for CPython 3.10-3.14,
   - Linux aarch64 wheels for CPython 3.10-3.14,
   - macOS x86_64 wheels for CPython 3.10-3.14,
   - macOS arm64 wheels for CPython 3.10-3.14,
   - one source distribution.
4. Confirm the wheel artifacts install and run the native smoke tests without a
   user-installed Embree or TBB runtime.
5. Review the bundled Embree and oneTBB notices in `horayzon/licenses/`.

## Configure PyPI Trusted Publishing

Create a pending PyPI Trusted Publisher for the first release:

- PyPI project name: `horayzon`
- Owner: `ChristianSteger`
- Repository: `HORAYZON`
- Workflow name: `.github/workflows/release.yml`
- Environment: `pypi`

Pending publishers do not reserve the PyPI project name until the first
successful publish. Configure the pending publisher and perform the first
release in the same release window.

No PyPI API token is required. The workflow uses GitHub's OIDC token through
`pypa/gh-action-pypi-publish`.

## Publish from upstream

1. Create a GitHub environment named `pypi` in `ChristianSteger/HORAYZON`.
   Add reviewer protection if desired.
2. Create and publish a GitHub release for the chosen tag.
3. Wait for the `Release distributions` workflow to complete.
4. Verify installation in a clean environment:

   ```bash
   python -m venv /tmp/horayzon-release-check
   /tmp/horayzon-release-check/bin/python -m pip install --upgrade pip
   /tmp/horayzon-release-check/bin/python -m pip install horayzon
   /tmp/horayzon-release-check/bin/python -c "import horayzon; print(horayzon.__file__)"
   ```

5. If publishing fails, do not reuse the same version. Fix the issue, bump the
   version, and publish a new release.
