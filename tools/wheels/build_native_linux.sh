#!/usr/bin/env bash
set -euxo pipefail

PREFIX="${HORAYZON_NATIVE_DIR:-/opt/horayzon-native}"
WORKDIR="${HORAYZON_NATIVE_BUILD_DIR:-/tmp/horayzon-native-build}"
TBB_VERSION="${HORAYZON_TBB_VERSION:-v2022.2.0}"
EMBREE_VERSION="${HORAYZON_EMBREE_VERSION:-v4.4.1}"

if [[ -f "${PREFIX}/include/embree4/rtcore.h" \
    && -f "${PREFIX}/include/tbb/parallel_for.h" ]]; then
    exit 0
fi

if command -v yum >/dev/null 2>&1; then
    yum install -y cmake git ninja-build
elif command -v dnf >/dev/null 2>&1; then
    dnf install -y cmake git ninja-build
elif command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y cmake git ninja-build
fi

rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}" "${PREFIX}"

git clone --depth 1 --branch "${TBB_VERSION}" \
    https://github.com/uxlfoundation/oneTBB.git "${WORKDIR}/oneTBB"
cmake -S "${WORKDIR}/oneTBB" -B "${WORKDIR}/oneTBB-build" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DTBB_TEST=OFF
cmake --build "${WORKDIR}/oneTBB-build" --target install

git clone --depth 1 --branch "${EMBREE_VERSION}" \
    https://github.com/RenderKit/embree.git "${WORKDIR}/embree"
cmake -S "${WORKDIR}/embree" -B "${WORKDIR}/embree-build" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DEMBREE_BACKFACE_CULLING=OFF \
    -DEMBREE_FILTER_FUNCTION=OFF \
    -DEMBREE_GEOMETRY_CURVE=OFF \
    -DEMBREE_GEOMETRY_POINT=OFF \
    -DEMBREE_GEOMETRY_SUBDIVISION=OFF \
    -DEMBREE_GEOMETRY_USER=OFF \
    -DEMBREE_ISPC_SUPPORT=OFF \
    -DEMBREE_TASKING_SYSTEM=TBB \
    -DEMBREE_TUTORIALS=OFF
cmake --build "${WORKDIR}/embree-build" --target install
