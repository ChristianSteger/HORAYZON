"""Print the first wheel in a directory compatible with this interpreter."""

import sys
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: select_compatible_wheel.py WHEELHOUSE")

    wheelhouse = Path(sys.argv[1])
    supported_tags = set(sys_tags())
    for wheel in sorted(wheelhouse.glob("*.whl")):
        _, _, _, wheel_tags = parse_wheel_filename(wheel.name)
        if supported_tags.intersection(wheel_tags):
            print(wheel)
            return

    raise SystemExit(f"No compatible wheel found in {wheelhouse}")


if __name__ == "__main__":
    main()
