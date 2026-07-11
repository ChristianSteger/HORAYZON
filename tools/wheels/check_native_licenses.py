"""Fail wheel builds if bundled native dependency notices are missing."""

from pathlib import Path

REQUIRED_LICENSE_FILES = (
    "APACHE-2.0.txt",
    "THIRD_PARTY_NOTICES.txt",
)


def main():
    license_dir = Path(__file__).resolve().parents[2] / "horayzon" / "licenses"
    missing = [
        name
        for name in REQUIRED_LICENSE_FILES
        if not (license_dir / name).is_file()
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(
            f"Missing native dependency license files: {missing_list}"
        )


if __name__ == "__main__":
    main()
