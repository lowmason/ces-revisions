"""Build deterministic Stage 5 operator fixtures and their provenance record."""

import argparse
from pathlib import Path

import numpyro

numpyro.enable_x64()

from ces_revisions.operators import build as operator_build


def build_artifacts(out_dir: Path) -> int:
    result, levels = operator_build.build()
    record = operator_build.write(result, levels, out_dir)
    for name, entry in record["artifacts"].items():
        print(f"{name}: {entry['rows']} rows")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--out-dir", type=Path, default=operator_build.OUTPUT_DIR)
    arguments = parser.parse_args(argv)
    return build_artifacts(arguments.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
