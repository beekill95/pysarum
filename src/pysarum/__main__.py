from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import NB_AGENTS


def parse_args(argv: Sequence[str] | None):
    parser = argparse.ArgumentParser()

    parser.add_argument("backend", choices=["cpu", "glsl"])

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    match args.backend:
        case "cpu":
            from .simulation import run

            run(NB_AGENTS, seed=5, sim_width=800, sim_height=800)
        case "glsl":
            from .main_glsl import run

            run()
        case _:
            raise ValueError(f"Unknown or unsupported backend: `{args.backend}`")

    return 0


exit(main())
