from __future__ import annotations

from .config import NB_AGENTS
from .simulation import run


def main() -> int:
    run(NB_AGENTS, seed=5, sim_width=1000, sim_height=1000)
    return 0


exit(main())
