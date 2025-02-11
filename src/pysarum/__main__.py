from __future__ import annotations

from .simulation import run
from .config import NB_AGENTS


def main() -> int:
    run(NB_AGENTS, 5)
    return 0


exit(main())
