from __future__ import annotations

from .simulation import run


def main() -> int:
    run(5000, 5)
    return 0


exit(main())
