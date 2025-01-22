from __future__ import annotations

import pyray as pr


def main() -> int:
    pr.init_window(800, 450, "Pysarum")

    while not pr.window_should_close():
        pr.begin_drawing()
        pr.clear_background(pr.WHITE)
        pr.draw_text("Hello, Physarum Polycephalum", 190, 200, 20, pr.VIOLET)
        pr.end_drawing()

    pr.close_window()

    return 0


exit(main())
