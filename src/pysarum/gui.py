from __future__ import annotations

import pyray as pr


class GUI:
    def __init__(self) -> None:
        self._nb_agents_ptr = pr.ffi.new("int *", 5000)
        self._agent_speed_ptr = pr.ffi.new("int *", 50)
        self._agent_sensor_distance_ptr = pr.ffi.new("int *", 3)
        self._agent_sensor_size_ptr = pr.ffi.new("int *", 3)
        self._agent_sensor_angle_ptr = pr.ffi.new("int *", 45)
        self._agent_turn_angle_ptr = pr.ffi.new("int *", 60)

        self._nb_agents_edit_mode = False
        self._agent_speed_edit_mode = False

    @property
    def width(self):
        return 200

    @property
    def height(self):
        return 800

    def __call__(self, pos_x: int, pos_y: int):
        self.render(pos_x, pos_y)

    def render(self, pos_x: int, pos_y: int):
        # FIXME: this doesn't render based on the given position.
        input_height = 16

        pr.gui_window_box((pos_x, pos_y, self.width, self.height), "Pysarum Config")

        offset_x = pos_x + 10
        offset_y = pos_y + 32
        pr.gui_group_box((offset_x, offset_y, 182, 5 * input_height), "Agents")

        offset_y += input_height
        if pr.gui_value_box(
            (offset_x + 46, offset_y, 132, input_height),
            "Quantity",
            self._nb_agents_ptr,
            0,
            10000,
            self._nb_agents_edit_mode,
        ):
            self._nb_agents_edit_mode = not self._nb_agents_edit_mode

        offset_y += 2 * input_height
        if pr.gui_value_box(
            (offset_x + 46, offset_y, 132, input_height),
            "Speed",
            self._agent_speed_ptr,
            0,
            1000,
            self._agent_speed_edit_mode,
        ):
            self._agent_speed_edit_mode = not self._agent_speed_edit_mode

        offset_y += 2 * input_height
        if pr.gui_value_box(
            (offset_x + 46, offset_y, 132, input_height), "Sensor Distance"
        ):
            pass
