from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class Species(NamedTuple):
    speed: float
    turn_angle: float

    sensor_size: int
    sensor_distance: int
    sensor_angle: float


# Pheromone characteristics.
EVAPORATION_SPEED = 0.2
DIFFUSION_RADIUS = 5
DIFFUSION_KERNEL = np.ones((DIFFUSION_RADIUS, DIFFUSION_RADIUS), dtype=np.float32) / (
    DIFFUSION_RADIUS * DIFFUSION_RADIUS
)

# Agents' default characteristics.
AGENT_SPEED = 90.0
SENSOR_SIZE = 5
SENSOR_DISTANCE = 9
SENSOR_ANGLE = math.pi / 3
DIRECTION_ANGLE = math.pi / 3

# Different species.
SPECIES = (
    Species(
        speed=AGENT_SPEED,
        turn_angle=DIRECTION_ANGLE,
        sensor_size=SENSOR_SIZE,
        sensor_distance=SENSOR_DISTANCE,
        sensor_angle=SENSOR_ANGLE,
    ),
    Species(
        speed=AGENT_SPEED,
        turn_angle=DIRECTION_ANGLE,
        sensor_size=SENSOR_SIZE,
        sensor_distance=SENSOR_DISTANCE,
        sensor_angle=SENSOR_ANGLE,
    ),
)
