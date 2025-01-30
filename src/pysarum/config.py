from __future__ import annotations

import math

import numpy as np

AGENT_SPEED = 90.0
EVAPORATION_SPEED = 0.2
DIFFUSION_RADIUS = 5
DIFFUSION_KERNEL = np.ones((DIFFUSION_RADIUS, DIFFUSION_RADIUS), dtype=np.float32) / (
    DIFFUSION_RADIUS * DIFFUSION_RADIUS
)
SENSOR_SIZE = 5
SENSOR_DISTANCE = 9
SENSOR_ANGLE = math.pi / 3
DIRECTION_ANGLE = math.pi / 3
