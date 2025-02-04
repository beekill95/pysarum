from __future__ import annotations

import importlib
import math
import random
from dataclasses import dataclass

import numpy as np
import pyray as pr

from . import config


@dataclass(slots=True)
class Agent:
    x: float
    y: float
    direction: float
    species_: int

    @property
    def int_x(self):
        return int(self.x)

    @property
    def int_y(self):
        return int(self.y)

    @property
    def species(self):
        return config.SPECIES[self.species_]

    def velocity_x(self):
        return math.cos(self.direction) * self.species.speed

    def velocity_y(self):
        return math.sin(self.direction) * self.species.speed

    def move(self, dt: float, pheromone_trails: np.ndarray, width: int, height: int):
        # pheromone_trails must be of size <nb_species, height, width>
        assert pheromone_trails.ndim == 3

        # Move the agent and drop pheromone.
        pheromone_trail = pheromone_trails[self.species_]
        self._steer_direction(pheromone_trail)

        new_x = self.x + self.velocity_x() * dt
        new_y = self.y + self.velocity_y() * dt

        if new_x >= width or new_x <= 0:
            new_x = 0 if new_x < 0 else width

            # Choose random angle instead of just moving in the opposite direction.
            angle = random.random() * math.pi
            self.direction = (
                (angle - math.pi / 2) if new_x <= 0 else (angle + math.pi / 2)
            )

        if new_y >= height or new_y <= 0:
            new_y = 0 if new_y < 0 else height

            # Choose random angle instead of just moving in the opposite direction.
            angle = random.random() * math.pi
            self.direction = angle if new_y <= 0 else (angle + math.pi)

        # Mark the trail with pheromone.
        # TODO: is there a way to do this efficiently?
        nb_steps = int(math.ceil(max(abs(new_x - self.x), abs(new_y - self.y))))
        if nb_steps:
            delta_x = (new_x - self.x) / nb_steps
            delta_y = (new_y - self.y) / nb_steps
            for i in range(nb_steps):
                x = int(self.x + i * delta_x)
                y = int(self.y + i * delta_y)

                # TODO: how to remove this check?
                if (0 <= x < width) and (0 <= y < height):
                    pheromone_trail[y, x] = 1.0

        # Update the self's position.
        self.x = new_x
        self.y = new_y

    def _steer_direction(self, pheromone_trail: np.ndarray):
        species = self.species
        left_sensor = self._sense(
            pheromone_trail, self.direction + species.sensor_angle
        )
        front_sensor = self._sense(pheromone_trail, self.direction)
        right_sensor = self._sense(
            pheromone_trail, self.direction - species.sensor_angle
        )

        if front_sensor < min(left_sensor, right_sensor):
            # Rotate randomly.
            self.direction = random.choice(
                [
                    self.direction + species.turn_angle,
                    self.direction - species.turn_angle,
                ]
            )
        elif front_sensor < left_sensor:
            self.direction += species.turn_angle
        elif front_sensor < right_sensor:
            self.direction -= species.turn_angle

        # Otherwise, keep the current direction.

    def _sense(self, pheromone_trail: np.ndarray, sensor_direction: float):
        species = self.species
        sensor_pos_x = self.x + math.cos(sensor_direction) * species.sensor_distance
        sensor_pos_y = self.y + math.sin(sensor_direction) * species.sensor_distance

        height, width = pheromone_trail.shape

        xs, ys = (
            sensor_pos_x - species.sensor_size // 2,
            sensor_pos_y - species.sensor_size // 2,
        )
        xe, ye = xs + species.sensor_size, ys + species.sensor_size

        if (xe <= 0) or (ye <= 0) or (xs >= width) or (ys >= height):
            return -math.inf

        # FIXME: this still result into an empty slices.
        pheromones = (
            pheromone_trail[
                max(int(ys), 0) : min(int(ye), height),
                max(int(xs), 0) : min(int(xe), width),
            ]
            .flatten()
            .tolist()
        )

        return sum(pheromones) / len(pheromones) if len(pheromones) else -math.inf


def create_random_agent(world_width: int, world_height: int):
    angle = random.random() * 2 * math.pi
    species = random.randint(1, len(config.SPECIES)) - 1

    return Agent(
        # random.randint(0, world_width),
        # random.randint(0, world_height),
        world_width / 2,
        world_height / 2,
        angle,
        species,
    )


def evaporate_pheromone(pheromone_trails: np.ndarray, dt: float):
    # Evaporate pheromone trails of different species at a fixed rate.
    # Support both 2d and 3d pheromone trail array.
    return np.maximum(pheromone_trails - dt * config.EVAPORATION_SPEED, 0.0)


def diffuse_pheromone(pheromone_trails: np.ndarray):
    # Basically applying convolution 2d on the pheromone trail.
    # Implementation copied from https://stackoverflow.com/a/54977551
    # The pheromone_trails must be of shape <nb_species, height, width>
    from numpy.fft import fft2, ifft2, ifftshift

    assert pheromone_trails.ndim == 3
    img = pheromone_trails
    kernel = config.DIFFUSION_KERNEL

    sz = (
        img.shape[-1] - kernel.shape[0],
        img.shape[-2] - kernel.shape[1],
    )  # total amount of padding
    kernel = np.pad(
        kernel,
        (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)),
        "constant",
    )
    kernel = ifftshift(kernel)

    return np.real(ifft2(fft2(img) * fft2(kernel)[None, ...]))


def run(nb_agents: int, seed: int = 0):
    sim_width = 800
    sim_height = 800

    random.seed(seed)

    pr.init_window(sim_width, sim_height, "Pysarum")

    agents = [create_random_agent(sim_width, sim_height) for _ in range(nb_agents)]
    pheromone_trails = np.zeros(
        (len(config.SPECIES), sim_height, sim_width), dtype=np.float32
    )
    pheromone_textures = None

    while not pr.window_should_close():
        # Reload config.
        importlib.reload(config)

        # Update agents position, and also mark the pheromone.
        dt = pr.get_frame_time()
        for agent in agents:
            agent.move(dt, pheromone_trails, sim_width, sim_height)

        pheromone_trails = evaporate_pheromone(pheromone_trails, dt)
        pheromone_trails = diffuse_pheromone(pheromone_trails)

        # Convert the pheromone trail into image.
        pheromone_images = pr.Image(
            (pheromone_trail.flatten() * 255).astype(np.uint8),
            sim_width,
            sim_height,
            1,
            pr.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE,
        )
        if pheromone_textures is None:
            pheromone_texture = pr.load_texture_from_image(pheromone_image)
        else:
            pr.update_texture(pheromone_texture, pheromone_image.data)

        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        # pr.draw_text("Hello, Physarum Polycephalum", 190, 200, 20, pr.VIOLET)
        pr.draw_texture(pheromone_texture, 0, 0, pr.WHITE)

        pr.end_drawing()

    if pheromone_texture is not None:
        pr.unload_texture(pheromone_texture)

    pr.close_window()
