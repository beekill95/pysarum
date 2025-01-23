from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import pyray as pr

AGENT_SPEED = 50.0
EVAPORATION_SPEED = 0.1
DIFFUSION_RADIUS = 5
DIFFUSION_KERNEL = np.ones((DIFFUSION_RADIUS, DIFFUSION_RADIUS), dtype=np.float32) / (
    DIFFUSION_RADIUS * DIFFUSION_RADIUS
)


@dataclass(slots=True)
class Agent:
    x: float
    y: float
    direction: float

    @property
    def int_x(self):
        return int(self.x)

    @property
    def int_y(self):
        return int(self.y)

    def velocity_x(self):
        return math.cos(self.direction) * AGENT_SPEED

    def velocity_y(self):
        return math.sin(self.direction) * AGENT_SPEED

    def move(self, dt: float, pheromone_trail: np.ndarray, width: int, height: int):
        new_x = self.x + self.velocity_x() * dt
        new_y = self.y + self.velocity_y() * dt

        if new_x >= width or new_x <= 0:
            new_x = 0 if new_x < 0 else width
            self.direction = math.pi - self.direction

            # self.velocity_x = -self.velocity_x
            # Choose random angle instead of just moving in the opposite direction.
            # angle = random.random() * math.pi

        if new_y >= height or new_y <= 0:
            new_y = 0 if new_y < 0 else height
            self.direction = -self.direction

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


def create_random_agent(world_width: int, world_height: int):
    angle = random.random() * 2 * math.pi

    return Agent(
        # random.randint(0, world_width),
        # random.randint(0, world_height),
        world_width / 2,
        world_height / 2,
        angle,
    )


def evaporate_pheromone(pheromone_trail: np.ndarray, dt: float):
    return np.maximum(pheromone_trail - dt * EVAPORATION_SPEED, 0.0)


def diffuse_pheromone(pheromone_trail: np.ndarray):
    # Basically applying convolution 2d on the pheromone trail.
    # Implementation copied from https://stackoverflow.com/a/54977551
    from numpy.fft import fft2, ifft2, ifftshift

    img = pheromone_trail
    kernel = DIFFUSION_KERNEL

    sz = (
        img.shape[0] - kernel.shape[0],
        img.shape[1] - kernel.shape[1],
    )  # total amount of padding
    kernel = np.pad(
        kernel,
        (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)),
        "constant",
    )
    kernel = ifftshift(kernel)

    return np.real(ifft2(fft2(img) * fft2(kernel)))


def run(width: int, height: int, nb_agents: int, seed: int = 0):
    random.seed(seed)
    pr.init_window(width, height, "Pysarum")

    agents = [create_random_agent(width, height) for _ in range(nb_agents)]
    pheromone_trail = np.zeros((height, width), dtype=np.float32)
    pheromone_texture = None

    while not pr.window_should_close():
        # Update agents position, and also mark the pheromone.
        dt = pr.get_frame_time()
        for agent in agents:
            agent.move(dt, pheromone_trail, width, height)

        pheromone_trail = evaporate_pheromone(pheromone_trail, dt)
        pheromone_trail = diffuse_pheromone(pheromone_trail)

        # Convert the pheromone trail into image.
        pheromone_image = pr.Image(
            (pheromone_trail.flatten() * 255).astype(np.uint8),
            width,
            height,
            1,
            pr.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE,
        )
        if pheromone_texture is None:
            pheromone_texture = pr.load_texture_from_image(pheromone_image)
        else:
            pr.update_texture(pheromone_texture, pheromone_image.data)

        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        # pr.draw_text("Hello, Physarum Polycephalum", 190, 200, 20, pr.VIOLET)
        pr.draw_texture(pheromone_texture, 0, 0, pr.WHITE)

        for agent in agents:
            pr.draw_pixel(agent.int_x, agent.int_y, pr.RED)

        pr.end_drawing()

    if pheromone_texture is not None:
        pr.unload_texture(pheromone_texture)

    pr.close_window()
