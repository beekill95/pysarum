from __future__ import annotations

import importlib
import math
import random

import numba as nb
import numpy as np
import numpy.typing as npt
import pyray as pr
from numba import njit
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import ifftshift

from . import config

AgentDtype = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("direction", np.float32),
        ("species", np.uint8),
    ]
)
AgentNbDtype = nb.from_dtype(AgentDtype)
AgentsNbDtpe = nb.types.Array(AgentNbDtype, 1, "C")


@njit("f8(f4[:, :, :], u4, UniTuple(f8, 2), f8, u4, u4)", cache=True)
def _sense(
    pheromone_trails: np.ndarray,
    species_idx: int,
    agent_location: tuple[float, float],
    sensor_direction: float,
    sensor_size: int,
    sensor_distance: int,
) -> float:
    x, y = agent_location
    sensor_pos_x = x + math.cos(sensor_direction) * sensor_distance
    sensor_pos_y = y + math.sin(sensor_direction) * sensor_distance

    _, height, width = pheromone_trails.shape

    xs, ys = (
        sensor_pos_x - sensor_size // 2,
        sensor_pos_y - sensor_size // 2,
    )
    xe, ye = xs + sensor_size, ys + sensor_size

    if (xe <= 0) or (ye <= 0) or (xs >= width) or (ys >= height):
        return -math.inf

    pheromones = pheromone_trails[
        :,
        max(int(ys), 0) : min(int(ye), height),
        max(int(xs), 0) : min(int(xe), width),
    ]
    species_pheromones = pheromones[species_idx, :, :]

    if species_pheromones.size:
        species_pheromone = species_pheromones.sum()
        all_pheromone = pheromones.sum()
        return (species_pheromone / species_pheromones.size) - (
            all_pheromone - species_pheromone
        ) / (pheromones.size - species_pheromones.size)
    else:
        return -math.inf


@njit("i4(f4[:, :, :], u4, UniTuple(f8, 2), f8, f8, u4, u4)", cache=True)
def _steer_direction(
    pheromone_trails: np.ndarray,
    species_idx: int,
    agent_location: tuple[float, float],
    agent_direction: float,
    sensor_angle: float,
    sensor_size: int,
    sensor_distance: int,
) -> int:
    """
    Sense three different directions: front, left and right,
    based on the amount of pheromone level of these sensors,
    turning to the direction with higher level of pheromone,
    or choosing randomly of the levels are equal.

    Returns: 0 to stay on the current direction, -1 to turn to the left,
        +1 to turn to the right.
    """
    left_sensor = _sense(
        pheromone_trails,
        species_idx,
        agent_location,
        agent_direction + sensor_angle,
        sensor_size,
        sensor_distance,
    )
    front_sensor = _sense(
        pheromone_trails,
        species_idx,
        agent_location,
        agent_direction,
        sensor_size,
        sensor_distance,
    )
    right_sensor = _sense(
        pheromone_trails,
        species_idx,
        agent_location,
        agent_direction - sensor_angle,
        sensor_size,
        sensor_distance,
    )

    if front_sensor < min(left_sensor, right_sensor):
        # Rotate randomly.
        return np.random.choice(np.array([-1, 1]))
    elif front_sensor < left_sensor:
        return 1
    elif front_sensor < right_sensor:
        return -1

    return 0


@njit(AgentsNbDtpe(nb.u4, nb.u4, nb.u4, nb.u4), cache=True)
def generate_random_agents(
    *, nb_agents: int, world_width: int, world_height: int, nb_species: int
) -> npt.NDArray[AgentDtype]:
    agents = np.zeros(nb_agents, dtype=AgentDtype)

    for i in range(nb_agents):
        agents[i].x = world_width / 2.0
        agents[i].y = world_height / 2.0
        agents[i].direction = random.random() * 2 * math.pi
        agents[i].species = random.randint(0, nb_species - 1)

    return agents


@njit("f4[:, :, :](f4[:, :, :], f4, f4)", cache=True)
def evaporate_pheromone(
    pheromone_trails: np.NDArray[np.float32], dt: float, evaporation_speed: float
) -> np.NDArray[np.float32]:
    # Evaporate pheromone trails of different species at a fixed rate.
    # Support both 2d and 3d pheromone trail array.
    return np.maximum(pheromone_trails - dt * evaporation_speed, np.float32(0))


@njit("f4[:, :, :](f4[:, :, :], f4[:, :])", cache=True)
def diffuse_pheromone(
    pheromone_trails: np.NDArray[np.float32], padded_kernel: np.NDArray[np.float32]
) -> np.NDArray[np.float32]:
    # Basically applying convolution 2d on the pheromone trail.
    # Implementation copied from https://stackoverflow.com/a/54977551
    # The pheromone_trails must be of shape <nb_species, height, width>
    assert pheromone_trails.ndim == 3
    kernel = ifftshift(padded_kernel)
    return np.real(ifft2(fft2(pheromone_trails) * fft2(kernel)[None, ...]))


@njit(cache=True)
def _simulate(
    agents: npt.NDArray[AgentDtype],
    pheromone_trails: npt.NDArray[np.float32],
    dt: float,
    *,
    padded_diffusion_kernel: npt.NDArray[np.float32],
    species_list: tuple[config.Species, ...],
    evaporation_speed: float,
) -> npt.NDArray[np.float32]:
    "Perform simulation step forward dt."
    _, width, height = pheromone_trails.shape

    # Move all the agents at once.
    for agent in agents:
        # Steer the agent.
        species = species_list[agent.species]
        direction = _steer_direction(
            pheromone_trails,
            agent.species,
            (agent.x, agent.y),
            agent.direction,
            species.sensor_angle,
            species.sensor_size,
            species.sensor_distance,
        )
        new_direction = agent.direction + direction * species.turn_angle

        # Calculate the agent new position.
        new_x = agent.x + math.cos(new_direction) * species.speed * dt
        new_y = agent.y + math.sin(new_direction) * species.speed * dt

        if new_x >= width or new_x <= 0:
            new_x = 0 if new_x < 0 else width

            # Choose random angle instead of just moving in the opposite direction.
            angle = random.random() * math.pi
            new_direction = (
                (angle - math.pi / 2) if new_x <= 0 else (angle + math.pi / 2)
            )

        if new_y >= height or new_y <= 0:
            new_y = 0 if new_y < 0 else height

            # Choose random angle instead of just moving in the opposite direction.
            angle = random.random() * math.pi
            new_direction = angle if new_y <= 0 else (angle + math.pi)

        # Mark the trail with pheromone.
        # TODO: is there a way to do this efficiently?
        nb_steps = int(math.ceil(max(abs(new_x - agent.x), abs(new_y - agent.y))))
        if nb_steps:
            delta_x = (new_x - agent.x) / nb_steps
            delta_y = (new_y - agent.y) / nb_steps
            for i in range(nb_steps):
                x = int(agent.x + i * delta_x)
                y = int(agent.y + i * delta_y)

                # TODO: how to remove this check?
                if (0 <= x < width) and (0 <= y < height):
                    pheromone_trails[agent.species, y, x] = 1.0

        # Update the agent's location and direction.
        agent.x = new_x
        agent.y = new_y
        agent.direction = new_direction

    # Diffuse and evaporate the pheromones.
    pheromone_trails = evaporate_pheromone(pheromone_trails, dt, evaporation_speed)
    pheromone_trails = diffuse_pheromone(pheromone_trails, padded_diffusion_kernel)
    return pheromone_trails


def create_pheromone_trail_color(
    width: int, height: int, species_color: tuple[int, int, int]
) -> np.ndarray:
    trail = np.zeros((height, width, 4), dtype=np.uint8)

    trail[:, :, 0] = species_color[0]
    trail[:, :, 1] = species_color[1]
    trail[:, :, 2] = species_color[2]

    return trail


def run(nb_agents: int, seed: int = 0):
    sim_width = 800
    sim_height = 800

    random.seed(seed)

    pr.init_window(sim_width, sim_height, "Pysarum")

    agents = generate_random_agents(
        nb_agents=nb_agents,
        world_width=sim_width,
        world_height=sim_height,
        nb_species=len(config.SPECIES),
    )
    nb_species = len(config.SPECIES)
    pheromone_trails = np.zeros((nb_species, sim_height, sim_width), dtype=np.float32)
    pheromone_textures: list[pr.Texture | None] = [None] * nb_species

    species_trail_colors = [
        create_pheromone_trail_color(sim_width, sim_height, species.color)
        for species in config.SPECIES
    ]

    # Pad the diffusion kernel to match the pheromone_trails shape.
    diffusion_kernel = config.DIFFUSION_KERNEL
    sz = (
        pheromone_trails.shape[-1] - diffusion_kernel.shape[0],
        pheromone_trails.shape[-2] - diffusion_kernel.shape[1],
    )  # total amount of padding
    diffusion_kernel = np.pad(
        diffusion_kernel,
        (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)),
        "constant",
    )

    while not pr.window_should_close():
        # Reload config.
        importlib.reload(config)

        # Update agents position, and also mark the pheromone.
        dt = pr.get_frame_time()
        pheromone_trails = _simulate(
            agents,
            pheromone_trails,
            np.float32(dt),
            padded_diffusion_kernel=diffusion_kernel,
            species_list=config.SPECIES,
            evaporation_speed=np.float32(config.EVAPORATION_SPEED),
        )

        # Convert the pheromone trail into image.
        for i in range(nb_species):
            trail_colors = species_trail_colors[i]
            trail_colors[:, :, 3] = (pheromone_trails[i] * 255).astype(np.uint8)
            pheromone_image = pr.Image(
                trail_colors.flatten(),
                sim_width,
                sim_height,
                1,
                pr.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
            )
            if pheromone_textures[i] is None:
                pheromone_textures[i] = pr.load_texture_from_image(pheromone_image)
            else:
                pr.update_texture(pheromone_textures[i], pheromone_image.data)

        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        # pr.draw_text("Hello, Physarum Polycephalum", 190, 200, 20, pr.VIOLET)

        # Draw the first texture.
        pr.draw_texture(pheromone_textures[0], 0, 0, pr.WHITE)

        # Blend multiple textures into one image.
        pr.begin_blend_mode(pr.BlendMode.BLEND_ADDITIVE)
        for i in range(1, nb_species):
            pr.draw_texture(pheromone_textures[i], 0, 0, pr.WHITE)
        pr.end_blend_mode()

        pr.end_drawing()

    # Unload all textures.
    for i in range(nb_species):
        texture = pheromone_textures[i]
        if texture is not None:
            pr.unload_texture(texture)

    pr.close_window()
