from __future__ import annotations

import math
import os
import random

import moderngl
import numpy as np
import numpy.typing as npt
import pyray as pr


from . import config

SHADERS_ROOT = os.path.join(os.path.dirname(__file__), "shaders")

AgentDtype = np.dtype(
    [
        ("location", "f4", (2,)),
        ("direction", "f4"),
        ("species", "u4"),
    ],
    align=True,
)
SpeciesDtype = np.dtype(
    [
        ("speed", "f4"),
        ("turn_angle", "f4"),
        ("sensor_size", "u4"),
        ("sensor_distance", "u4"),
        ("sensor_angle", "u4"),
        ("color", "f4", (4,)),
    ],
    align=True,
)


def load_shader_source(shader_name: str) -> str:
    """
    This helper function will load the shader source code from the given path.
    Additionally, it will replace the `#include` directive with the source of the file in the given path.
    Note, this doesn't recursively replace `#include` directive in the included file.

    Parameters:
        path: path to the given shader source to load.

    Returns:
        the source with `#include` directive replaced.

    Raise:
        FileNotFound error if either the given path or the paths in the `#include` directive not found.
    """
    path = os.path.join(SHADERS_ROOT, shader_name)
    root = os.path.dirname(path)

    with open(path) as infile:
        lines = infile.readlines()

    preprocessed: list[str] = []
    for line in lines:
        if line.startswith("#include"):
            line = line.rstrip()
            assert line[-1] == '"', (
                f"Doesn't support complex include directive: `{line}`"
            )
            include_path = line[10:-1]

            with open(os.path.join(root, include_path)) as infile:
                preprocessed.extend(infile.readlines())
        else:
            preprocessed.append(line)

    return "".join(preprocessed)


def generate_random_agents(
    *, nb_agents: int, world_width: int, world_height: int, nb_species: int
) -> npt.NDArray[AgentDtype]:
    agents = np.zeros(nb_agents, dtype=AgentDtype)

    for i in range(nb_agents):
        agents[i].location = np.array([world_width / 2.0, world_height / 2.0])
        agents[i].direction = random.random() * 2 * math.pi
        agents[i].species = random.randint(0, nb_species - 1)

    return agents


def run():
    # TODO: just testing that compute shader works with raylib.
    WIDTH, HEIGHT = 800, 600
    pr.init_window(WIDTH, HEIGHT, "Compute Shader Test")

    ctx = moderngl.create_context()

    # 2. Define Compute Shaders.
    compute_shader = ctx.compute_shader(load_shader_source("move.glsl"))
    visualize_shader = ctx.compute_shader(load_shader_source("visualize.glsl"))

    # 3. Create data structures and bind to GPU.
    # Convert species to buffer.
    species_arr = np.zeros(len(config.SPECIES), dtype=SpeciesDtype)
    for i, species in enumerate(config.SPECIES):
        species_arr[i].speed = species.speed
        species_arr[i].turn_angle = species.turn_angle
        species_arr[i].sensor_size = species.sensor_size
        species_arr[i].sensor_distance = species.sensor_distance
        species_arr[i].sensor_angle = species.sensor_angle
        species_arr[i].color = (
            np.array(species.color + (255,), dtype=np.float32) / 255.0
        )

    species_buffer = ctx.buffer(species_arr.tobytes())
    species_buffer.bind_to_storage_buffer(0)

    # Generate random agents.
    agents = generate_random_agents(
        nb_agents=config.NB_AGENTS,
        world_width=WIDTH,
        world_height=HEIGHT,
        nb_species=len(config.SPECIES),
    )
    agents_buffer = ctx.buffer(agents.tobytes())
    agents_buffer.bind_to_storage_buffer(1)

    # In/Out pheromone trails.
    pheromone_trails_in = ctx.texture3d(
        (WIDTH, HEIGHT, len(config.SPECIES)), components=1, dtype="f4"
    )
    pheromone_trails_out = ctx.texture3d(
        (WIDTH, HEIGHT, len(config.SPECIES)), components=1, dtype="f4"
    )

    pheromone_trails_in.bind_to_image(unit=2, read=True, write=False, level=0)
    pheromone_trails_out.bind_to_image(unit=3, read=True, write=True, level=0)

    # 4. Create a Texture for ModernGL to write to
    # We use a buffer that Raylib can also see
    pheromone_trails_img = ctx.texture((WIDTH, HEIGHT), components=4, dtype="f4")
    pheromone_trails_img.bind_to_image(unit=4, read=True, write=True, level=0)

    # We are 'lying' to Raylib and telling it to use the ID
    # that ModernGL already created.
    shared_tex = pr.Texture()
    shared_tex.id = pheromone_trails_img.glo
    shared_tex.width = WIDTH
    shared_tex.height = HEIGHT
    shared_tex.mipmaps = 1
    shared_tex.format = pr.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8

    # 5. Main Loop
    while not pr.window_should_close():
        # Dispatch Compute Shader
        compute_shader.run(WIDTH // 16, HEIGHT // 16)

        # Wait for the shader to complete.
        ctx.memory_barrier(moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # Shader to visualize the result.
        visualize_shader.run(WIDTH // 16, HEIGHT // 16)
        ctx.memory_barrier(moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # Render.
        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        pr.draw_texture(shared_tex, 0, 0, pr.WHITE)
        pr.end_drawing()

        # Switch buffer.
        pheromone_trails_in, pheromone_trails_out = (
            pheromone_trails_out,
            pheromone_trails_in,
        )
        pheromone_trails_in.bind_to_image(unit=2, read=True, write=False, level=0)
        pheromone_trails_out.bind_to_image(unit=3, read=True, write=True, level=0)

    pr.close_window()
