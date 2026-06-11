from __future__ import annotations

import os

import moderngl
import pyray as pr

SHADERS_ROOT = os.path.join(os.path.dirname(__file__), "shaders")


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
            assert line[-1] == '"', "Doesn't support complex include directive"
            include_path = line[10:-1]

            with open(os.path.join(root, include_path)) as infile:
                preprocessed.extend(infile.readlines())
        else:
            preprocessed.append(line)

    return "".join(preprocessed)


def run():
    # TODO: just testing that compute shader works with raylib.
    pr.init_window(800, 600, "Compute Shader Test")

    ctx = moderngl.create_context()

    # 2. Define Compute Shader
    compute_shader = ctx.compute_shader(load_shader_source(os.path.join()))

    # 3. Create a Texture for ModernGL to write to
    # We use a buffer that Raylib can also see
    mgl_texture = ctx.texture((800, 600), 4, dtype="f1")
    mgl_texture.bind_to_image(0, read=False, write=True)

    # pr_tex = pr.load_texture_from_image(pr.gen_image_color(800, 600, pr.BLANK))

    # 2. Get the underlying OpenGL Texture ID (The handle)
    texture_id = mgl_texture.glo

    # 3. Create a Raylib Texture2D manually
    # We are 'lying' to Raylib and telling it to use the ID
    # that ModernGL already created.
    shared_tex = pr.Texture()
    shared_tex.id = texture_id
    shared_tex.width = 800
    shared_tex.height = 600
    shared_tex.mipmaps = 1
    shared_tex.format = pr.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8

    # 4. Main Loop
    while not pr.window_should_close():
        # Dispatch Compute Shader
        compute_shader.run(800 // 16, 600 // 16)

        # No need these two steps since we've configured raylib to read from moderngl's texture
        # 2. Get the data from ModernGL texture (as bytes)
        # pixels = mgl_texture.read(alignment=1)
        # c_pixel_data = (ctypes.c_ubyte * len(pixels)).from_buffer_copy(pixels)

        # 3. Update the Raylib texture with these new pixels
        # pr.update_texture(pr_tex, pr.ffi.new("unsigned char []", pixels))

        #
        # pr.glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # 4. Render
        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        pr.draw_texture(shared_tex, 0, 0, pr.WHITE)
        pr.end_drawing()

    pr.close_window()
