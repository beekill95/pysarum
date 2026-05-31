from __future__ import annotations

import moderngl
import pyray as pr

SHADER_SOURCE = """\
#version 430
layout(local_size_x=16, local_size_y=16) in;
layout(rgba32f, binding=0) uniform image2D dest_tex;

void main() {
    ivec2 texel_pos = ivec2(gl_GlobalInvocationID.xy);
    vec4 color = vec4(float(texel_pos.x)/800.0, float(texel_pos.y)/600.0, 1.0, 1.0);
    imageStore(dest_tex, texel_pos, color);
}
"""


def run():
    # TODO: just testing that compute shader works with raylib.
    pr.init_window(800, 600, "Compute Shader Test")

    ctx = moderngl.create_context()

    # 2. Define Compute Shader
    compute_shader = ctx.compute_shader(SHADER_SOURCE)

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
