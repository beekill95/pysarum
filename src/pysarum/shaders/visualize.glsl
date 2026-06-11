#version 430
#include "common.glsl"

layout(local_size_x = 1, local_size_y = 1) in;

layout(std430, binding = 0) readonly buffer SpeciesList {
    Species species[];
};

layout(r32f, binding = 3) uniform image3D in_trails;
layout(rgba32f, binding = 4) uniform image2D out_img;

void main() {
    ivec2 loc = gl_GlobalInvocationID.xy;
    ivec3 trails_size = imageSize(in_trails);

    if (loc.x >= trails_size.x || loc.y >= trails_size.y) {
        return;
    }

    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < trails_size.z; i++) {
        color += imageLoad(in_trails, ivec3(loc, i)).x * species[i].color;
    }

    // Normalizing the alpha channel.
    color.a /= float(trails_size.z);

    // Set the color to the output image.
    imageStore(out_img, loc, color);
}
