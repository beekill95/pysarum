#version 430

// Implement average kernel for diffusion.
layout(local_size_x = 16, local_size_y = 16) in;
layout(r32f, binding = 3) uniform image3D trails;

uniform uint kernel_size;

void main() {
    int half_size = int(kernel_size) / 2;
    ivec3 trails_size = imageSize(trails);
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

    for (int z = 0; z < trails_size.z; z++) {
        float value = 0.0;
        float count = 0.0;

        for (int i = max(pos.x - half_size, 0); i < min(pos.x + half_size + 1, trails_size.x - 1); i++) {
            for (int j = max(pos.y - half_size, 0); j < min(pos.y + half_size + 1, trails_size.y - 1); j++) {
                value += imageLoad(trails, ivec3(i, j, z)).r;
                count += 1.0;
            }
        }

        if (count > 0.0) {
            imageStore(trails, ivec3(pos, z), vec4(value / count));
        }
    }
}
