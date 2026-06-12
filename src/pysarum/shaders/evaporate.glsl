#version 430

layout(local_size_x = 16, local_size_y = 16) in;
layout(r32f, binding = 3) uniform image3D in_trails;

uniform float dt;
uniform float evaporation_speed;

void main() {
    for (int i = 0; i < imageSize(in_trails).z; i++) {
        ivec3 pos = ivec3(gl_GlobalInvocationID.xy, i);
        float value = imageLoad(in_trails, pos).r - dt * evaporation_speed;
        imageStore(in_trails, pos, vec4(max(value, 0.0)));
    }
}
