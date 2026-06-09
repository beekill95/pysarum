#version 430
#extension GL_ARB_shading_language_include : require
#include "./agent.glsl"

layout(local_size_x=1) in;
layout(std430, binding=0) readonly buffer SpeciesList { Species species[]; };
layout(std430, binding=1) buffer Agents { Agent agents[]; };
layout(r32f, binding=2) readonly image3D in_trails;
layout(r32f, binding=3) writeonly image3D out_trails;

uniform float dt;

float random(vec2 seed) {
    // Pseudo random generator from Google AI.
    return fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453);
}

float sense(uint agent_idx, float direction) {}

float steer(uint agent_idx) {
    // Determine whether to keep the current direction (return 0),
    // turning right (return 1), or turning left (return -1).
    uint species_idx = agents.species;
    float left_sensor = steer(agent_idx, direction + species[species_idx].sensor_angle)
    float front_sensor = steer(agent_idx, direction)
    float right_sensor = steer(agent_idx, direction - species[species_idx].sensor_angle)

    if (front_sensor < left_sensor && front_sensor < right_sensor) {
        float value = random(vec2(float(agent_idx), dt));
        // TODO
    }
    // TODO
}

void main() {
    uint agent_idx = gl_GlobalInvocationID.x;
    vec2 random_seed = vec2(float(agent_idx), dt);

    // Determine whether the agent will turn to different directions?
    float turn = steer(agent_idx);
    float new_direction = agents[agent_idx].direction +  turn * agents[agent_idx].turn_angle;

    // Caculate the agent's new position.
    vec2 new_location = agents[agent_idx].location + dt * agents[agent_idx].speed;

    // Make sure the new location is within the trails, if not, move the agent in a random direction.
    // TODO

    // Mark the trails.
    // TODO

    // Update the agent's location and current direction.
    agents[agent_idx].direction = new_direction;
    agents[agent_idx].location = new_location;
}
