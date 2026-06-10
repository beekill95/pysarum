#version 430
#extension GL_ARB_shading_language_include : require
#include "./agent.glsl"

#define NEG_INF -3.40282347e+38

layout(local_size_x = 1) in;
layout(std430, binding = 0) readonly buffer SpeciesList {
    Species species[];
};
layout(std430, binding = 1) buffer Agents {
    Agent agents[];
};
layout(r32f, binding = 2) readonly image3D in_trails;
layout(r32f, binding = 3) writeonly image3D out_trails;

uniform float dt;

float random(vec2 seed) {
    // Pseudo random generator from Google AI.
    return fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453);
}

float sense(uint agent_idx, float direction) {
    ivec3 trails_size = imageSize(in_trails);

    uint species_idx = agent_idx[agent_idx].species;
    float sensor_angle = species[species_idx].sensor_angle;
    vec2 sensor_loc = agents[agent_idx].location + vec2(cos(sensor_angle), sin(sensor_angle)) * species[species_idx].sensor_distance;

    vec2 sensoring_topleft = sensor_loc - float(species[species_idx].sensor_size) / 2.0;
    vec2 sensoring_botright = sensor_loc + float(species[species_idx].sensor_size) / 2.0;

    // If the sensoring area is outside the trails region, return minus infinity.
    if (sensoring_botright.x < 0
            || sensoring_botright.y < 0
            || sensoring_topleft.x >= trails_size.x
            || sensoring_topleft.y >= trails_size.y) {
        return NEG_INF;
    }

    float species_pheromone = 0.0;
    float other_pheromone = 0.0;

    float species_pheromone_cnt = 0.0;
    float other_pheromone_cnt = 0.0;

    // Calculating the strength of the species's pheromones and the other species' pheromones.
    // Remember to handle the out of region cases.
    for (int i = int(sensoring_topleft.x); i < int(sensoring_botright.x); i++) {
        for (int j = int(sensoring_topleft.y); j < int(sensoring_botright.y); j++) {
            if (i < 0 || j < 0 || i >= trails_size.x || j >= trails_size.y) {
                continue;
            }

            species_pheromone += imageLoad(in_trails, ivec3(i, j, species_idx)).r;
            species_pheromone_cnt += 1.0;

            for (int z = 0; z < trails_size.z; z++) {
                if (z != species_idx) {
                    other_pheromone += imageLoad(in_trails, ivec3(i, j, z)).r;
                    other_pheromone_cnt += 1.0;
                }
            }
        }
    }

    return species_pheromone / species_pheromone_cnt - other_pheromone / other_pheromone_cnt;
}

float steer(uint agent_idx) {
    // Determine whether to keep the current direction (return 0),
    // turning right (return 1), or turning left (return -1).
    uint species_idx = agents.species;
    float left_sensor = sense(agent_idx, direction + species[species_idx].sensor_angle)
    float front_sensor = sense(agent_idx, direction)
    float right_sensor = sense(agent_idx, direction - species[species_idx].sensor_angle)

    if (front_sensor < left_sensor && front_sensor < right_sensor) {
        float value = random(vec2(float(agent_idx), dt));
        return float(round(value));
    }

    if (front_sensor < left_sensor) {
        return 1.0;
    } else {
        return -1.0;
    }
}

void main() {
    uint agent_idx = gl_GlobalInvocationID.x;
    vec2 random_seed = vec2(float(agent_idx), dt);

    // Determine whether the agent will turn to different directions?
    float turn = steer(agent_idx);
    float new_direction = agents[agent_idx].direction + turn * agents[agent_idx].turn_angle;

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
