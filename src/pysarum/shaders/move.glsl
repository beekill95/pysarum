#version 430
#include "./common.glsl"

layout(local_size_x = 1) in;
layout(std430, binding = 0) readonly buffer SpeciesList {
    Species species[];
};
layout(std430, binding = 1) buffer Agents {
    Agent agents[];
};
layout(r32f, binding = 2) uniform readonly image3D in_trails;
layout(r32f, binding = 3) uniform writeonly image3D out_trails;

uniform float dt;

float random(out vec2 seed) {
    // Pseudo random generator from Google AI.
    float result = fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453);

    // TODO: Generate a new seed.
    seed += 1.0;

    return result;
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
    float left_sensor = sense(agent_idx, direction + species[species_idx].sensor_angle);
    float front_sensor = sense(agent_idx, direction);
    float right_sensor = sense(agent_idx, direction - species[species_idx].sensor_angle);

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
    Agent agent = agents[agent_idx];
    vec2 random_seed = vec2(float(agent_idx), dt);
    ivec3 trails_size = imageSize(in_trails);

    // Determine whether the agent will turn to different directions?
    float turn = steer(agent_idx);
    float new_direction = agent.direction + turn * agent.turn_angle;

    // Caculate the agent's new position.
    vec2 new_location = agent.location + dt * agent.speed;

    // Make sure the new location is within the trails, if not, move the agent in a random direction.
    if (new_location.x >= trails_size.x || new_location.x <= 0) {
        float random_angle = random(random_seed) * PI;

        if (new_location.x <= 0) {
            new_location.x = 0;
            new_direction = random_angle - PI / 2.0;
        } else {
            new_location.x = trails_size.x - 1;
            new_direction = random_angle + PI / 2.0;
        }
    }

    if (new_location.y >= trails_size.y || new_location.y <= 0) {
        float random_angle = random(random_seed) * PI;

        if (new_location.y <= 0) {
            new_location.y = 0;
            new_direction = random_angle;
        } else {
            new_location.y = trails_size.y - 1;
            new_direction = random_angle + PI;
        }
    }

    // Mark the trails with pheromone.
    vec2 diff = new_location - agent.location;
    vec2 distance = abs(diff);
    uint nb_steps = uint(ceil(max(distance.x, distance.y)));
    float step = diff / float(nb_steps);
    for (uint i = 0; i < nb_steps; i++) {
        ivec2 xy = ivec2(agent.location + float(i) * step);
        imageStore(out_trails, ivec3(xy, agent.species), vec4(1.0));
    }

    // Update the agent's location and current direction.
    agents[agent_idx].direction = new_direction;
    agents[agent_idx].location = new_location;
}
