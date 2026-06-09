struct Agent {
    vec2 location;
    float direction;
    uint species;
};

struct Species {
    float speed;
    float turn_angle;

    uint sensor_size;
    uint sensor_distance;
    float sensor_angle;

    vec3 color;
};
