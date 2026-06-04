precision highp float;

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_scale;
uniform vec4 u_colors[4];
uniform float u_morphAmount;

#define MAX_BLOBS 64
uniform int u_numBlobs;
uniform vec3 u_blobs[MAX_BLOBS];
uniform vec4 u_blob_props[MAX_BLOBS];

vec3 hash3(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise2d(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < 8; i++) {
        if (i >= octaves) break;
        value += amplitude * noise2d(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value;
}

float blobShape(vec2 p, vec2 center, vec2 size, float angle, float seed, float morph_amount, float growth_bias) {
    vec2 offset = p - center;
    
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    vec2 rotated = vec2(
        offset.x * cos_a + offset.y * sin_a,
        -offset.x * sin_a + offset.y * cos_a
    );
    
    float ellipse_dist = length(rotated / size);
    
    vec2 worldPos = p;
    float noise_scale = 0.15;
    float morph_time = u_time * 0.002;
    
    vec3 hash_seed = hash3(vec2(seed, 0.0));
    vec2 noise_offset = hash_seed.xy * 1000.0;
    
    vec2 morph_coord = worldPos * noise_scale + noise_offset + vec2(morph_time * 2.0);
    float morph_noise = fbm(morph_coord, 3);
    
    float edge_falloff = 0.25;
    float dist_factor = 1.0 - smoothstep(1.0, 1.0 + edge_falloff, ellipse_dist);
    
    float threshold = 0.5;
    if (ellipse_dist > 1.0) {
        float edge_factor = (ellipse_dist - 1.0) / edge_falloff;
        threshold += edge_factor * 0.4;
    } else {
        threshold *= (1.0 - dist_factor * 0.3);
    }
    
    float base_noise = fbm(worldPos * noise_scale + noise_offset, 3);
    
    vec2 offsets[4];
    offsets[0] = vec2(1.0, 0.0);
    offsets[1] = vec2(-1.0, 0.0);
    offsets[2] = vec2(0.0, 1.0);
    offsets[3] = vec2(0.0, -1.0);
    
    float neighbor_count = 0.0;
    for (int i = 0; i < 4; i++) {
        vec2 neighbor_pos = p + offsets[i];
        vec2 neighbor_offset = neighbor_pos - center;
        vec2 neighbor_rotated = vec2(
            neighbor_offset.x * cos_a + neighbor_offset.y * sin_a,
            -neighbor_offset.x * sin_a + neighbor_offset.y * cos_a
        );
        float neighbor_dist = length(neighbor_rotated / size);
        float neighbor_factor = 1.0 - smoothstep(1.0, 1.0 + edge_falloff, neighbor_dist);
        float neighbor_noise = fbm(neighbor_pos * noise_scale + noise_offset, 3);
        neighbor_count += step(threshold, neighbor_factor * neighbor_noise);
    }
    
    bool is_boundary = neighbor_count < 4.0 && dist_factor > 0.1;
    
    if (morph_amount > 0.0) {
        float boundary_noise = fbm(morph_coord * 1.5, 2);
        float remove_prob = 0.0;
        
        if (growth_bias > 0.0) {
            if (boundary_noise > 0.65) {
                remove_prob = 0.02;
            } else if (boundary_noise > 0.6) {
                remove_prob = 0.01;
            }
        } else if (growth_bias < 0.0) {
            if (boundary_noise > 0.6) {
                remove_prob = 0.08;
            } else if (boundary_noise > 0.55) {
                remove_prob = 0.04;
            }
        } else {
            if (boundary_noise > 0.6) {
                remove_prob = 0.06;
            } else if (boundary_noise > 0.55) {
                remove_prob = 0.03;
            }
        }
        
        float remove_hash = hash(morph_coord + vec2(seed * 0.1, 0.0));
        if (is_boundary && remove_hash < remove_prob * morph_amount) {
            return 0.0;
        }
        
        if (!is_boundary && neighbor_count >= 2.0) {
            float expansion_noise = fbm(morph_coord * 0.8, 2);
            float add_prob = 0.0;
            
            if (growth_bias > 0.0) {
                if (expansion_noise < 0.48) {
                    add_prob = 0.08;
                } else if (expansion_noise < 0.52) {
                    add_prob = 0.05;
                } else if (expansion_noise < 0.58) {
                    add_prob = 0.02;
                }
            } else if (growth_bias < 0.0) {
                if (expansion_noise < 0.42) {
                    add_prob = 0.03;
                } else if (expansion_noise < 0.48 && neighbor_count >= 3.0) {
                    add_prob = 0.01;
                }
            } else {
                if (expansion_noise < 0.42) {
                    add_prob = 0.04;
                } else if (expansion_noise < 0.48 && neighbor_count >= 3.0) {
                    add_prob = 0.02;
                }
            }
            
            float add_hash = hash(morph_coord + vec2(seed * 0.1, 1.0));
            if (add_hash < add_prob * morph_amount) {
                threshold *= 0.92;
            }
        }
    }
    
    float morph_influence = (morph_noise - 0.5) * 0.01 * morph_amount;
    float blob_val = dist_factor * (1.0 + morph_influence);
    
    float final_val = blob_val * base_noise;
    return step(threshold, final_val);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_scale;
    vec2 pixel = floor(uv);
    
    vec4 color = u_colors[0];
    
    float shadow_layer = 0.0;
    float light_layer = 0.0;
    float highlight_layer = 0.0;
    
    for (int i = 0; i < MAX_BLOBS; i++) {
        if (i >= u_numBlobs) break;
        
        vec3 blob_data = u_blobs[i];
        vec4 blob_props = u_blob_props[i];
        float layer_type = blob_data.x;
        vec2 center = vec2(blob_data.y, blob_data.z);
        float growth_bias = blob_props.x;
        
        vec2 world_center = mod(center, u_resolution);
        
        vec2 local_pos = pixel - world_center;
        
        float seed = blob_data.x * 10000.0 + blob_data.y * 100.0 + blob_data.z * 10.0 + float(i) * 1000.0;
        vec3 hash_seed = hash3(vec2(seed, 1.0));
        
        float base_size = 18.0;
        float size_x = base_size + hash_seed.x * 12.0;
        float size_y = base_size + hash_seed.y * 12.0;
        float angle = hash_seed.z * 6.28318;
        
        if (layer_type > 1.5) {
            size_x *= 0.65;
            size_y *= 0.65;
        }
        
        float dist = length(local_pos);
        float max_dist = max(size_x, size_y) * 1.5;
        if (dist < max_dist) {
            float morph_amt = 1.0;
            if (layer_type > 1.5) {
                morph_amt = 0.5;
            }
            float blob_val = blobShape(pixel, world_center, vec2(size_x, size_y), angle, seed, morph_amt * u_morphAmount, growth_bias);
            
            if (blob_val > 0.5) {
                if (layer_type < 0.5) {
                    shadow_layer = 1.0;
                } else if (layer_type < 1.5) {
                    light_layer = 1.0;
                } else {
                    highlight_layer = 1.0;
                }
            }
        }
    }
    
    if (shadow_layer > 0.5) {
        color = u_colors[1];
    }
    if (light_layer > 0.5) {
        color = u_colors[2];
    }
    if (highlight_layer > 0.5 && light_layer > 0.5) {
        color = u_colors[3];
    }
    
    gl_FragColor = color;
}
