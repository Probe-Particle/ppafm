precision highp float;

uniform sampler2D uTexture;
uniform float uIso;
uniform float uMin;
uniform float uMax;

varying vec2 vUv;

// Simple Red-White-Blue colormap
vec3 colormap(float t) {
    // t is expected to be in [0, 1]
    // 0.0 -> Blue
    // 0.5 -> White
    // 1.0 -> Red
    
    vec3 c;
    if (t < 0.5) {
        // Blue to White
        float x = t * 2.0;
        c = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 1.0, 1.0), x);
    } else {
        // White to Red
        float x = (t - 0.5) * 2.0;
        c = mix(vec3(1.0, 1.0, 1.0), vec3(1.0, 0.0, 0.0), x);
    }
    return c;
}

// Heatmap style: Black -> Red -> Yellow -> White
vec3 colormap_heat(float t) {
    float x = clamp(t, 0.0, 1.0);
    vec3 c;
    if (x < 0.33) {
        c = mix(vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), x / 0.33);
    } else if (x < 0.66) {
        c = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), (x - 0.33) / 0.33);
    } else {
        c = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0), (x - 0.66) / 0.34);
    }
    return c;
}

void main() {
    float val = texture2D(uTexture, vUv).r;
    
    // Normalize for visualization
    // We assume val is positive (density)
    // uMax should be the max value in the texture or a user-defined scale
    
    float normVal = val / uMax;
    
    // Apply colormap
    vec3 color = colormap_heat(normVal);
    
    gl_FragColor = vec4(color, 1.0);
}
