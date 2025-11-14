// Fragment shader for Stage 1: render atoms as radial blobs to verify geometry / z-plane
#ifdef GL_ES
precision highp float;
#endif

uniform vec2  uResolution;   // viewport size in pixels
uniform float uZPlane;       // imaging plane z
uniform float uScale;        // lateral scale (Å per screen unit)
uniform vec2  uCenter;       // center of the view in sample coordinates

uniform int   uNumAtoms;
uniform vec4  uAtoms[256];   // xyz in Å, w unused for now

varying vec2 vUv;

// void main() {
//     // Map screen coordinates to sample XY coordinates
//     vec2 uv  = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0; // -1..1
//     vec2 posXY = uCenter + uv * uScale;
//     float z   = uZPlane;

//     float density = 0.0;

//     const float decay = 1.0;   // radial decay; tweak as needed
//     const float sigma = 0.7;   // width of Gaussian (Å)
//     const float sigma2 = sigma * sigma;

//     for (int i = 0; i < 256; i++) {
//         if (i >= uNumAtoms) break;
//         vec3 atom = uAtoms[i].xyz;
//         vec3 dr   = vec3(posXY, z) - atom;
//         float r2  = dot(dr, dr);
//         // Gaussian-like blob
//         float w   = exp(-r2 / (2.0 * sigma2));
//         density  += w;
//     }

//     // Simple tone mapping
//     //float val = 1.0 - exp(-density * 1.5);

//     //float val = sin(uv.x * uv.y);
//     //uAtoms[i].xyz
//     //vec3 col = vec3( posXY-uAtoms[0].xy*0.00001, 0.0 );
//     vec3 col = vec3(density);
//     gl_FragColor = vec4(col, 1.0);
// }



void main() {
    vec2 uv  = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
    vec2 posXY = uCenter + uv * uScale;
    float density = 0.0;
    float sigma   = 0.7;
    float sigma2  = sigma * sigma;

    for (int i = 0; i < 256; i++) {
        if (i >= uNumAtoms) break;
        vec2 dr = posXY - uAtoms[i].xy;        // 2D distance only
        float r2 = dot(dr, dr);
        density += exp(-r2 / (2.0 * sigma2));
    }
    // Stronger tone mapping
    float val = 1.0 - exp(-density * 3.0);
    gl_FragColor = vec4(vec3(val), 1.0);
}


// ---- DEBUG KERNEL ----
// void main() {
//     vec2 uv  = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
//     vec2 posXY = uCenter + uv * uScale;
//     if (uNumAtoms > 0) {
//         vec2 dr = posXY - uAtoms[0].xy;
//         float r2 = dot(dr, dr);
//         float sigma = 1.0;
//         // One bright blob around atom 0
//         float val = exp(-r2 / (2.0 * sigma * sigma));
//         gl_FragColor = vec4(vec3(val), 1.0);
//     } else {
//         //gl_FragColor = vec4(0.0,0.5,0.0,1.0);
//         gl_FragColor = vec4( posXY,0.0,1.0);
//     }
// }