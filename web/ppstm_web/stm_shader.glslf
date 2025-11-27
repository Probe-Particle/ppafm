precision highp float;

uniform sampler2D uEigenvectors; // N x N texture (RGBA: px, py, pz, s)
uniform float uEigenvalues[100]; // Max 100 states
uniform sampler2D uAtomTexture;  // N x 1 texture (RGBA: x, y, z, 0)
uniform sampler2D uAtomParams;   // N x 1 texture (RGBA: decay, 0, 0, 0)
uniform float uBias;
uniform float uZTip;
uniform vec2 uResolution;
uniform float uL;
uniform bool uShowAtoms;
uniform float uAtomSize;
uniform int uAtomCount;

uniform bool uSingleOrbitalMode;
uniform int uSelectedOrbital;
uniform float uColorScale;

varying vec2 vUv;

// Simple Red-White-Blue colormap
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

// --- Modular Functions ---

// Evaluate contribution from a single atom 'j' to the wavefunction
// Returns phi_j(r)
float eval_atom_decay(int j, vec3 pos) {
    // Fetch atom position
    float u = (float(j) + 0.5) / float(uAtomCount);
    vec4 atomData = texture2D(uAtomTexture, vec2(u, 0.5));
    vec3 atom = atomData.xyz;
    
    // Fetch atom params (decay)
    vec4 params = texture2D(uAtomParams, vec2(u, 0.5));
    float decay = params.r;
    
    float dx = pos.x - atom.x;
    float dy = pos.y - atom.y;
    float dz = uZTip - atom.z;
    float r = sqrt(dx*dx + dy*dy + dz*dz);
    
    return exp(-decay * r);
}

// Evaluate wavefunction for orbital 'k' at position 'pos'
float eval_orbital(int k, vec3 pos) {
    float psi = 0.0;
    for (int j = 0; j < 100; j++) {
        if (j >= uAtomCount) break;
        
        float phi = eval_atom_decay(j, pos);
        
        // Fetch coefficient (s-orbital in Red channel)
        float u = (float(j) + 0.5) / float(uAtomCount);
        float v = (float(k) + 0.5) / float(uAtomCount);
        vec4 coeffs = texture2D(uEigenvectors, vec2(u, v));
        float coeff_s = coeffs.r; 
        
        psi += coeff_s * phi;
    }
    return psi;
}

void main() {
    // Physical coordinates with Aspect Ratio Correction
    float aspect = uResolution.x / uResolution.y;
    float x = (vUv.x * 2.0 - 1.0) * uL * aspect;
    float y = (vUv.y * 2.0 - 1.0) * uL;
    vec3 pos = vec3(x, y, 0.0); 
    
    float rho = 0.0;
    
    if (uSingleOrbitalMode) {
        // --- Single Orbital Mode ---
        // Visualize |psi|^2 for the selected orbital
        float psi = eval_orbital(uSelectedOrbital, pos);
        rho = psi * psi;
    } else {
        // --- Full STM Mode ---
        // Sum |psi_k|^2 for all states k within energy window [Ef, Ef + V]
        
        float Ef = 0.0;
        float rangeMin = (uBias > 0.0) ? Ef : Ef + uBias;
        float rangeMax = (uBias > 0.0) ? Ef + uBias : Ef;
        
        for (int k = 0; k < 100; k++) {
            if (k >= uAtomCount) break; 
            
            float en = uEigenvalues[k];
            if (en >= rangeMin && en <= rangeMax) {
                // Active state
                float psi = eval_orbital(k, pos);
                rho += psi * psi;
            }
        }
    }
    
    // Map rho to color
    vec3 color = colormap_heat(rho * uColorScale); 
    
    // Atoms overlay (Always on top if enabled)
    if (uShowAtoms) {
        for (int i = 0; i < 100; i++) {
            if (i >= uAtomCount) break;
            
            // Fetch atom from texture
            float u = (float(i) + 0.5) / float(uAtomCount);
            vec4 atomData = texture2D(uAtomTexture, vec2(u, 0.5));
            vec3 atom = atomData.xyz;
            
            float dx = x - atom.x;
            float dy = y - atom.y;
            float distSq = dx*dx + dy*dy;
            float rSq = uAtomSize * uAtomSize;
            
            if (distSq < rSq) {
                // Pseudo-3D Sphere Shading
                float z = sqrt(rSq - distSq);
                vec3 normal = normalize(vec3(dx, dy, z));
                vec3 lightDir = normalize(vec3(-0.5, -0.5, 1.0));
                
                float diffuse = max(dot(normal, lightDir), 0.0);
                float ambient = 0.4; 
                float specular = pow(max(dot(normal, lightDir), 0.0), 16.0);
                
                vec3 atomColor = vec3(0.0, 1.0, 0.0); // Bright Green
                color = atomColor * (diffuse + ambient) + vec3(1.0) * specular * 0.5;
            }
        }
    }
    
    gl_FragColor = vec4(color, 1.0);
}
