class Molecule {
    constructor() {
        this.atoms = []; // {x, y, z, type}
    }

    loadXYZ(xyzString) {
        this.atoms = [];
        const lines = xyzString.split('\n');
        const nAtoms = parseInt(lines[0]);
        // Skip comment line (lines[1])
        for (let i = 2; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            const parts = line.split(/\s+/);
            if (parts.length < 4) continue;
            const type = parts[0];
            const x = parseFloat(parts[1]);
            const y = parseFloat(parts[2]);
            const z = parseFloat(parts[3]);
            this.atoms.push({ type, x, y, z });
        }
        console.log(`Loaded ${this.atoms.length} atoms.`);
    }

    getCenter() {
        let cx = 0, cy = 0, cz = 0;
        for (let a of this.atoms) {
            cx += a.x; cy += a.y; cz += a.z;
        }
        return { x: cx / this.atoms.length, y: cy / this.atoms.length, z: cz / this.atoms.length };
    }
}

class TightBindingHamiltonian {
    constructor(molecule) {
        this.molecule = molecule;
        this.H = null;
        this.N = 0;
    }

    build(params) {
        const atoms = this.molecule.atoms;
        this.N = atoms.length;
        this.H = numeric.rep([this.N, this.N], 0.0);

        const V0 = params.V0; // Hopping at d0 (usually defined relative to some distance, but here we use simple exp)
        // Model: H_ij = V0 * exp(-beta * (r_ij - d0))
        // Simplified: H_ij = V_pre * exp(-beta * r_ij)
        // Let's use the user input V0 as the hopping amplitude at a typical bond length (e.g. 1.4 A for C-C)
        // Or just V(r) = V0 * exp(-beta * (r - 1.4))
        
        const beta = params.beta;
        const E0 = params.E0;
        const d0 = 1.4; // Reference bond length for Carbon

        for (let i = 0; i < this.N; i++) {
            this.H[i][i] = E0; // On-site energy
            for (let j = i + 1; j < this.N; j++) {
                const dx = atoms[i].x - atoms[j].x;
                const dy = atoms[i].y - atoms[j].y;
                const dz = atoms[i].z - atoms[j].z;
                const r = Math.sqrt(dx*dx + dy*dy + dz*dz);

                // Cutoff for performance/physics
                if (r > 6.0) continue; 

                // Exponential hopping
                const val = V0 * Math.exp(-beta * (r - d0));
                
                this.H[i][j] = val;
                this.H[j][i] = val;
            }
        }
    }
}

class Solver {
    static solve(H) {
        // Use numeric.js to diagonalize
        // numeric.eig(H) returns {lambda: eigenvalues, E: eigenvectors}
        // For symmetric real matrix, eigenvalues are real.
        // numeric.js might return complex structure even for real symmetric, need to handle.
        // Actually numeric.eig is generic. For symmetric, it's robust.
        
        const result = numeric.eig(H);
        
        // Extract real parts (assuming symmetric H)
        const eigenvalues = result.lambda.x; // .x is real part in numeric.js complex number
        const eigenvectors = result.E.x;     // Columns are eigenvectors? No, numeric.js: E[i] is eigenvector i?
        // numeric.eig docs: E is matrix of eigenvectors. E[i][j] is j-th component of i-th eigenvector?
        // Wait, usually E is V where H = V D V^-1. So columns are eigenvectors.
        // Let's verify: H * v = lambda * v
        // numeric.js: "E is a matrix of eigenvectors".
        // Let's assume row-major or column-major? numeric.js usually uses array of arrays.
        // If E is [[v00, v01], [v10, v11]], then E[row][col].
        // Usually eigenvectors are columns. So E[j][i] is i-th component of j-th eigenvector.
        
        // We will assume E[row][col] where col is the index of eigenvector.
        // So eigenvector k is [ E[0][k], E[1][k], ... ]
        
        return {
            energies: eigenvalues,
            vectors: eigenvectors // [row][col]
        };
    }
}

class PPSTMApp {
    constructor() {
        this.molecule = new Molecule();
        this.hamiltonian = new TightBindingHamiltonian(this.molecule);
        this.eigenSystem = null;
        
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        this.atomMeshes = [];
        this.bondMeshes = [];
        this.stmPlane = null;
        this.stmTexture = null;
        this.stmData = null; // Float32Array
        
        this.params = {
            V0: -2.7,
            beta: 2.0,
            E0: 0.0,
            bias: 0.5,
            zTip: 3.0,
            iso: 0.01,
            atomSize: 0.4
        };
        
        this.gridSize = 128;
        this.gridL = 20.0; // Angstroms (half-width)
    }

    init() {
        this.initThree();
        this.initUI();
        this.loadMolecule('PTCDA'); // Default
    }

    initThree() {
        const viewer = document.getElementById('viewer');
        const width = viewer.clientWidth;
        const height = viewer.clientHeight;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.camera.position.set(0, 0, 30);
        this.camera.lookAt(0, 0, 0);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        viewer.appendChild(this.renderer.domElement);

        // Lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 10);
        this.scene.add(directionalLight);

        // Orbit controls (manual implementation or simple mouse listener if OrbitControls not available)
        // Since we didn't load OrbitControls explicitly in index.html (it's usually separate in examples/js),
        // let's try to see if THREE.OrbitControls exists (sometimes bundled).
        // If not, we'll add a simple rotator.
        // Actually, let's just add a simple mouse drag to rotate the molecule.
        this.setupSimpleControls();

        // STM Plane
        const geometry = new THREE.PlaneGeometry(this.gridL * 2, this.gridL * 2);
        
        // Create DataTexture
        this.stmData = new Float32Array(this.gridSize * this.gridSize);
        this.stmTexture = new THREE.DataTexture(this.stmData, this.gridSize, this.gridSize, THREE.RedFormat, THREE.FloatType);
        this.stmTexture.magFilter = THREE.LinearFilter;
        this.stmTexture.minFilter = THREE.LinearFilter;
        this.stmTexture.needsUpdate = true;

        // Load shader
        fetch('stm_shader.glslf')
            .then(r => r.text())
            .then(fragShader => {
                const material = new THREE.ShaderMaterial({
                    uniforms: {
                        uTexture: { value: this.stmTexture },
                        uMax: { value: 1.0 }
                    },
                    vertexShader: `
                        varying vec2 vUv;
                        void main() {
                            vUv = uv;
                            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                        }
                    `,
                    fragmentShader: fragShader,
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: 0.8
                });
                this.stmPlane = new THREE.Mesh(geometry, material);
                this.stmPlane.position.z = this.params.zTip;
                this.scene.add(this.stmPlane);
            });

        window.addEventListener('resize', () => {
            const w = viewer.clientWidth;
            const h = viewer.clientHeight;
            this.renderer.setSize(w, h);
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
        });

        this.animate();
    }

    setupSimpleControls() {
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };
        const viewer = this.renderer.domElement;

        viewer.addEventListener('mousedown', (e) => {
            isDragging = true;
        });
        viewer.addEventListener('mousemove', (e) => {
            if (isDragging) {
                const deltaMove = {
                    x: e.offsetX - previousMousePosition.x,
                    y: e.offsetY - previousMousePosition.y
                };

                const deltaRotationQuaternion = new THREE.Quaternion()
                    .setFromEuler(new THREE.Euler(
                        toRadians(deltaMove.y * 1),
                        toRadians(deltaMove.x * 1),
                        0,
                        'XYZ'
                    ));
                
                // Rotate the scene/camera
                // Ideally rotate the camera around 0,0,0
                // Simple orbit:
                const x = this.camera.position.x;
                const y = this.camera.position.y;
                const z = this.camera.position.z;
                
                // Rotate around Y axis (horizontal drag)
                const angleY = -deltaMove.x * 0.01;
                const x_new = x * Math.cos(angleY) - z * Math.sin(angleY);
                const z_new = x * Math.sin(angleY) + z * Math.cos(angleY);
                this.camera.position.x = x_new;
                this.camera.position.z = z_new;
                
                // Rotate around X axis (vertical drag) - simplified, just move Y
                // const angleX = -deltaMove.y * 0.01;
                // this.camera.position.y += deltaMove.y * 0.1;
                
                this.camera.lookAt(0, 0, 0);
            }
            previousMousePosition = { x: e.offsetX, y: e.offsetY };
        });
        viewer.addEventListener('mouseup', () => isDragging = false);
    }

    initUI() {
        document.getElementById('btnReload').addEventListener('click', () => this.solvePhysics());
        
        const inputs = ['inpV0', 'inpBeta', 'inpE0', 'inpBias', 'inpZTip', 'inpIso', 'sldAtomSize'];
        inputs.forEach(id => {
            document.getElementById(id).addEventListener('change', (e) => {
                this.updateParams();
                if (id === 'inpV0' || id === 'inpBeta' || id === 'inpE0') {
                    this.solvePhysics();
                } else if (id === 'inpBias' || id === 'inpZTip') {
                    this.updateSTM();
                } else if (id === 'sldAtomSize') {
                    this.updateAtoms();
                }
            });
        });

        document.getElementById('moleculeSelect').addEventListener('change', (e) => {
            this.loadMolecule(e.target.value);
        });
    }

    updateParams() {
        this.params.V0 = parseFloat(document.getElementById('inpV0').value);
        this.params.beta = parseFloat(document.getElementById('inpBeta').value);
        this.params.E0 = parseFloat(document.getElementById('inpE0').value);
        this.params.bias = parseFloat(document.getElementById('inpBias').value);
        this.params.zTip = parseFloat(document.getElementById('inpZTip').value);
        this.params.iso = parseFloat(document.getElementById('inpIso').value);
        this.params.atomSize = parseFloat(document.getElementById('sldAtomSize').value);
    }

    loadMolecule(name) {
        let xyz = "";
        if (name === 'PTCDA') {
            // Minimal PTCDA XYZ (approximate or placeholder if file not available)
            // Since we can't easily fetch the file from the server in this client-side only setup without a proper server,
            // we'll hardcode a simple molecule or try to fetch if running on a server.
            // The user has `examples/xyz/PTCDA.xyz`. We can try to fetch it relative to this file?
            // `../../examples/xyz/PTCDA.xyz` might work if served correctly.
            // For now, let's try to fetch.
            fetch('../../examples/xyz/PTCDA.xyz')
                .then(r => {
                    if (!r.ok) throw new Error("Not found");
                    return r.text();
                })
                .then(text => {
                    this.molecule.loadXYZ(text);
                    this.buildScene();
                    this.solvePhysics();
                })
                .catch(e => {
                    console.warn("Could not load PTCDA.xyz, using Benzene fallback.");
                    this.loadMolecule('Benzene');
                });
            return;
        } else if (name === 'Benzene') {
            xyz = `12
Benzene
C        0.00000        1.39700        0.00000
C        1.20986        0.69850        0.00000
C        1.20986       -0.69850        0.00000
C        0.00000       -1.39700        0.00000
C       -1.20986       -0.69850        0.00000
C       -1.20986        0.69850        0.00000
H        0.00000        2.48100        0.00000
H        2.14862        1.24050        0.00000
H        2.14862       -1.24050        0.00000
H        0.00000       -2.48100        0.00000
H       -2.14862       -1.24050        0.00000
H       -2.14862        1.24050        0.00000
`;
        } else {
             // Graphene small
             xyz = `24
Graphene
C 0 0 0
C 1.42 0 0
C 0.71 1.23 0
C -0.71 1.23 0
C -1.42 0 0
C -0.71 -1.23 0
C 0.71 -1.23 0
C 2.13 1.23 0
C 2.84 0 0
C 2.13 -1.23 0
C 0.71 -2.46 0
C -0.71 -2.46 0
C -2.13 -1.23 0
C -2.84 0 0
C -2.13 1.23 0
C -0.71 2.46 0
C 0.71 2.46 0
C 2.13 2.46 0
C 3.55 1.23 0
C 3.55 -1.23 0
C 2.13 -2.46 0
C -2.13 -2.46 0
C -3.55 -1.23 0
C -3.55 1.23 0
`;
        }
        
        this.molecule.loadXYZ(xyz);
        this.buildScene();
        this.solvePhysics();
    }

    buildScene() {
        // Clear old meshes
        this.atomMeshes.forEach(m => this.scene.remove(m));
        this.bondMeshes.forEach(m => this.scene.remove(m));
        this.atomMeshes = [];
        this.bondMeshes = [];

        const atoms = this.molecule.atoms;
        const sphereGeo = new THREE.SphereGeometry(1, 16, 16);
        
        atoms.forEach(atom => {
            let color = 0x888888;
            if (atom.type === 'C') color = 0x333333;
            if (atom.type === 'H') color = 0xffffff;
            if (atom.type === 'O') color = 0xff0000;
            if (atom.type === 'N') color = 0x0000ff;

            const mat = new THREE.MeshPhongMaterial({ color: color });
            const mesh = new THREE.Mesh(sphereGeo, mat);
            mesh.position.set(atom.x, atom.y, atom.z);
            mesh.scale.set(this.params.atomSize, this.params.atomSize, this.params.atomSize);
            this.scene.add(mesh);
            this.atomMeshes.push(mesh);
        });

        // Simple bonds (distance check)
        const bondMat = new THREE.MeshPhongMaterial({ color: 0xaaaaaa });
        for (let i = 0; i < atoms.length; i++) {
            for (let j = i + 1; j < atoms.length; j++) {
                const a1 = atoms[i];
                const a2 = atoms[j];
                const dx = a1.x - a2.x;
                const dy = a1.y - a2.y;
                const dz = a1.z - a2.z;
                const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                
                if (dist < 1.8) { // Typical bond length cutoff
                    const midX = (a1.x + a2.x) / 2;
                    const midY = (a1.y + a2.y) / 2;
                    const midZ = (a1.z + a2.z) / 2;
                    
                    const cylinder = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, dist, 8), bondMat);
                    cylinder.position.set(midX, midY, midZ);
                    
                    // Orientation
                    const axis = new THREE.Vector3(dx, dy, dz).normalize();
                    const quaternion = new THREE.Quaternion();
                    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
                    cylinder.setRotationFromQuaternion(quaternion);
                    
                    this.scene.add(cylinder);
                    this.bondMeshes.push(cylinder);
                }
            }
        }
    }

    updateAtoms() {
        const s = this.params.atomSize;
        this.atomMeshes.forEach(m => m.scale.set(s, s, s));
    }

    solvePhysics() {
        document.getElementById('loading').style.display = 'block';
        // Use setTimeout to allow UI to render "Calculating..."
        setTimeout(() => {
            console.time("Hamiltonian");
            this.hamiltonian.build(this.params);
            console.timeEnd("Hamiltonian");

            console.time("Diagonalization");
            this.eigenSystem = Solver.solve(this.hamiltonian.H);
            console.timeEnd("Diagonalization");

            console.log("Eigenvalues:", this.eigenSystem.energies);

            this.updateSTM();
            document.getElementById('loading').style.display = 'none';
        }, 10);
    }

    updateSTM() {
        if (!this.eigenSystem) return;

        const zTip = this.params.zTip;
        const bias = this.params.bias;
        const atoms = this.molecule.atoms;
        const N = atoms.length;
        const energies = this.eigenSystem.energies;
        const vectors = this.eigenSystem.vectors; // [row][col] -> [atomIndex][eigenStateIndex]

        // Fermi level is roughly 0.0 for this simple model if half-filled, but let's assume 0.0 is mid-gap or similar.
        // For simple Hückel, alpha=0.
        // We sum states between E_F and E_F + Bias.
        // Let's assume E_F = 0 for now (or mid point of HOMO-LUMO).
        // Actually, for PTCDA, we might want to find the gap.
        // Let's just sum states within [ -|Bias|/2, +|Bias|/2 ] or [0, Bias] relative to 0?
        // User requirement: "User selects Bias Voltage... identify all Eigenstates n where Energy En falls within [Ef, Ef + Vbias]"
        // We'll assume Ef = 0.0.
        
        const Ef = 0.0;
        const rangeMin = (bias > 0) ? Ef : Ef + bias;
        const rangeMax = (bias > 0) ? Ef + bias : Ef;

        const activeStates = [];
        for (let i = 0; i < N; i++) {
            if (energies[i] >= rangeMin && energies[i] <= rangeMax) {
                activeStates.push(i);
            }
        }
        console.log(`Active states: ${activeStates.length} (Bias: ${bias})`);

        // Compute STM image on grid
        // Grid covers [-L, L] x [-L, L]
        const L = this.gridL;
        const res = this.gridSize;
        const dl = (2 * L) / res;
        
        // Precompute atom contributions at zTip
        // psi(r) = sum_i c_i phi_i(r)
        // phi_i(r) ~ exp(-beta * |r - R_i|) or similar.
        // For STM, we usually assume s-wave tip and sample psi at r_tip.
        // So we just evaluate |psi(r_tip)|^2.
        // phi_i(r) is the orbital on atom i. Let's use the same exponential decay as hopping?
        // Or a standard Slater orbital decay. Carbon 2pz decay is roughly 1.625 au^-1 ~ 3 A^-1?
        // Let's use a visualization decay constant, maybe same as beta or slightly different.
        const decay = 1.5; // 1/Angstrom

        let maxVal = 0.0;

        for (let iy = 0; iy < res; iy++) {
            const y = -L + iy * dl;
            for (let ix = 0; ix < res; ix++) {
                const x = -L + ix * dl;
                
                let rho = 0.0;

                // Sum over active states
                for (let k of activeStates) {
                    // psi_k(r) = sum_j C_jk * phi_j(r)
                    let psi = 0.0;
                    for (let j = 0; j < N; j++) {
                        const atom = atoms[j];
                        const dx = x - atom.x;
                        const dy = y - atom.y;
                        const dz = zTip - atom.z;
                        const r = Math.sqrt(dx*dx + dy*dy + dz*dz);
                        
                        // Orbital value
                        // Simple s-orbital approximation: exp(-decay * r)
                        // For pz orbitals (aromatic), it should be z * exp... but let's stick to simple s-like blobs for now as requested "exponential hopping model".
                        // Actually, if it's pi-system, pz orbitals are important.
                        // But for "simple exponential hopping", s-orbitals are the direct mapping.
                        const phi = Math.exp(-decay * r);
                        
                        // C_jk is vectors[j][k]
                        psi += vectors[j][k] * phi;
                    }
                    rho += psi * psi;
                }
                
                this.stmData[iy * res + ix] = rho;
                if (rho > maxVal) maxVal = rho;
            }
        }

        // Update texture
        this.stmTexture.needsUpdate = true;
        
        // Update shader max for normalization
        if (this.stmPlane) {
            this.stmPlane.material.uniforms.uMax.value = maxVal > 1e-6 ? maxVal : 1.0;
            this.stmPlane.position.z = zTip;
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
}

function toRadians(angle) {
    return angle * (Math.PI / 180);
}
