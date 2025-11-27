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

        const V0 = params.V0;
        const beta = params.beta;
        const E0 = params.E0;
        const d0 = 1.4;

        for (let i = 0; i < this.N; i++) {
            this.H[i][i] = E0;
            for (let j = i + 1; j < this.N; j++) {
                const dx = atoms[i].x - atoms[j].x;
                const dy = atoms[i].y - atoms[j].y;
                const dz = atoms[i].z - atoms[j].z;
                const r = Math.sqrt(dx * dx + dy * dy + dz * dz);

                // Cutoff for performance/physics
                // Increased cutoff to ensure symmetry for small molecules like PTCDA
                if (r > 12.0) continue;

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
        const result = numeric.eig(H);
        const eigenvalues = result.lambda.x;
        const eigenvectors = result.E.x;

        // Sort eigenvalues
        const indices = new Array(eigenvalues.length);
        for (let i = 0; i < eigenvalues.length; i++) indices[i] = i;
        indices.sort((a, b) => eigenvalues[a] - eigenvalues[b]);

        const sortedEigenvalues = new Float64Array(eigenvalues.length);
        const sortedEigenvectors = []; // Array of arrays (columns)

        for (let i = 0; i < eigenvalues.length; i++) {
            sortedEigenvalues[i] = eigenvalues[indices[i]];
            // Extract column k = indices[i]
            // numeric.js E[row][col]
            const vec = new Float64Array(eigenvalues.length);
            for (let row = 0; row < eigenvalues.length; row++) {
                vec[row] = eigenvectors[row][indices[i]];
            }
            sortedEigenvectors.push(vec);
        }

        return {
            energies: sortedEigenvalues,
            vectors: sortedEigenvectors // [eigenStateIndex][atomIndex] (transposed relative to numeric.js E)
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
        this.quad = null;
        this.stmTexture = null; // Removed, using shader calc
        this.eigenTexture = null; // New: Eigenvectors

        this.params = {
            V0: -2.7,
            beta: 2.0,
            E0: 0.0,
            bias: 1.0,
            zTip: 3.0,
            atomSize: 0.1, // Updated default
            showAtoms: true,
            singleOrbitalMode: true, // Added
            selectedOrbital: 0, // Added, Relative to HOMO
            colorScale: 10000.0 // Updated default
        };

        this.gridSize = 128;
        this.gridL = 20.0; // Angstroms (half-width)
    }

    init() {
        this.initThree();
        this.initUI();
        this.loadMolecule('PTCDA');
    }

    initThree() {
        const viewer = document.getElementById('viewer');
        const width = viewer.clientWidth;
        const height = viewer.clientHeight;

        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1); // NDC camera

        this.renderer = new THREE.WebGLRenderer({ antialias: false });
        this.renderer.setSize(width, height);
        viewer.appendChild(this.renderer.domElement);

        // STM Plane (Full Screen Quad)
        const geometry = new THREE.PlaneGeometry(2, 2);

        // Load shader
        fetch('stm_shader.glslf')
            .then(r => r.text())
            .then(fragShader => {
                const material = new THREE.ShaderMaterial({
                    uniforms: {
                        uEigenvectors: { value: null }, // N x N (RGBA)
                        uEigenvalues: { value: new Float32Array(100) }, // Initialize with dummy data
                        uAtomTexture: { value: null }, // N x 1 (RGBA: x,y,z,0)
                        uAtomParams: { value: null },  // N x 1 (RGBA: decay, 0, 0, 0)
                        uBias: { value: this.params.bias },
                        uZTip: { value: this.params.zTip },
                        uResolution: { value: new THREE.Vector2(width, height) },
                        uL: { value: this.gridL },
                        uShowAtoms: { value: this.params.showAtoms },
                        uAtomSize: { value: this.params.atomSize },
                        uAtomCount: { value: 0 },
                        uSingleOrbitalMode: { value: this.params.singleOrbitalMode }, // Initial value
                        uSelectedOrbital: { value: this.params.selectedOrbital }, // Initial value
                        uColorScale: { value: this.params.colorScale } // Initial value
                    },
                    vertexShader: `
                        varying vec2 vUv;
                        void main() {
                            vUv = uv;
                            gl_Position = vec4(position, 1.0);
                        }
                    `,
                    fragmentShader: fragShader,
                    depthTest: false,
                    depthWrite: false
                });
                this.quad = new THREE.Mesh(geometry, material);
                this.scene.add(this.quad);
                this.render();
            });

        window.addEventListener('resize', () => {
            const w = viewer.clientWidth;
            const h = viewer.clientHeight;
            this.renderer.setSize(w, h);
            if (this.quad) {
                this.quad.material.uniforms.uResolution.value.set(w, h);
            }
            this.render();
        });
    }

    initUI() {
        document.getElementById('btnReload').addEventListener('click', () => this.solvePhysics());

        const inputs = ['inpV0', 'inpBeta', 'inpE0', 'bias', 'zTip', 'sldAtomSize', 'orbitalIndex', 'colorScale'];
        inputs.forEach(id => {
            const el = document.getElementById(id);
            if (!el) {
                console.error(`Element with id '${id}' not found`);
                return;
            }
            el.addEventListener('change', (e) => {
                this.updateParams();
                if (id === 'inpV0' || id === 'inpBeta' || id === 'inpE0') {
                    this.solvePhysics();
                } else if (id === 'bias' || id === 'zTip' || id === 'orbitalIndex' || id === 'singleOrbitalMode' || id === 'colorScale') {
                    this.updateSTM();
                } else if (id === 'sldAtomSize' || id === 'showAtoms') {
                    this.updateVisuals();
                }
                this.updateSummary();
            });

            // Mouse wheel support
            if (el.type === 'number') {
                el.addEventListener('wheel', (e) => {
                    e.preventDefault();
                    const step = parseFloat(el.step) || 0.1;
                    const delta = e.deltaY > 0 ? -step : step;
                    el.value = (parseFloat(el.value) + delta).toFixed(2); // simple fix for float precision
                    el.dispatchEvent(new Event('change'));
                });
            }
        });

        document.getElementById('moleculeSelect').addEventListener('change', (e) => {
            this.loadMolecule(e.target.value);
        });

        document.getElementById('showAtoms').addEventListener('change', (e) => {
            this.params.showAtoms = e.target.checked;
            this.updateVisuals();
        });

        document.getElementById('singleOrbitalMode').addEventListener('change', (e) => {
            this.params.singleOrbitalMode = e.target.checked;
            this.updateSTM();
            this.updateSummary();
        });

        document.getElementById('btnEditGeom').addEventListener('click', () => {
            const el = document.getElementById('txtXYZ');
            el.style.display = el.style.display === 'none' ? 'block' : 'none';
        });

        document.getElementById('btnEditParams').addEventListener('click', () => {
            const el = document.getElementById('txtParams');
            el.style.display = el.style.display === 'none' ? 'block' : 'none';
            if (!el.value) {
                el.value = JSON.stringify(this.params, null, 2);
            }
        });

        document.getElementById('btnApplyAdvanced').addEventListener('click', () => {
            const xyz = document.getElementById('txtXYZ').value;
            if (xyz) {
                this.molecule.loadXYZ(xyz);
            }

            const pStr = document.getElementById('txtParams').value;
            if (pStr) {
                try {
                    const p = JSON.parse(pStr);
                    Object.assign(this.params, p);
                    this.syncUI();
                } catch (e) {
                    alert("Invalid JSON parameters");
                }
            }
            this.solvePhysics();
        });
    }

    syncUI() {
        document.getElementById('inpV0').value = this.params.V0;
        document.getElementById('inpBeta').value = this.params.beta;
        document.getElementById('inpE0').value = this.params.E0;
        document.getElementById('bias').value = this.params.bias;
        document.getElementById('zTip').value = this.params.zTip;
        document.getElementById('sldAtomSize').value = this.params.atomSize;
        document.getElementById('showAtoms').checked = this.params.showAtoms;
        document.getElementById('singleOrbitalMode').checked = this.params.singleOrbitalMode;
        document.getElementById('orbitalIndex').value = this.params.selectedOrbital;
        document.getElementById('colorScale').value = Math.log10(this.params.colorScale);
        this.updateVisuals();
        this.updateSummary();
    }

    updateParams() {
        this.params.V0 = parseFloat(document.getElementById('inpV0').value);
        this.params.beta = parseFloat(document.getElementById('inpBeta').value);
        this.params.E0 = parseFloat(document.getElementById('inpE0').value);
        this.params.bias = parseFloat(document.getElementById('bias').value);
        this.params.zTip = parseFloat(document.getElementById('zTip').value);
        this.params.atomSize = parseFloat(document.getElementById('sldAtomSize').value);
        this.params.singleOrbitalMode = document.getElementById('singleOrbitalMode').checked;
        this.params.selectedOrbital = parseInt(document.getElementById('orbitalIndex').value);
        this.params.colorScale = Math.pow(10, parseFloat(document.getElementById('colorScale').value));
    }

    updateSummary() {
        const s = this.params;
        const summary = `V0: ${s.V0} eV, Beta: ${s.beta} 1/A, E0: ${s.E0} eV
Bias: ${s.bias} V, Z-Tip: ${s.zTip} A
Atom Size: ${s.atomSize}, Color Scale: ${s.colorScale.toExponential(1)}
Mode: ${s.singleOrbitalMode ? 'Single Orbital (Rel. HOMO: ' + s.selectedOrbital + ')' : 'Full STM'}`;
        const el = document.getElementById('simSummary');
        if (el) el.textContent = summary;
    }

    loadMolecule(name) {
        let xyz = "";
        if (name === 'PTCDA') {
            fetch('../../examples/xyz/PTCDA.xyz')
                .then(r => {
                    if (!r.ok) throw new Error("Not found");
                    return r.text();
                })
                .then(text => {
                    this.molecule.loadXYZ(text);
                    document.getElementById('txtXYZ').value = text;
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
        document.getElementById('txtXYZ').value = xyz;
        this.solvePhysics();
    }

    solvePhysics() {
        document.getElementById('loading').style.display = 'block';
        setTimeout(() => {
            console.time("Hamiltonian");
            this.hamiltonian.build(this.params);
            console.timeEnd("Hamiltonian");

            console.time("Diagonalization");
            this.eigenSystem = Solver.solve(this.hamiltonian.H);
            console.timeEnd("Diagonalization");

            // Auto-center Fermi level
            const N = this.eigenSystem.energies.length;
            const nOcc = Math.floor(N / 2);
            if (nOcc > 0 && nOcc < N) {
                const homo = this.eigenSystem.energies[nOcc - 1];
                const lumo = this.eigenSystem.energies[nOcc];
                const ef = (homo + lumo) / 2.0;
                console.log(`Auto-centering Ef. HOMO: ${homo.toFixed(3)}, LUMO: ${lumo.toFixed(3)}, Gap: ${(lumo - homo).toFixed(3)}, Shift: ${-ef.toFixed(3)}`);

                for (let i = 0; i < N; i++) {
                    this.eigenSystem.energies[i] -= ef;
                }
                this.eigenSystem.homoIndex = nOcc - 1; // Store HOMO index
            } else {
                this.eigenSystem.homoIndex = -1; // No HOMO if N is too small
            }

            console.log("Eigenvalues (shifted):", this.eigenSystem.energies);
            this.updateEigenstatesList();

            // Pack eigenvectors into texture
            // Size N x N. 
            // Rows = States (k), Cols = Atoms (j)
            // vectors[k][j]
            // N is already defined above
            // const N = this.eigenSystem.energies.length;

            // RGBA Float Texture: 4 components per pixel
            const data = new Float32Array(N * N * 4);
            for (let k = 0; k < N; k++) {
                for (let j = 0; j < N; j++) {
                    const val = this.eigenSystem.vectors[k][j];
                    const idx = (k * N + j) * 4;
                    // Store s-orbital coefficient in Red (x) for now
                    // Alpha channel proved unreliable for float texture on some platforms/browsers
                    data[idx + 0] = val; // s
                    data[idx + 1] = 0.0; // py
                    data[idx + 2] = 0.0; // pz
                    data[idx + 3] = 1.0; // Alpha (unused/padding)
                }
            }

            if (this.eigenTexture) this.eigenTexture.dispose();
            this.eigenTexture = new THREE.DataTexture(data, N, N, THREE.RGBAFormat, THREE.FloatType);
            this.eigenTexture.magFilter = THREE.NearestFilter;
            this.eigenTexture.minFilter = THREE.NearestFilter;
            this.eigenTexture.needsUpdate = true;

            this.updateSTM();
            document.getElementById('loading').style.display = 'none';
        }, 10);
    }

    updateEigenstatesList() {
        const div = document.getElementById('eigenstates');
        if (!div || !this.eigenSystem) return;

        const energiesArray = Array.from(this.eigenSystem.energies);
        const lines = energiesArray.map((e, i) => {
            const nOcc = Math.floor(energiesArray.length / 2);
            const label = (i === nOcc - 1) ? "HOMO" : (i === nOcc) ? "LUMO" : `    `;
            return `${i.toString().padStart(3)}: ${e.toFixed(4)} ${label}`;
        });
        div.innerText = lines.join('\n');
    }

    updateSTM() {
        if (!this.eigenSystem || !this.quad) return;

        const atoms = this.molecule.atoms;

        // Update Uniforms
        this.quad.material.uniforms.uEigenvectors.value = this.eigenTexture;
        this.quad.material.uniforms.uEigenvalues.value = Float32Array.from(this.eigenSystem.energies);
        this.quad.material.uniforms.uBias.value = this.params.bias;
        this.quad.material.uniforms.uZTip.value = this.params.zTip;
        this.quad.material.uniforms.uSingleOrbitalMode.value = this.params.singleOrbitalMode;

        // Calculate absolute orbital index from relative
        let absOrbital = 0;
        if (this.eigenSystem && this.eigenSystem.homoIndex !== undefined) {
            absOrbital = this.eigenSystem.homoIndex + this.params.selectedOrbital;
            // Clamp to valid range
            if (absOrbital < 0) absOrbital = 0;
            if (absOrbital >= 100) absOrbital = 99;
        }
        this.quad.material.uniforms.uSelectedOrbital.value = absOrbital;

        this.quad.material.uniforms.uColorScale.value = this.params.colorScale;


        // Create Atom Texture (N x 1)
        const N = atoms.length;
        // 4 components per atom: x, y, z, padding
        const atomData = new Float32Array(N * 4);
        atoms.forEach((a, i) => {
            atomData[i * 4 + 0] = a.x;
            atomData[i * 4 + 1] = a.y;
            atomData[i * 4 + 2] = a.z;
            atomData[i * 4 + 3] = 0.0;
        });

        const atomTexture = new THREE.DataTexture(atomData, N, 1, THREE.RGBAFormat, THREE.FloatType);
        atomTexture.magFilter = THREE.NearestFilter;
        atomTexture.minFilter = THREE.NearestFilter;
        atomTexture.needsUpdate = true;

        this.quad.material.uniforms.uAtomTexture.value = atomTexture;

        // Create Atom Params Texture (N x 1)
        // RGBA: decay, 0, 0, 0
        const paramData = new Float32Array(N * 4);
        atoms.forEach((a, i) => {
            // Default decay = 1.5 (or 2.0, adjustable later)
            paramData[i * 4 + 0] = 1.5;
            paramData[i * 4 + 1] = 0.0;
            paramData[i * 4 + 2] = 0.0;
            paramData[i * 4 + 3] = 0.0;
        });
        const paramTexture = new THREE.DataTexture(paramData, N, 1, THREE.RGBAFormat, THREE.FloatType);
        paramTexture.magFilter = THREE.NearestFilter;
        paramTexture.minFilter = THREE.NearestFilter;
        paramTexture.needsUpdate = true;

        this.quad.material.uniforms.uAtomParams.value = paramTexture;
        this.quad.material.uniforms.uAtomCount.value = atoms.length;

        this.render();
    }

    updateVisuals() {
        if (this.quad) {
            this.quad.material.uniforms.uShowAtoms.value = this.params.showAtoms;
            this.quad.material.uniforms.uAtomSize.value = this.params.atomSize;
        }
        this.render();
    }

    render() {
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
}
