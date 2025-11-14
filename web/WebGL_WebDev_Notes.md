# WebGL / Web Dev Notes for SimpleSimulationEngine `js/`

## 1. How to run these demos

- **Always serve over HTTP, not `file://`**
  - In `~/git/SimpleSimulationEngine/js` run:
    ```bash
    python3 -m http.server 8000
    ```
  - This serves the `js/` directory at `http://127.0.0.1:8000/`.

- **In Windsurf / IDE**
  - Open the Browser Preview that points to the proxy URL (e.g. `http://127.0.0.1:40409`).
  - Navigate to the demos by path:
    - GLSL solid modeling:
      - `GLSL_solid_modeling/index.html`
      - `GLSL_solid_modeling/ListOfPrimitives.html`
    - OBJ loader:
      - `OBJ_Loader/armor.html`
    - Planet designer:
      - `PlanetDesigner/PlanetDesigner.html`
      - `PlanetDesigner/TurbulenceDesigner.html`
    - Shader debugger:
      - `ShaderDebug/ShaderViewer.html`

## 2. Why `file://` didn’t work

- Browsers apply stricter security rules for `file://` URLs.
- Loading local resources like `.glslf` or `.obj` via `file://` often fails or gives strange behavior.
- Some hacks using `<object data="*.glslf">` used to work in some browsers, but now:
  - The browser often **offers the file as a download**.
  - `element.contentDocument` is usually `undefined`, so JS code that tries to read
    `contentDocument.body.childNodes[0].textContent` throws errors.

Serving over HTTP avoids these issues and matches how real web apps run.

## 3. Robust way to load GLSL / text assets

Instead of `<object>` tags, use **`fetch()` + JS strings**:

```js
var shaderSrc = "";

fetch("MyShader.glslf")
    .then(response => response.text())
    .then(text => { shaderSrc = text; });
```

Then build the full fragment shader string in JS and pass it to Three.js:

```js
var material = new THREE.ShaderMaterial({
    uniforms: uniforms,
    vertexShader: basicShader.vertexShader,
    fragmentShader: shaderSrc,
});
```

This is what we now do in:

- `ShaderDebug/ShaderViewer.html` (loads `Vorticity.glslf`, `Harmonics.glslf`).
- `GLSL_solid_modeling/GLSLscreen.js` (loads `Primitives.glslf`, `RayTracer.glslf`).
- `PlanetDesigner/PlanetDesigner.html` (loads `Planet.glslf`).
- `PlanetDesigner/TurbulenceDesigner.html` (loads `WrapFun3.glslf`).

## 4. Avoiding browser download prompts for `.glslf`

- Old approach: `<object data="*.glslf" id="...">` in HTML.
  - Modern browsers treat these as unknown/foreign types → show a **download** dialog.
  - `contentDocument` is not available.
- New approach: comment out those `<object>`s and load the text with `fetch()`.

Result: no more download prompts; GLSL is just text handled by JS.

## 5. Dealing with caching / "hard reload"

Browsers cache JS aggressively, so after changing `.js` files:

- Use a **hard reload** in the browser:
  - Linux/Windows: `Ctrl+Shift+R` or `Ctrl+F5`.
- Or add a **cache-busting query parameter** to the script URL:

```html
<script src="GLSLscreen.js?v=2"></script>
```

- Every time you change `GLSLscreen.js`, bump `v=2` to `v=3`, etc.
- The browser sees a different URL and must download the new file.

We already use this for:

- `GLSL_solid_modeling/index.html`
- `GLSL_solid_modeling/ListOfPrimitives.html`

You can do the same for other pages if needed.

## 6. Debugging workflow (JavaScript + GLSL)

- **Open DevTools → Console** on the page you’re testing.
- Look for:
  - JS errors: `Uncaught TypeError`, `ReferenceError`, etc.
  - Three.js/WebGL errors: `THREE.WebGLShader: Shader Error`, GLSL compile errors.
  - Network errors: `404` for `.glslf`, `.obj`, `.js`.

Typical issues and fixes:

- `Cannot read properties of undefined (reading 'textContent')`:
  - Caused by trying to use `element.contentDocument` on an `<object>` that doesn’t have a DOM.
  - Fix: switch to `fetch()` for text assets.

- Blank viewport, no errors:
  - Check that `init_GLSLScreen(...)` is actually called.
  - Add `console.log(...)` in init functions.

- Shader compile errors:
  - Inspect the combined fragment shader string (e.g. log it to the console) to see what GLSL the GPU actually sees.

## 7. Minimal recipe to add a new WebGL demo

1. Put files in a subfolder of `js/`, e.g. `js/MyDemo/`.
2. Use an HTML file that:
   - Includes Three.js from CDN.
   - Includes your `GLSLscreen.js` or other helper.
   - Optionally loads GLSL via `fetch()` as text.
3. Start the server in `js/`:
   ```bash
   cd ~/git/SimpleSimulationEngine/js
   python3 -m http.server 8000
   ```
4. In the IDE/browser preview, open:
   ```
   http://127.0.0.1:8000/MyDemo/MyDemo.html
   ```
5. Use DevTools console for debugging.

## 8. Summary of what was fixed

- **ShaderDebug/ShaderViewer.html**
  - Loads `Vorticity.glslf` and `Harmonics.glslf` via `fetch()`.
  - No more `<object>` + `contentDocument`.

- **GLSL_solid_modeling**
  - `GLSLscreen.js` loads `Primitives.glslf` and `RayTracer.glslf` via `fetch()`.
  - `index.html` and `ListOfPrimitives.html` no longer embed GLSL via `<object>`.
  - Cache busting added to `GLSLscreen.js` include.

- **PlanetDesigner**
  - `PlanetDesigner.html` loads `Planet.glslf` via `fetch()` and substitutes `TURBULENCE_FUNCTION`.
  - `TurbulenceDesigner.html` loads `WrapFun3.glslf` via `fetch()` and substitutes `__NWRAPS__` and `__VIEW_FUNC__`.

These patterns should be reusable for any future WebGL/Three.js experiments in this repo.
