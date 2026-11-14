// The page as light: the mid-plane z = 0 of a slab of a scalar wave field, seen the way an eye
// sees light. The only inputs are boundary conditions, on the slab's two faces.
//
// The field is three waves, at the frequencies of visible light at 460 nm (blue), 540 nm (green)
// and 620 nm (red), scaled down so that 550 nm would be 10 Hz: 12.0, 10.2 and 8.9 Hz. Three are
// enough: the eye has three kinds of colour receptor, so three spectral lines can match any colour
// on the page, as a laser projector does. Each pixel's colour is what those frequencies look like
// together: the intensity of each on the mid-plane, weighted by the eye's colour matching
// functions (CIE 1931, analytic fit) and turned into sRGB. Nothing else sets a colour.
//
// The slab: x, y across the screen (periodic, much wider than the screen's text), z from -h to h,
// h = DEPTH wavelengths. Everywhere inside, the wave equation u_tt = c^2 lap u, the same c for
// every frequency (set by the knob: the wavelength at 10 Hz in CSS px). On each face the boundary
// condition sends in a prescribed wave and lets whatever arrives leave (an open face with an
// incoming wave). Both faces send the same wave, so the field is mirror-symmetric about z = 0.
//
// The slab is uniform, so it is solved exactly, one in-plane pattern q at a time, with FFTs instead
// of time steps. Pattern q at frequency w crosses the slab as a plane wave tilted by sin(theta) =
// |q| / k (k = w / c), if |q| < k; finer patterns die out within a wavelength of the face and
// never reach the middle (so the page can only be held to about half a wavelength). For each
// pattern the crossing is a 1D Klein-Gordon problem in z (u_tt = c^2 u_zz - c^2 q^2 u), whose
// response to a wave switched on at a face is known exactly (the signalling solution): nothing
// until h / c, then a ringing front that settles to the steady state. That response, E(|q|, age),
// is tabulated once per layout. In space, it is every point of the faces sending out a spherical
// wavelet: on the screen each change first appears straight across from where it was sent,
// then spreads out in rings, and the rings from everywhere interfere into the page.
//
// A face switched everywhere at once sends its change straight across: the tilted waves' sideways
// parts cancel, and the change appears on the whole screen together, as a fade. Two things in the
// drive make the waves show instead, both still only what the faces send:
//  - The light is slanted. The fit starts from (and keeps) a phase that tilts the light carrying
//    the left half of the page rightward and the right half leftward (sin = TILT). Tilt is only
//    phase, so the standing page is unchanged, but that light travels sideways as it crosses.
//  - A change is sent out of focus first. The faces send it focused DEFOCUS wavelengths from the
//    screen (a phase (kz - k) x distance on each pattern), then pull the focus onto the screen
//    over FOCUS s. Out of focus, the page is its own diffraction pattern: fringes and rings
//    round every edge, and, the light being slanted, each half displaced toward its own side.
//    As the focus comes in, the two halves travel in from left and right and the rings close
//    into the text.
// The response to that moving focus is exact too: the same signalling solution, with the
// drive's changing phase inside the integral (so a fast-moving focus shifts each pattern's
// frequency, as it must), tabulated once.
(() => {
'use strict';

const WAVELENGTH = 5;      // CSS px: the wavelength at 10 Hz (?wavelength= overrides): the knob
// The frequencies: visible light at these wavelengths (nm), scaled so that 550 nm is 10 Hz.
const LAMBDA = [460, 540, 620];
const F_REF = 10, LAMBDA_REF = 550;
const FREQ = LAMBDA.map(l => F_REF * LAMBDA_REF / l);
// Linear sRGB of unit intensity at each frequency (CIE 1931 analytic fit, scaled so all three at
// unit intensity have unit green), and a non-negative mix of frequencies for each sRGB channel
// with SEE * MIX = identity: the page's colours split exactly into the frequencies' intensities.
const SEE = [
  [-0.00643, -0.43319, 1.69758],
  [-0.07621, 1.17375, -0.09754],
  [1.37121, -0.12184, -0.02223],
];
const MIX = [[0.01429, 0.08145, 0.73388], [0.05097, 0.87585, 0.04892], [0.60214, 0.22381, 0.01526], [0, 0, 0]];
const DEPTH = 4;           // the slab's half-thickness h, in wavelengths at 10 Hz
const PPW = 2.5;           // grid points per wavelength of the finest (blue) light
const MAX_TEXELS = 1 << 20;   // grid points at most (else the wavelength is made longer)
const BAND = 0.9;          // patterns fitted up to this fraction of each frequency's wavenumber
const TAPER = 0.75;        // ... rolling off smoothly from this fraction of that
const GS_ITERS = 6;        // phase-fit rounds per page change
const RING = 12;           // grid points around the screen left free by the fit
const RAMP = 0.25;         // s: the faces fade over to a new drive over this
const DEFOCUS = -150;      // wavelengths at 10 Hz: a change is first sent focused this far from the screen
                           // (negative: beyond it, so slanted light starts on its own side) ...
const FOCUS = 3;           // s: ... and the faces pull the focus onto the screen over this
const TILT = 0.25;         // sin of the slant of the light carrying each half of the page, toward the middle
const MAX_SNAP = 4;        // page changes kept in flight at most
const RANKS = 1;           // strips from the edges to the middle, each switched at its own time (1: all at once)
const SWEEP = 0;           // s: from the edges' switch to the middle's
const STEP_GAP = 220;      // ms: at most one drive change per this interval while scrolling
const HT = DEPTH / F_REF;  // s: the head-on crossing time h / c
const ARRIVE = HT + FOCUS + 2;   // s: the response is tabulated up to this age ...
const SETTLE_BLEND = 1;    // s: ... its last slow ringing faded out over this, before the end
const NQ = 128, NA = 512;  // table size: pattern sizes 0..k; ages 0..ARRIVE
const DONE = SWEEP + ARRIVE;   // s: until a change has fully arrived everywhere

const root = document.documentElement;
const canvas = document.getElementById('field');
const doc = document.getElementById('content');
const hudText = document.getElementById('hud-text');
const modeButton = document.getElementById('mode');
const knob = document.getElementById('knob');
const knobInput = knob.querySelector('input'), knobOut = knob.querySelector('output');
const reduce = matchMedia('(prefers-reduced-motion: reduce)');
const params = new URLSearchParams(location.search);

let gl = null;
try {
  gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: false });
} catch (e) { gl = null; }
if (!gl || !gl.getExtension('EXT_color_buffer_float')) return;   // the plain HTML page stays
const LINEAR = !!gl.getExtension('OES_texture_float_linear');     // filtered table lookups

// ---------------------------------------------------------------- shaders
// The grid (F.x x F.y, powers of two, periodic) holds the screen at off and a free ring around it.
// Spectra hold two complex numbers per texel: frequencies 2p and 2p + 1 of part p (the fourth is
// spare).
const HEAD = '#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\nprecision highp sampler2DArray;\nout vec4 o;\n';
const VS = '#version 300 es\nin vec2 p;\nvoid main(){ gl_Position = vec4(p, 0.0, 1.0); }';
const CPLX = `
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
vec2 cis(float a) { return vec2(cos(a), sin(a)); }`;
// |q| (radians per grid point) of spectral index n, and the band limit for the part's two
// frequencies.
const Q = `
uniform ivec2 F;
float qlen(ivec2 n) {
  return length(6.28318531 * vec2(n.x < F.x / 2 ? n.x : n.x - F.x, n.y < F.y / 2 ? n.y : n.y - F.y) / vec2(F));
}
vec2 taper(float q, vec2 kc) {
  vec2 t = clamp((q / kc - ${TAPER}) / (1.0 - ${TAPER}), 0.0, 1.0);
  return 0.5 + 0.5 * cos(3.14159265 * t);
}`;

// One radix-2 Stockham pass on two complex numbers per texel.
const FS_FFT = HEAD + CPLX + `
uniform sampler2D src;
uniform int size, sub, horiz;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int idx = horiz == 1 ? p.x : p.y;
  int hs = sub / 2;
  int ev = (idx / sub) * hs + (idx % hs);
  int od = ev + size / 2;
  vec4 e = texelFetch(src, horiz == 1 ? ivec2(ev, p.y) : ivec2(p.x, ev), 0);
  vec4 q = texelFetch(src, horiz == 1 ? ivec2(od, p.y) : ivec2(p.x, od), 0);
  vec2 w = cis(-6.283185307179586 * float(idx % sub) / float(sub));
  o = e + vec4(cmul(w, q.xy), cmul(w, q.zw));
}`;

// Start of the fit: each frequency's target amplitude (square root of its share of the page's
// light) with the phase the field has now, where it is already showing it.
const FS_GS_INIT = HEAD + CPLX + `
uniform sampler2D T, C;        // the page (linear sRGB); the field now (conjugated), made at cs rows ago
uniform int cs, hasC;
uniform ivec2 F;
uniform vec3 share[2];         // this part's two rows of MIX
uniform vec2 k;                // this part's two wavenumbers, radians per grid point
uniform float xc;              // the screen's middle column
// The phase to start from: the field's own where it already has about the right strength (it
// is showing this page); elsewhere that of light slanted toward the middle, from the left on
// the left half and from the right on the right.
vec2 seed(vec2 z, float a, float tilt) {
  float m = length(z);
  float trust = m > 1e-6 ? 1.0 - smoothstep(0.08, 0.25, abs(m - a)) : 0.0;
  vec2 u = trust * (m > 1e-6 ? z / m : vec2(1.0, 0.0)) + (1.0 - trust + 1e-4) * cis(tilt);
  float l = length(u);
  return l > 1e-6 ? u / l : cis(tilt);
}
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec3 t = texelFetch(T, n, 0).rgb;
  float a0 = sqrt(max(dot(share[0], t), 0.0)), a1 = sqrt(max(dot(share[1], t), 0.0));
  vec4 c = vec4(0.0);
  int y = n.y + cs;
  if (hasC == 1 && y >= 0 && y < F.y) c = texelFetch(C, ivec2(n.x, y), 0);
  vec2 tilt = -${TILT} * k * abs(float(n.x) + 0.5 - xc);
  o = vec4(a0 * seed(vec2(c.x, -c.y), a0, tilt.x), a1 * seed(vec2(c.z, -c.w), a1, tilt.y));
}`;
// Keep only the patterns that can cross, conjugated and scaled for the inverse transform.
const FS_GS_BAND = HEAD + Q + `
uniform sampler2D X;
uniform vec2 kc;
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec2 w = taper(qlen(n), kc) / float(F.x * F.y);
  vec4 x = texelFetch(X, n, 0) * vec4(w.x, w.x, w.y, w.y);
  o = vec4(x.x, -x.y, x.z, -x.w);
}`;
// Keep that field's phase; put the target amplitude back on the screen, leave the ring free and
// keep the rest dark.
const FS_GS_PHASE = HEAD + `
uniform sampler2D Y, T;
uniform vec3 share[2];
uniform ivec4 win;             // the screen: x, y, w, h
uniform int ring;
vec2 unit(vec2 z) { float m = length(z); return m > 1e-20 ? z / m : vec2(1.0, 0.0); }
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec4 y = texelFetch(Y, n, 0);
  vec2 z0 = vec2(y.x, -y.y), z1 = vec2(y.z, -y.w);
  ivec2 r = max(win.xy - n, n - (win.xy + win.zw - 1));   // > 0 outside the screen
  int out_ = max(r.x, r.y);
  if (out_ > ring) { o = vec4(0.0); return; }
  if (out_ > 0) { o = vec4(z0, z1); return; }
  vec3 t = texelFetch(T, n, 0).rgb;
  float a0 = sqrt(max(dot(share[0], t), 0.0)), a1 = sqrt(max(dot(share[1], t), 0.0));
  o = vec4(a0 * unit(z0), a1 * unit(z1));
}`;
// The fitted field's spectrum, band-limited and normalised: what the faces deliver to the middle.
const FS_STORE = HEAD + Q + `
uniform sampler2D X;
uniform vec2 kc;
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec2 w = taper(qlen(n), kc) / float(F.x * F.y);
  o = texelFetch(X, n, 0) * vec4(w.x, w.x, w.y, w.y);
}`;

// The middle now, pattern by pattern: each kept change times the response to it so far (less
// the response to the next change, which replaces it), shifted to where the page is now.
// Conjugated for the inverse transform.
const FS_COMBINE = HEAD + CPLX + Q + `
uniform sampler2DArray S;
uniform sampler2D E;           // the response E(|q| / k, age), the part's two frequencies
uniform int count, fromDark, part;
uniform int slot[${MAX_SNAP}], sh[${MAX_SNAP}];
uniform float age[${MAX_SNAP}];
uniform vec2 k;                // the part's two wavenumbers, radians per grid point
vec4 at(vec2 qk, float a) {    // the table, bilinear
  float y = a / ${ARRIVE.toFixed(3)} * ${NA - 1}.0;
  vec2 x = clamp(qk, 0.0, 1.0) * ${NQ - 1}.0;
  ${LINEAR ? `
  vec2 r = texture(E, vec2((x.x + 0.5) / ${NQ}.0, (y + 0.5) / ${NA}.0)).xy;
  vec2 g = texture(E, vec2((x.y + 0.5) / ${NQ}.0, (y + 0.5) / ${NA}.0)).zw;
  return vec4(r, g);` : `
  int y0 = int(floor(y)); float fy = y - float(y0);
  ivec2 x0 = ivec2(floor(x)); vec2 fx = x - vec2(x0);
  ivec2 x1 = min(x0 + 1, ${NQ - 1}); int y1 = min(y0 + 1, ${NA - 1});
  vec4 r0 = mix(texelFetch(E, ivec2(x0.x, y0), 0), texelFetch(E, ivec2(x1.x, y0), 0), fx.x);
  vec4 r1 = mix(texelFetch(E, ivec2(x0.x, y1), 0), texelFetch(E, ivec2(x1.x, y1), 0), fx.x);
  vec4 g0 = mix(texelFetch(E, ivec2(x0.y, y0), 0), texelFetch(E, ivec2(x1.y, y0), 0), fx.y);
  vec4 g1 = mix(texelFetch(E, ivec2(x0.y, y1), 0), texelFetch(E, ivec2(x1.y, y1), 0), fx.y);
  return vec4(mix(r0, r1, fy).xy, mix(g0, g1, fy).zw);`}
}
vec4 resp(vec2 qk, float a) {  // 0 before the front, 1 once settled
  if (a >= ${ARRIVE.toFixed(3)}) return vec4(1.0, 0.0, 1.0, 0.0);
  if (a <= ${HT.toFixed(3)}) return vec4(0.0);
  return at(qk, a);
}
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec2 qk = qlen(n) / k;
  vec4 sum = vec4(0.0);
  for (int j = 0; j < ${MAX_SNAP}; j++) {
    if (j >= count) break;
    int m = ((n.y * sh[j]) % F.y + F.y) % F.y;
    vec2 e = cis(6.28318531 * float(m) / float(F.y));
    for (int r = 0; r < ${RANKS}; r++) {
      float delay = ${RANKS > 1 ? `${SWEEP.toFixed(3)} * float(r) / ${RANKS - 1}.0` : '0.0'};
      vec4 w = j == 0 && fromDark == 0 ? vec4(1.0, 0.0, 1.0, 0.0) : resp(qk, age[j] - delay);
      if (j + 1 < count) w -= resp(qk, age[j + 1] - delay);
      if (w == vec4(0.0)) continue;
      vec4 d = texelFetch(S, ivec3(n, (slot[j] * ${RANKS} + r) * 2 + part), 0);
      sum += vec4(cmul(w.xy, cmul(d.xy, e)), cmul(w.zw, cmul(d.zw, e)));
    }
  }
  o = vec4(sum.x, -sum.y, sum.z, -sum.w);
}`;
// One strip of the fitted field: by distance from the screen's left and right edges (0 at the
// edges, RANKS at the middle), soft hats that add up to one.
const FS_MASK = HEAD + `
uniform sampler2D X;
uniform int r, ox, w;
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  float x = float(n.x - ox) + 0.5;
  float d = min(x, float(w) - x) / (0.5 * float(w)) * ${RANKS}.0 - 0.5;
  float m = max(0.0, 1.0 - abs(clamp(d, 0.0, ${RANKS - 1}.0) - float(r)));
  o = texelFetch(X, n, 0) * m;
}`;
const FS_COPY = HEAD + `
uniform sampler2D X;
void main() { o = texelFetch(X, ivec2(gl_FragCoord.xy), 0); }`;

// What the eye sees: each frequency's intensity, through the colour matching functions, as
// linear sRGB.
const FS_SEE = HEAD + `
uniform sampler2D C0, C1;
uniform mat3 see;
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec4 a = texelFetch(C0, n, 0), b = texelFetch(C1, n, 0);
  vec3 I = vec3(dot(a.xy, a.xy), dot(a.zw, a.zw), dot(b.xy, b.xy));
  o = vec4(see * I, 0.0);
}`;

// The screen: the colour between grid points reconstructed with a Lanczos-3 kernel, then sRGB.
const FS_DISPLAY = HEAD + `
uniform sampler2D E0;          // linear sRGB on the grid
uniform ivec2 F, off;
uniform vec2 res, cells;       // canvas px; grid points across the screen
uniform float fy;              // fraction of a grid row the page is past the field
vec3 E(int x, int y) { return texelFetch(E0, clamp(ivec2(x, y), ivec2(0), F - 1), 0).rgb; }
float lz(float x) {
  x = abs(x);
  if (x < 1e-4) return 1.0;
  if (x >= 3.0) return 0.0;
  float p = 3.14159265 * x;
  return 3.0 * sin(p) * sin(p / 3.0) / (p * p);
}
void main() {
  vec2 g = vec2(gl_FragCoord.x / res.x * cells.x, (1.0 - gl_FragCoord.y / res.y) * cells.y + fy) + vec2(off) - 0.5;
  ivec2 i = ivec2(floor(g)); vec2 f = fract(g);
  float wx[6], wy[6], sx = 0.0, sy = 0.0;
  for (int k = 0; k < 6; k++) {
    wx[k] = lz(f.x - float(k - 2)); sx += wx[k];
    wy[k] = lz(f.y - float(k - 2)); sy += wy[k];
  }
  vec3 e = vec3(0.0);
  for (int b = 0; b < 6; b++) {
    vec3 r = vec3(0.0);
    for (int a = 0; a < 6; a++) r += wx[a] * E(i.x + a - 2, i.y + b - 2);
    e += wy[b] * r;
  }
  e = clamp(e / (sx * sy), 0.0, 1.0);
  o = vec4(mix(12.92 * e, 1.055 * pow(e, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, e)), 1.0);
}`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
let P;
try {
  const vs = compile(gl.VERTEX_SHADER, VS);
  const program = fs => {
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, 'p');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    const count = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const name = gl.getActiveUniform(p, i).name;
      u[name] = gl.getUniformLocation(p, name);
      if (name.endsWith('[0]')) u[name.slice(0, -3)] = u[name];
    }
    return { p, u: new Proxy(u, { get: (o, k) => o[k] ?? null }) };
  };
  P = {
    fft: program(FS_FFT), gsInit: program(FS_GS_INIT), gsBand: program(FS_GS_BAND), gsPhase: program(FS_GS_PHASE),
    store: program(FS_STORE), combine: program(FS_COMBINE), copy: program(FS_COPY), mask: program(FS_MASK),
    see: program(FS_SEE), disp: program(FS_DISPLAY),
  };
} catch (e) {
  console.warn('waves: falling back to the plain page', e);
  return;
}

const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

function nearest(target) {
  gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}
function complete() { return gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE; }
// A 2D float render target; half precision where allowed and asked for.
function target(w, h, half) {
  const formats = half ? [[gl.RGBA16F, gl.HALF_FLOAT], [gl.RGBA32F, gl.FLOAT]] : [[gl.RGBA32F, gl.FLOAT]];
  for (const [ifmt, type] of formats) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    nearest(gl.TEXTURE_2D);
    gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, gl.RGBA, type, null);
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    if (complete()) return { t, f };
    gl.deleteFramebuffer(f); gl.deleteTexture(t);
  }
  throw new Error('no float render target');
}
// A stack of float layers (half precision unless `full`), each renderable.
function layers(w, h, n, full) {
  for (const [ifmt, type] of full ? [[gl.RGBA32F, gl.FLOAT]] : [[gl.RGBA16F, gl.HALF_FLOAT], [gl.RGBA32F, gl.FLOAT]]) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
    nearest(gl.TEXTURE_2D_ARRAY);
    gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, ifmt, w, h, n, 0, gl.RGBA, type, null);
    const fbos = [];
    let ok = true;
    for (let i = 0; i < n && ok; i++) {
      const f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, t, 0, i);
      ok = complete();
      fbos.push(f);
    }
    if (ok) return { t, fbos };
    fbos.forEach(f => gl.deleteFramebuffer(f)); gl.deleteTexture(t);
  }
  throw new Error('no float layer target');
}
function run(prog, fbo, w, h, tex, set) {
  gl.useProgram(prog.p);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.viewport(0, 0, w, h);
  let unit = 0;
  for (const k in tex) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    const [kind, t] = Array.isArray(tex[k]) ? tex[k] : [gl.TEXTURE_2D, tex[k]];
    gl.bindTexture(kind, t);
    gl.uniform1i(prog.u[k], unit++);
  }
  if (set) set(prog.u);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}



// ---------------------------------------------------------------- state
let Lx = 0, Ly = 0;            // CSS px
let W = 0, H = 0, FX = 0, FY = 0, OX = 0, OY = 0;   // grid points: screen; grid; screen offset
let d = 1;                     // grid points per CSS px
let wavelength = WAVELENGTH;
const phys = { wl: 0, k: [0, 0, 0, 0] };
let R = null;
let t = 0;                     // s of time
let snaps = [];                // page changes in flight, oldest first: { slot, t0, row }
let fromDark = true;           // the oldest change came in over darkness
let curRow = 0, hasCur = false;   // the grid row the field now (R.cur) was made at
let raf = 0, moved = false, lastStep = -1e9, colors = null;
const SEE_MAT = new Float32Array([SEE[0][0], SEE[1][0], SEE[2][0], SEE[0][1], SEE[1][1], SEE[2][1], SEE[0][2], SEE[1][2], SEE[2][2]]);
const pow2AtLeast = n => 1 << Math.ceil(Math.log2(Math.max(2, n)));

function freeResources() {
  if (!R) return;
  for (const r of [...R.pair, R.keep, R.page, ...R.cur, R.see]) { gl.deleteTexture(r.t); gl.deleteFramebuffer(r.f); }
  gl.deleteTexture(R.S.t); R.S.fbos.forEach(f => gl.deleteFramebuffer(f));
  R.E.forEach(e => gl.deleteTexture(e));
  R = null;
}

// The exact response of the middle to a wave switched on (over RAMP) at the faces, for each
// frequency, pattern size |q| = k x (x = 0..1, NQ samples) and age (0..ARRIVE, NA samples),
// relative to its steady state: the Klein-Gordon signalling solution
//   u(t) = f(t - T) - q h int_T^t J1(c q sqrt(tau^2 - T^2)) / sqrt(tau^2 - T^2) f(t - tau) dtau,
// T = h / c, f = e^{-i w t} S(t), divided by the steady state e^{-i w t} e^{i h sqrt(k^2 - q^2)}.
function besselJ1(x) {
  const ax = Math.abs(x);
  if (ax < 8) {
    const y = x * x;
    return x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606))))) /
      (144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y)))));
  }
  const z = 8 / ax, y = z * z, xx = ax - 2.356194491;
  const p1 = 1 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
  const p2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
  return Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * p1 - z * Math.sin(xx) * p2) * Math.sign(x);
}
function computeResponse() {
  const ramp = x => x <= 0 ? 0 : x >= RAMP ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * x / RAMP);
  const DT = 1 / 300;                                  // s: integration step
  const NT = Math.ceil((ARRIVE - HT) / DT) + 1;
  const NM = Math.ceil(Math.max(FOCUS, RAMP) / DT) + 1;   // the drive is steady after this many steps
  const tables = [new Float32Array(NQ * NA * 4), new Float32Array(NQ * NA * 4)];
  const pr = new Float64Array(NT), pi = new Float64Array(NT), cr = new Float64Array(NT + 1), ci = new Float64Array(NT + 1);
  const mr = new Float64Array(NM + 1), mi = new Float64Array(NM + 1);
  for (let b = 0; b < 3; b++) {
    const w = 2 * Math.PI * FREQ[b], cT = HT;          // units: time in s, lengths in c * s
    const k = w;                                       // k = w / c with c = 1
    const h = HT;                                      // h = c T
    const tab = tables[b >> 1], ch = (b & 1) * 2;
    for (let iq = 0; iq < NQ; iq++) {
      const q = k * iq / (NQ - 1), m = q;              // m = c q
      const kz = Math.sqrt(Math.max(k * k - q * q, 0));
      for (let i = 0; i < NT; i++) {                   // the kernel times e^{i w tau}, tau = T + i DT
        const tau = cT + i * DT, s2 = Math.sqrt(Math.max(tau * tau - cT * cT, 0));
        const kv = q * h * (s2 > 1e-9 ? besselJ1(m * s2) / s2 : m / 2) * DT * (i === 0 ? 0.5 : 1);
        pr[i] = kv * Math.cos(w * tau); pi[i] = kv * Math.sin(w * tau);
        cr[i + 1] = cr[i] + pr[i]; ci[i + 1] = ci[i] + pi[i];
      }
      // What the faces send for this pattern, relative to the steady drive: switched on over
      // RAMP, focused DEFOCUS wavelengths from the screen at first (a phase (kz - k) x
      // distance on each pattern), the focus pulled in over FOCUS, slowing to a stop.
      const dphi = (kz - k) * DEFOCUS / F_REF;
      for (let i = 0; i <= NM; i++) {
        const sT = i * DT, u = Math.min(1, sT / FOCUS), ph = dphi * (1 - u) * (1 - u), am = ramp(sT);
        mr[i] = am * Math.cos(ph); mi[i] = am * Math.sin(ph);
      }
      const nr = Math.cos(-h * kz), ni = Math.sin(-h * kz);   // divide by the steady state
      for (let ia = 0; ia < NA; ia++) {
        const a = ARRIVE * ia / (NA - 1);
        let er = 0, ei = 0;
        if (a > cT) {
          const last = Math.min(NT - 1, Math.floor((a - cT) / DT));
          const l0 = Math.min(last, NM);
          const c0 = Math.cos(w * cT), s0 = Math.sin(w * cT);
          er = c0 * mr[l0] - s0 * mi[l0]; ei = c0 * mi[l0] + s0 * mr[l0];
          const full = last - NM;                       // the drive was steady up to here
          if (full >= 0) { er -= cr[full + 1]; ei -= ci[full + 1]; }
          for (let i = Math.max(0, full + 1); i <= last; i++) {
            const j = last - i;
            er -= pr[i] * mr[j] - pi[i] * mi[j]; ei -= pr[i] * mi[j] + pi[i] * mr[j];
          }
        }
        let rr = er * nr - ei * ni, ri = er * ni + ei * nr;
        const fade = Math.min(1, Math.max(0, (a - (ARRIVE - SETTLE_BLEND)) / SETTLE_BLEND));
        const f = fade * fade * (3 - 2 * fade);
        rr += (1 - rr) * f; ri += (0 - ri) * f;
        const o = (ia * NQ + iq) * 4 + ch;
        tab[o] = rr; tab[o + 1] = ri;
      }
    }
  }
  return tables;
}
let tableData = null;          // the tables depend on nothing in the layout: computed once
function responseTables() {
  if (!tableData) tableData = computeResponse();
  const tables = tableData;
  return tables.map(data => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    nearest(gl.TEXTURE_2D);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, NQ, NA, 0, gl.RGBA, gl.FLOAT, data);
    if (LINEAR) {
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    }
    return t;
  });
}

function allocate() {
  freeResources();
  if (Math.max(FX, FY) > gl.getParameter(gl.MAX_TEXTURE_SIZE)) throw new Error('grid too large');
  R = {
    pair: [target(FX, FY), target(FX, FY)],      // transforms and fitting
    keep: target(FX, FY),                        // the fitted field, while it is split into strips
    page: target(FX, FY, true),                  // the page (linear sRGB)
    cur: [target(FX, FY, true), target(FX, FY, true)],   // the field in the middle now (conjugated)
    see: target(FX, FY, true),                   // what it looks like (linear sRGB)
    S: layers(FX, FY, 2 * RANKS * MAX_SNAP),     // the changes in flight: slot s, strip r, part p
    E: responseTables(),                         // the response of each pattern to a change
  };
  for (const r of R.cur) { gl.bindFramebuffer(gl.FRAMEBUFFER, r.f); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
  snaps = []; fromDark = true; hasCur = false; t = 0;
}

// ---------------------------------------------------------------- page -> drive
const rc = document.createElement('canvas');
const ctx = rc.getContext('2d', { willReadFrequently: true });
const range = document.createRange();
let pageBuf = null;
const LIN = Float32Array.from({ length: 256 }, (_, i) => {
  const c = i / 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
});

// Draw the page's text in the colours it would have (its own text is transparent) into c,
// whose transform maps CSS px to the canvas, in the page's current scroll position.
function paintText(c, h) {
  c.textBaseline = 'middle';
  const walker = document.createTreeWalker(doc, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (!/\S/.test(node.data)) continue;
    const el = node.parentElement;
    const box = el.getBoundingClientRect();
    if (box.bottom < -40 || box.top > h + 40) continue;
    const cs = getComputedStyle(el);
    c.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    const link = !!el.closest('a');
    c.fillStyle = link ? colors.negCss : el.closest('.muted') ? colors.mutedCss : colors.fgCss;
    const words = /\S+/g;
    for (let m = words.exec(node.data); m; m = words.exec(node.data)) {
      range.setStart(node, m.index);
      range.setEnd(node, m.index + m[0].length);
      const rects = range.getClientRects();
      if (rects.length === 1) { stamp(c, m[0], rects[0], link); continue; }
      for (let i = 0; i < m[0].length; i++) {    // word broken across lines: letter by letter
        range.setStart(node, m.index + i);
        range.setEnd(node, m.index + i + 1);
        stamp(c, m[0][i], range.getClientRects()[0], link);
      }
    }
  }
}
function stamp(c, text, r, link) {
  if (!r || r.width === 0) return;
  c.fillText(text, r.left, r.top + r.height / 2, r.width + 1);
  if (link) c.fillRect(r.left, r.bottom - 1, r.width, 1.4);
}

// The page scrolled to grid row `row`, linear sRGB, on the grid: the screen at (OX, OY), dark around.
function capturePage(row) {
  rc.width = FX; rc.height = FY;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, FX, FY);
  ctx.fillStyle = colors.bgCss;
  ctx.fillRect(OX, OY, W, H);
  ctx.save();
  ctx.beginPath(); ctx.rect(OX, OY, W, H); ctx.clip();
  ctx.setTransform(d, 0, 0, d, OX, OY + (scrollY - row / d) * d);
  paintText(ctx, Ly);
  ctx.restore();
  const img = ctx.getImageData(0, 0, FX, FY).data;
  if (!pageBuf || pageBuf.length !== FX * FY * 4) pageBuf = new Float32Array(FX * FY * 4);
  for (let i = 0, n = FX * FY * 4; i < n; i++) pageBuf[i] = LIN[img[i]];
  gl.bindTexture(gl.TEXTURE_2D, R.page.t);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, FX, FY, gl.RGBA, gl.FLOAT, pageBuf);
}

// Forward FFT in place over the ping-pong pair, starting from pair[a]; the index of the result.
function fft(a) {
  const pair = R.pair;
  for (const [horiz, size] of [[1, FX], [0, FY]]) {
    for (let sub = 2; sub <= size; sub *= 2) {
      run(P.fft, pair[1 - a].f, FX, FY, { src: pair[a].t }, u => {
        gl.uniform1i(u.size, size); gl.uniform1i(u.sub, sub); gl.uniform1i(u.horiz, horiz);
      });
      a = 1 - a;
    }
  }
  return a;
}
const kcOf = part => [BAND * phys.k[2 * part], BAND * (phys.k[2 * part + 1] || phys.k[2])];

// Fit the page (captured at `row`) and store what the faces must deliver, into slot `slot`.
function fitDrive(slot, row) {
  const pair = R.pair, T = R.page.t;
  for (let part = 0; part < 2; part++) {
    const share = new Float32Array([...MIX[2 * part], ...MIX[2 * part + 1]]);
    const kc = kcOf(part);
    run(P.gsInit, pair[0].f, FX, FY, { T, C: R.cur[part].t }, u => {
      gl.uniform2i(u.F, FX, FY); gl.uniform3fv(u.share, share);
      gl.uniform2f(u.k, phys.k[2 * part], phys.k[2 * part + 1] || phys.k[2]); gl.uniform1f(u.xc, OX + W / 2);
      gl.uniform1i(u.cs, row - curRow); gl.uniform1i(u.hasC, hasCur ? 1 : 0);
    });
    let a = 0;
    for (let it = 0; it < GS_ITERS; it++) {
      a = fft(a);
      run(P.gsBand, pair[1 - a].f, FX, FY, { X: pair[a].t }, u => { gl.uniform2i(u.F, FX, FY); gl.uniform2fv(u.kc, kc); });
      a = fft(1 - a);
      run(P.gsPhase, pair[1 - a].f, FX, FY, { Y: pair[a].t, T }, u => {
        gl.uniform3fv(u.share, share); gl.uniform4i(u.win, OX, OY, W, H); gl.uniform1i(u.ring, RING);
      });
      a = 1 - a;
    }
    run(P.copy, R.keep.f, FX, FY, { X: pair[a].t });
    for (let r = 0; r < RANKS; r++) {
      run(P.mask, pair[0].f, FX, FY, { X: R.keep.t }, u => { gl.uniform1i(u.r, r); gl.uniform1i(u.ox, OX); gl.uniform1i(u.w, W); });
      const b = fft(0);
      run(P.store, R.S.fbos[(slot * RANKS + r) * 2 + part], FX, FY, { X: pair[b].t }, u => { gl.uniform2i(u.F, FX, FY); gl.uniform2fv(u.kc, kc); });
    }
  }
}

const rowNow = () => Math.floor(scrollY * d);
const room = () => FY - H - RING;     // rows a change can scroll before it would wrap round
const dropOldest = () => { snaps.shift(); fromDark = false; };

// The faces switch to the page as it is now.
function changeDrive(now) {
  const row = rowNow();
  if (snaps.length === MAX_SNAP) dropOldest();
  const used = new Set(snaps.map(s => s.slot));
  let slot = 0;
  while (used.has(slot)) slot++;
  capturePage(row);
  fitDrive(slot, row);
  snaps.push({ slot, t0: t, row });
  lastStep = now;
}

// Forget changes that have fully arrived (or scrolled too far to keep).
function prune() {
  const row = rowNow();
  while (snaps.length > 1 && (t - snaps[1].t0 > DONE || Math.abs(row - snaps[0].row) > room())) dropOldest();
}

// The field in the middle now, and what it looks like.
function render() {
  const row = rowNow();
  const n = snaps.length;
  const slot = new Int32Array(MAX_SNAP), sh = new Int32Array(MAX_SNAP), age = new Float32Array(MAX_SNAP);
  snaps.forEach((s, j) => { age[j] = t - s.t0; sh[j] = row - s.row; slot[j] = s.slot; });
  for (let part = 0; part < 2; part++) {
    run(P.combine, R.pair[0].f, FX, FY, { S: [gl.TEXTURE_2D_ARRAY, R.S.t], E: R.E[part] }, u => {
      gl.uniform2i(u.F, FX, FY);
      gl.uniform1i(u.count, n); gl.uniform1i(u.fromDark, fromDark ? 1 : 0); gl.uniform1i(u.part, part);
      gl.uniform1iv(u.slot, slot); gl.uniform1iv(u.sh, sh); gl.uniform1fv(u.age, age);
      gl.uniform2f(u.k, phys.k[2 * part], phys.k[2 * part + 1] || phys.k[2]);
    });
    const a = fft(0);
    run(P.copy, R.cur[part].f, FX, FY, { X: R.pair[a].t });
  }
  curRow = row; hasCur = true;
  run(P.see, R.see.f, FX, FY, { C0: R.cur[0].t, C1: R.cur[1].t }, u => gl.uniformMatrix3fv(u.see, false, SEE_MAT));
}

function hex(v) {
  const m = /#([0-9a-f]{6})/i.exec(v) || ['', '000000'];
  const n = parseInt(m[1], 16);
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
}
function readColors() {
  const cs = getComputedStyle(root);
  const v = name => cs.getPropertyValue(name).trim();
  colors = { bg: hex(v('--bg')), bgCss: v('--bg'), fgCss: v('--fg'), mutedCss: v('--muted'), negCss: v('--neg') };
}

function draw() {
  run(P.disp, null, canvas.width, canvas.height, { E0: R.see.t }, u => {
    gl.uniform2i(u.F, FX, FY); gl.uniform2i(u.off, OX, OY);
    gl.uniform2f(u.res, canvas.width, canvas.height);
    gl.uniform2f(u.cells, W, H);
    gl.uniform1f(u.fy, scrollY * d - rowNow());
  });
}

const settled = () => snaps.length > 0 && t - snaps[snaps.length - 1].t0 > DONE;
function hud() {
  const x = snaps.length ? Math.min(1, (t - snaps[snaps.length - 1].t0) / DONE) : 0;
  hudText.textContent =
    `colour = what the eye sees of light at ${LAMBDA.join(', ')} nm, run at ${FREQ.map(f => f.toFixed(1)).join(', ')} Hz · ` +
    `sent in only from the faces of a slab ${DEPTH * 2} wavelengths thick, out of focus first, then focused · wavelength at 10 Hz ${Math.round(phys.wl)} px · ` +
    `${x >= 1 ? 'standing' : `arriving ${Math.round(100 * x)}%`}`;
}

// ---------------------------------------------------------------- animation
function loop() { if (!raf) raf = requestAnimationFrame(tick); }

let lastFrame = 0;
function tick(now) {
  raf = 0;
  if (!R) return;                             // waves switched off
  t += lastFrame ? Math.min(1 / 30, (now - lastFrame) / 1000) : 1 / 60;
  if (moved && now - lastStep >= STEP_GAP) { changeDrive(now); moved = false; }
  prune();
  render();
  draw();
  hud();
  const more = moved || !settled();
  lastFrame = more ? now : 0;
  if (more) loop();
}

// Start already standing (reduced motion): everything has arrived.
function warmUp() {
  for (const s of snaps) s.t0 = t - DONE - 1;
  prune();
}

addEventListener('scroll', () => { if (R) { moved = true; loop(); } }, { passive: true });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (!R) return;
  readColors();
  changeDrive(performance.now());   // a new palette is a new page
  loop();
});

// ---------------------------------------------------------------- smooth scrolling
// Wheel steps ease toward their target instead of jumping, so the drives change gradually.
// Zoom (ctrl + wheel), sideways scrolling, touch, keys and the scrollbar stay native.
const glide = { target: 0, pos: 0, raf: 0, last: 0 };
function stopGlide() { if (glide.raf) { cancelAnimationFrame(glide.raf); glide.raf = 0; } }
function glideStep(now) {
  const dt = Math.min(0.05, (now - glide.last) / 1000);
  glide.last = now;
  glide.pos += (glide.target - glide.pos) * (1 - Math.exp(-dt / 0.14));
  if (Math.abs(glide.target - glide.pos) < 0.5) glide.pos = glide.target;
  scrollTo({ top: glide.pos, behavior: 'instant' });
  glide.raf = glide.pos === glide.target ? 0 : requestAnimationFrame(glideStep);
}
function enableGlide() {
  addEventListener('wheel', e => {
    if (!R || reduce.matches || e.ctrlKey || e.defaultPrevented || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? innerHeight : 1;
    if (!glide.raf) { glide.pos = glide.target = scrollY; glide.last = performance.now(); }
    const max = document.documentElement.scrollHeight - innerHeight;
    glide.target = Math.max(0, Math.min(max, glide.target + e.deltaY * unit));
    if (!glide.raf) glide.raf = requestAnimationFrame(glideStep);
  }, { passive: false });
  for (const type of ['pointerdown', 'keydown', 'touchstart']) addEventListener(type, stopGlide, { passive: true });
}

// The canvas is 100lvh tall, so a phone's URL bar showing or hiding does not resize the box.
// The grid resolves the finest (blue) light with PPW points per wavelength; if it would have too
// many points, the wavelength is made longer instead.
function layout() {
  Lx = canvas.clientWidth; Ly = canvas.clientHeight;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(Lx * dpr);
  canvas.height = Math.round(Ly * dpr);
  const blue = wl => wl * F_REF / FREQ[0];                         // blue wavelength, CSS px
  d = PPW / blue(wavelength);
  for (;;) {
    W = Math.max(8, Math.round(Lx * d)); H = Math.max(8, Math.round(Ly * d));
    FX = pow2AtLeast(W + 2 * RING); FY = pow2AtLeast(H + 2 * RING);
    if (FY - H - RING < H / 2) FY *= 2;                            // room to scroll a change
    if (FX * FY <= MAX_TEXELS || d < 0.05) break;
    d *= 0.95;
  }
  d = W / Lx;
  OX = (FX - W) >> 1; OY = (FY - H) >> 1;
  const wl = Math.max(wavelength, PPW / d * FREQ[0] / F_REF);
  phys.wl = wl;
  phys.k = FREQ.map(f => 2 * Math.PI / (wl * F_REF / f * d));     // radians per grid point
  allocate();
}

// Waves on: the canvas draws the page, sent in from the faces from scratch. Off: the plain page,
// with the simulation stopped and its GPU memory freed. The choice is remembered.
function setWaves(on, settledStart) {
  modeButton.setAttribute('aria-pressed', String(on));
  modeButton.textContent = on ? 'Waves: on' : 'Waves: off';
  knob.hidden = !on;
  try { localStorage.setItem('waves', on ? 'on' : 'off'); } catch (e) { /* not stored */ }
  if (!on) {
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    stopGlide();
    root.classList.remove('live');
    freeResources();
    return;
  }
  root.classList.add('live');
  readColors();
  try { layout(); } catch (e) {
    console.warn('waves: falling back to the plain page', e);
    root.classList.remove('live');
    freeResources();
    modeButton.hidden = true;
    knob.hidden = true;
    return;
  }
  showWavelength();
  moved = false; lastFrame = 0;
  changeDrive(performance.now());
  if (settledStart || reduce.matches) warmUp();
  loop();
}

// The wavelength slider: a new wavelength is a new field, so the page forms again from nothing.
function showWavelength() {
  knobInput.value = String(wavelength);
  knobOut.textContent = `${R ? Math.round(phys.wl) : wavelength} px`;   // as used (the grid may cap it)
}
let knobTimer = 0;
function setWavelength(v) {
  wavelength = v;
  try { localStorage.setItem('wavelength', String(v)); } catch (e) { /* not stored */ }
  if (!R) return;
  try { layout(); } catch (e) { console.warn('waves: grid for this wavelength failed', e); return; }
  moved = false; lastFrame = 0;
  changeDrive(performance.now());
  showWavelength();
  loop();
}

function start() {
  let on = true;
  try { on = localStorage.getItem('waves') !== 'off'; } catch (e) { /* default on */ }
  modeButton.hidden = false;
  modeButton.addEventListener('click', () => setWaves(!R));
  let saved = NaN;
  try { saved = +localStorage.getItem('wavelength'); } catch (e) { /* default */ }
  wavelength = Math.min(24, Math.max(4, +params.get('wavelength') || saved || WAVELENGTH));
  knobInput.addEventListener('input', () => {
    knobOut.textContent = `${knobInput.value} px`;
    clearTimeout(knobTimer);
    knobTimer = setTimeout(() => setWavelength(+knobInput.value), 200);
  });
  setWaves(on, params.has('still'));   // the drive switches on at load
  enableGlide();

  let timer = 0;
  addEventListener('resize', () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (!R || (canvas.clientWidth === Lx && canvas.clientHeight === Ly)) return;
      layout();
      changeDrive(performance.now());         // a new box: the page forms again
      loop();
    }, 180);
  });
}

canvas.addEventListener('webglcontextlost', e => {
  e.preventDefault();
  root.classList.remove('live');
  R = null;
  modeButton.hidden = true;
  knob.hidden = true;
});

(document.fonts ? document.fonts.ready : Promise.resolve()).then(start);
})();
