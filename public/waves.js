// The page as light: the slice z = 0 through a scalar wave field in a 3D slab, stepped forward in
// time, seen the way an eye sees light. The only inputs are boundary conditions.
//
// The field is three waves, at the frequencies of visible light at 460 nm (blue), 540 nm (green)
// and 620 nm (red), scaled down so that 550 nm would be 10 Hz: 12.0, 10.2 and 8.9 Hz. Three are
// enough: the eye has three kinds of colour receptor, so three spectral lines can match any colour
// on the page, as a laser projector does. Each pixel's colour is what those frequencies look like
// together: the intensity of each on the slice, weighted by the eye's colour matching functions
// (CIE 1931, analytic fit) and turned into sRGB. Nothing else sets a colour.
//
// The slab: x, y across the screen and a damping margin around it (open edges: waves leave as
// if into open space), z from -h to h. Inside, every grid point is a harmonic oscillator coupled
// to its neighbours, the damped wave equation u_tt = c^2 lap u - gamma u_t, stepped by leapfrog
// with the 7-point Laplacian, the same c for every frequency, so each has its own wavelength. c
// is set by the knob, the wavelength at 10 Hz in CSS px. The field is driven only through the
// boundary condition on the planes z = +-h, where it is prescribed: u = g(x, y, t). The drive is
// the same on both planes, so the field is mirror-symmetric and only z in [0, h] is stepped.
//
// The drive for a page: each frequency w's share of the page (its intensity, the phase free) is
// fitted within the band of in-plane patterns q that can cross the slab as waves; finer ones
// would only decay from the planes. For the discrete scheme the steady state is exact in closed
// form: with x(q) = 1 + (sigma(q) - W^2) / 2 (sigma the in-plane Laplacian's symbol, W^2 the
// scheme's own discrete w^2 with damping, over c^2), u_j = g cos(theta j) / cos(theta J) at layer
// j, cos(theta) = x; so the slice holds the page when g = page x T_J(x), T_J the Chebyshev
// polynomial. The fit leaves what light really does when made to draw: rings and coloured fringes
// where the frequencies disagree. A shorter wavelength makes it as close as wanted.
//
// When the page changes (it loads, or it scrolls and more of it comes into view), the drive fades
// over to the new page and the field carries on from where it is: the change comes in from the
// planes and rings out across the slab, coarse patterns first (they cross head-on), fine ones
// last (they cross at grazing angles). The field and drive scroll with the page: each stored
// field remembers the grid row it was made at and is read shifted; what scrolls past an edge leaves.
(() => {
'use strict';

const WAVELENGTH = 8;      // CSS px: the wavelength at 10 Hz (?wavelength= overrides): the knob
// The frequencies: visible light at these wavelengths (nm), scaled so that 550 nm is 10 Hz.
const LAMBDA = [460, 540, 620];
const F_REF = 10, LAMBDA_REF = 550;
const FREQ = LAMBDA.map(l => F_REF * LAMBDA_REF / l);
const OMEGA = FREQ.map(f => 2 * Math.PI * f);
// Linear sRGB of unit intensity at each frequency (CIE 1931 analytic fit, scaled so all three at
// unit intensity have unit green), and a non-negative mix of frequencies for each sRGB channel
// with SEE * MIX = identity: the page's colours split exactly into the frequencies' intensities.
const SEE = [
  [-0.00643, -0.43319, 1.69758],
  [-0.07621, 1.17375, -0.09754],
  [1.37121, -0.12184, -0.02223],
];
const MIX = [[0.01429, 0.08145, 0.73388], [0.05097, 0.87585, 0.04892], [0.60214, 0.22381, 0.01526], [0, 0, 0]];
const PPW = 5;             // grid points per wavelength of the finest (blue) light
const LAYERS = 6;          // grid layers from the slice to the boundary plane (h = LAYERS points)
const MAX_CELLS = 3e6;     // grid points in the slab at most (else the wavelength is made longer)
const BAND = 0.9;          // patterns fitted up to this fraction of each frequency's wavenumber
const TAPER = 0.75;        // ... rolling off smoothly from this fraction of that
const GS_ITERS = 6;        // phase-fit iterations per page change
const ROUGH = 0.15;        // randomness of the phase the fit starts from (0: flat, 1: random)
const GAMMA = 1.5;         // 1/s: damping; the field settles into a new page over a few 1/GAMMA
const SETTLE = 8 / GAMMA;  // s: after this, a page counts as standing and stepping stops
const PAD = 24;            // grid points of damping margin around the screen
const GAMMA_EDGE = 40;     // 1/s: damping at the outer edge of the margin
const CFL = 0.5;           // c dt / dx
const MAX_SUB = 8;         // time steps per frame at most (slower than real time beyond that)
const STEP_GAP = 220;      // ms: at most one drive change per this interval while scrolling
const RAMP = 0.4;          // s: a new drive fades in over this (a smooth switch rings less)

const root = document.documentElement;
const canvas = document.getElementById('field');
const doc = document.getElementById('content');
const hudText = document.getElementById('hud-text');
const modeButton = document.getElementById('mode');
const knob = document.getElementById('knob');
const knobInput = knob.querySelector('input'), knobOut = knob.querySelector('output');
const reduce = matchMedia('(prefers-reduced-motion: reduce)');
const params = new URLSearchParams(location.search);
let maxCells = MAX_CELLS;

let gl = null;
try {
  gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: false });
} catch (e) { gl = null; }
if (!gl || !gl.getExtension('EXT_color_buffer_float')) return;   // the plain HTML page stays

// ---------------------------------------------------------------- shaders
const HEAD = '#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\nprecision highp sampler2DArray;\nout vec4 o;\n';
const VS = '#version 300 es\nin vec2 p;\nvoid main(){ gl_Position = vec4(p, 0.0, 1.0); }';
const CPLX = `
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
vec2 cis(float a) { return vec2(cos(a), sin(a)); }`;

// ---- the drive, fitted on the fit grid (F.x x F.y, a power of two at least the slab, periodic;
// cell (i, j) is the slab's grid point (i, j), margins included). Part p holds frequencies 2p and 2p + 1 (the
// fourth is spare), each complex.

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
  float a = -6.283185307179586 * float(idx % sub) / float(sub);
  vec2 w = cis(a);
  o = e + vec4(cmul(w, q.xy), cmul(w, q.zw));
}`;

// The in-plane Laplacian's symbol (grid spacing 1) at fit-grid frequency n: sigma = |q|^2 for
// coarse patterns.
const SIGMA = `
uniform ivec2 F;
float sigma(ivec2 n) {
  vec2 q = 6.28318531 * vec2(n.x < F.x / 2 ? n.x : n.x - F.x, n.y < F.y / 2 ? n.y : n.y - F.y) / vec2(F);
  return 4.0 * (sin(0.5 * q.x) * sin(0.5 * q.x) + sin(0.5 * q.y) * sin(0.5 * q.y));
}
vec2 taper(ivec2 n, vec2 kc) {   // raised-cosine band limit, per frequency of the part
  vec2 t = clamp((sqrt(sigma(n)) / kc - ${TAPER}) / (1.0 - ${TAPER}), 0.0, 1.0);
  return 0.5 + 0.5 * cos(3.14159265 * t);
}`;

// Start: each frequency's target amplitude (square root of its share of the page's light) with a
// phase pattern fixed to the page (it moves with the text).
const FS_GS_INIT = HEAD + CPLX + `
uniform sampler2D T;           // the page, linear sRGB, on the fit grid
uniform int part, row;
uniform float rough;
uniform vec3 share[2];         // this part's two rows of MIX
float hash(int x, int y, int c) {
  uint h = uint(x) * 1664525u ^ uint(y) * 22695477u ^ uint(c) * 2654435761u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
  return float(h) / 4294967296.0;
}
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec3 t = texelFetch(T, n, 0).rgb;
  float a0 = sqrt(max(dot(share[0], t), 0.0)), a1 = sqrt(max(dot(share[1], t), 0.0));
  int y = n.y + row;
  o = vec4(a0 * cis(rough * 6.2831853 * hash(n.x, y, 2 * part)), a1 * cis(rough * 6.2831853 * hash(n.x, y, 2 * part + 1)));
}`;
// Keep only the patterns that can cross the slab, conjugated and scaled for the inverse transform.
const FS_GS_BAND = HEAD + SIGMA + `
uniform sampler2D X;
uniform vec2 kc;
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec2 w = taper(n, kc) / float(F.x * F.y);
  vec4 x = texelFetch(X, n, 0) * vec4(w.x, w.x, w.y, w.y);
  o = vec4(x.x, -x.y, x.z, -x.w);
}`;
// Keep that field's phase, put the target amplitude back.
const FS_GS_PHASE = HEAD + `
uniform sampler2D Y, T;
uniform vec3 share[2];
vec2 unit(vec2 z) { float m = length(z); return m > 1e-20 ? z / m : vec2(1.0, 0.0); }
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec4 y = texelFetch(Y, n, 0);
  vec3 t = texelFetch(T, n, 0).rgb;
  float a0 = sqrt(max(dot(share[0], t), 0.0)), a1 = sqrt(max(dot(share[1], t), 0.0));
  o = vec4(a0 * unit(vec2(y.x, -y.y)), a1 * unit(vec2(y.z, -y.w)));
}`;
// The fitted slice field's spectrum -> the boundary value that holds it: times T_J(x(q)), from the
// scheme's own steady state (see the top). Band-limited, conjugated and scaled for the inverse.
const FS_DRIVE = HEAD + CPLX + SIGMA + `
uniform sampler2D X;
uniform vec2 kc;
uniform vec2 W2re, W2im;       // the scheme's w^2 (with damping) over c^2, per frequency of the part
uniform int J;
vec2 cheb(vec2 x) {            // T_J(x), complex
  vec2 a = vec2(1.0, 0.0), b = x;
  for (int k = 1; k < 32; k++) { if (k >= J) break; vec2 c = 2.0 * cmul(x, b) - a; a = b; b = c; }
  return J == 0 ? a : b;
}
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  float s = sigma(n);
  vec2 w = taper(n, kc) / float(F.x * F.y);
  vec4 x = texelFetch(X, n, 0);
  vec2 g0 = w.x * cmul(x.xy, cheb(vec2(1.0 + 0.5 * (s - W2re.x), -0.5 * W2im.x)));
  vec2 g1 = w.y * cmul(x.zw, cheb(vec2(1.0 + 0.5 * (s - W2re.y), -0.5 * W2im.y)));
  o = vec4(g0.x, -g0.y, g1.x, -g1.y);
}`;
// Undo the conjugation of the inverse transform.
const FS_UNCONJ = HEAD + `
uniform sampler2D X;
void main() { vec4 x = texelFetch(X, ivec2(gl_FragCoord.xy), 0); o = vec4(x.x, -x.y, x.z, -x.w); }`;

// ---- the slab (N.x x N.y points across, the screen at PAD..; layers 0 = slice .. J - 1, the
// boundary plane is layer J).

// The drive on the boundary plane now: fading from the previous page's (D0) to the current one's
// (D1), each two textures (parts) on the fit grid, read where the page has moved it since.
const DRIVE = CPLX + `
uniform sampler2D D0a, D0b, D1a, D1b;
uniform ivec2 F;
uniform int pad, dsh0, dsh1;   // rows the page has moved since each drive was made
uniform float fade, t;
uniform vec3 w;
vec3 drive1(sampler2D a, sampler2D b, ivec2 p, int sh) {
  ivec2 f = ivec2(p.x, p.y + sh);
  if (f.y < 0 || f.y >= F.y) return vec3(0.0);   // scrolled out of what it was made for
  vec4 x = texelFetch(a, f, 0); vec2 y = texelFetch(b, f, 0).xy;
  // Re(G e^{-i w t}) = Re G cos + Im G sin
  return vec3(x.x, x.z, y.x) * cos(w * t) + vec3(x.y, x.w, y.y) * sin(w * t);
}
vec3 driveAt(ivec2 p) {
  return mix(drive1(D0a, D0b, p, dsh0), drive1(D1a, D1b, p, dsh1), fade);
}`;

// Fade the drive over: the drive as it is now, into a new pair (at shift 0 from now).
const FS_BLEND = HEAD + `
uniform sampler2D A0, A1;      // the same part of the previous and current drives
uniform ivec2 F;
uniform int sh0, sh1;
uniform float fade;
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  int y0 = n.y + sh0, y1 = n.y + sh1;
  vec4 a = y0 < 0 || y0 >= F.y ? vec4(0.0) : texelFetch(A0, ivec2(n.x, y0), 0);
  vec4 b = y1 < 0 || y1 >= F.y ? vec4(0.0) : texelFetch(A1, ivec2(n.x, y1), 0);
  o = mix(a, b, fade);
}`;

// One leapfrog step of layer j, every frequency (r, g, b = blue, green, red) at once. The two
// previous levels are read where the page has moved them since they were made.
const FS_STEP = HEAD + DRIVE + `
uniform sampler2DArray Up, Uc;
uniform ivec2 N;
uniform int j, J, shP, shC;
uniform float dt, c2, g0, gEdge;
vec3 U(sampler2DArray s, ivec2 p, int l, int sh) {
  p.y += sh;
  if (p.x < 0 || p.y < 0 || p.x >= N.x || p.y >= N.y) return vec3(0.0);
  if (l == J) return driveAt(p);        // the boundary condition
  return texelFetch(s, ivec3(p, l < 0 ? -l : l), 0).rgb;   // mirror at the slice
}
float damping(ivec2 p) {
  int d = min(min(p.x, N.x - 1 - p.x), min(p.y, N.y - 1 - p.y));
  float r = clamp(float(pad - d) / float(pad), 0.0, 1.0);
  return g0 + gEdge * r * r;
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 uc = U(Uc, p, j, shC), up = U(Up, p, j, shP);
  vec3 lap = U(Uc, p + ivec2(1, 0), j, shC) + U(Uc, p - ivec2(1, 0), j, shC)
           + U(Uc, p + ivec2(0, 1), j, shC) + U(Uc, p - ivec2(0, 1), j, shC)
           + U(Uc, p, j + 1, shC) + U(Uc, p, j - 1, shC) - 6.0 * uc;
  float g = damping(p);
  o = vec4((2.0 * uc - (1.0 - 0.5 * g * dt) * up + dt * dt * c2 * lap) / (1.0 + 0.5 * g * dt), 0.0);
}`;

// What the eye sees on the slice: each frequency's intensity from three successive steps (for
// u = a cos(w t) + anything slow, the second and central differences give a cos and a sin at the
// middle step exactly), through the colour matching functions, as linear sRGB.
const FS_SEE = HEAD + `
uniform sampler2DArray U0, U1, U2;   // two steps ago, one ago, now
uniform ivec2 N;
uniform int s0, s1, s2;
uniform float dt;
uniform vec3 w;
uniform mat3 see;
vec3 at(sampler2DArray s, ivec2 p, int sh) {
  p.y += sh;
  return p.y < 0 || p.y >= N.y ? vec3(0.0) : texelFetch(s, ivec3(p, 0), 0).rgb;
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 a = at(U0, p, s0), b = at(U1, p, s1), c = at(U2, p, s2);
  vec3 ph = w * dt;
  vec3 x = (c - 2.0 * b + a) / (2.0 * (cos(ph) - 1.0)), y = (c - a) / (2.0 * sin(ph));
  o = vec4(see * (x * x + y * y), 0.0);
}`;

// The screen: the colour between grid points reconstructed with a Lanczos-3 kernel, then sRGB.
const FS_DISPLAY = HEAD + `
uniform sampler2D F;           // linear sRGB on the grid
uniform ivec2 N;
uniform vec2 res, cells;       // canvas px; grid points across the screen
uniform int pad;
uniform float fy;              // fraction of a grid row the page is past the field
vec3 E(int x, int y) { return texelFetch(F, clamp(ivec2(x, y), ivec2(0), N - 1), 0).rgb; }
float lz(float x) {
  x = abs(x);
  if (x < 1e-4) return 1.0;
  if (x >= 3.0) return 0.0;
  float p = 3.14159265 * x;
  return 3.0 * sin(p) * sin(p / 3.0) / (p * p);
}
void main() {
  vec2 g = vec2(gl_FragCoord.x / res.x * cells.x, (1.0 - gl_FragCoord.y / res.y) * cells.y + fy) + float(pad) - 0.5;
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
    drive: program(FS_DRIVE), unconj: program(FS_UNCONJ), blend: program(FS_BLEND),
    step: program(FS_STEP), see: program(FS_SEE), disp: program(FS_DISPLAY),
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
let W = 0, H = 0, Nx = 0, Ny = 0, FX = 0, FY = 0;   // grid points: screen; slab; fit grid
let d = 1;                     // grid points per CSS px
let wavelength = WAVELENGTH;
const phys = { c: 0, c2: 0, wl: 0, dt: 0, W2re: [0, 0, 0, 0], W2im: [0, 0, 0, 0], kc: [0, 0, 0, 0] };
let R = null;
let simT = 0;                  // s of simulated time
let lrow = [0, 0, 0];          // grid row (page scroll) each field level was made at
let drow = [0, 0];             // ... and the previous and current drives
let changedAt = -1e9;          // simulated time of the last drive change
let raf = 0, moved = false, lastStep = -1e9, colors = null;
const SEE_MAT = new Float32Array([SEE[0][0], SEE[1][0], SEE[2][0], SEE[0][1], SEE[1][1], SEE[2][1], SEE[0][2], SEE[1][2], SEE[2][2]]);
const pow2AtLeast = n => 1 << Math.ceil(Math.log2(Math.max(2, n)));

function freeResources() {
  if (!R) return;
  for (const r of [R.see, R.page, ...R.fit, ...R.D.flat()]) { gl.deleteTexture(r.t); gl.deleteFramebuffer(r.f); }
  for (const s of R.U) { gl.deleteTexture(s.t); s.fbos.forEach(f => gl.deleteFramebuffer(f)); }
  R = null;
}

function allocate() {
  freeResources();
  const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  if (Math.max(Nx, Ny, FX, FY) > max) throw new Error('grid too large');
  R = {
    U: [0, 1, 2].map(() => layers(Nx, Ny, LAYERS, true)),   // the field: one step ago, now, two ago
    see: target(Nx, Ny, true),                             // what the slice looks like
    page: target(FX, FY),                                  // the page (linear sRGB) on the fit grid
    fit: [target(FX, FY), target(FX, FY)],                 // fitting
    D: [0, 1, 2].map(() => [target(FX, FY), target(FX, FY)]),   // drives: previous, current, spare
  };
  for (const s of R.U) for (const f of s.fbos) { gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
  for (const r of R.D.flat()) { gl.bindFramebuffer(gl.FRAMEBUFFER, r.f); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
  simT = 0; changedAt = -1e9;
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

// The page scrolled to grid row `row`, linear sRGB, on the fit grid (the slab and its margins; the
// background beyond).
function capturePage(row) {
  rc.width = FX; rc.height = FY;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = colors.bgCss;
  ctx.fillRect(0, 0, FX, FY);
  ctx.setTransform(d, 0, 0, d, PAD, (scrollY - row / d) * d + PAD);   // the page runs on into the margins
  paintText(ctx, Ly + PAD / d);
  const img = ctx.getImageData(0, 0, FX, FY).data;
  if (!pageBuf || pageBuf.length !== FX * FY * 4) pageBuf = new Float32Array(FX * FY * 4);
  for (let i = 0, n = FX * FY * 4; i < n; i++) pageBuf[i] = LIN[img[i]];
  gl.bindTexture(gl.TEXTURE_2D, R.page.t);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, FX, FY, gl.RGBA, gl.FLOAT, pageBuf);
}

// Forward FFT in place over the ping-pong pair (FX x FY), starting from pair[a]; the index of the result.
function fft(pair, a) {
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

// The boundary value that holds the page (captured at `row`) on the slice, into drive pair `slot`.
function fitDrive(slot, row) {
  const pair = R.fit, T = R.page.t;
  for (let part = 0; part < 2; part++) {
    const share = new Float32Array([...MIX[2 * part], ...MIX[2 * part + 1]]);
    const kc = [phys.kc[2 * part], phys.kc[2 * part + 1]];
    run(P.gsInit, pair[0].f, FX, FY, { T }, u => {
      gl.uniform1i(u.part, part); gl.uniform1i(u.row, row); gl.uniform3fv(u.share, share);
      gl.uniform1f(u.rough, +(params.get('rough') ?? ROUGH));
    });
    let a = 0;
    for (let it = 0; it < GS_ITERS; it++) {
      a = fft(pair, a);
      run(P.gsBand, pair[1 - a].f, FX, FY, { X: pair[a].t }, u => { gl.uniform2i(u.F, FX, FY); gl.uniform2fv(u.kc, kc); });
      a = fft(pair, 1 - a);
      run(P.gsPhase, pair[1 - a].f, FX, FY, { Y: pair[a].t, T }, u => gl.uniform3fv(u.share, share));
      a = 1 - a;
    }
    a = fft(pair, a);
    run(P.drive, pair[1 - a].f, FX, FY, { X: pair[a].t }, u => {
      gl.uniform2i(u.F, FX, FY); gl.uniform2fv(u.kc, kc); gl.uniform1i(u.J, LAYERS);
      gl.uniform2f(u.W2re, phys.W2re[2 * part], phys.W2re[2 * part + 1]);
      gl.uniform2f(u.W2im, phys.W2im[2 * part], phys.W2im[2 * part + 1]);
    });
    a = fft(pair, 1 - a);
    run(P.unconj, R.D[slot][part].f, FX, FY, { X: pair[a].t });
  }
}

const rowNow = () => Math.floor(scrollY * d);
const fadeAt = t => { const x = Math.min(1, Math.max(0, (t - changedAt) / RAMP)); return 0.5 - 0.5 * Math.cos(Math.PI * x); };

// Switch the drive to the page as it is now, fading over from wherever it has got to.
function changeDrive(now) {
  const row = rowNow();
  for (let part = 0; part < 2; part++) {
    run(P.blend, R.D[2][part].f, FX, FY, { A0: R.D[0][part].t, A1: R.D[1][part].t }, u => {
      gl.uniform2i(u.F, FX, FY);
      gl.uniform1i(u.sh0, row - drow[0]); gl.uniform1i(u.sh1, row - drow[1]);
      gl.uniform1f(u.fade, fadeAt(simT));
    });
  }
  R.D = [R.D[2], R.D[1], R.D[0]];
  drow[0] = row;
  capturePage(row);
  fitDrive(1, row);
  drow[1] = row;
  changedAt = simT;
  lastStep = now;
}

// Fixed steps: as many as fit the time since the last frame.
let pending = 0;
function advance(frame) {
  const dt = phys.dt;
  pending += frame;
  const n = Math.min(MAX_SUB, Math.floor(pending / dt));
  pending = n === MAX_SUB ? 0 : pending - n * dt;   // too slow: run slower than real time
  const row = rowNow();
  for (let k = 0; k < n; k++) {
    const fade = fadeAt(simT);
    for (let j = 0; j < LAYERS; j++) {
      run(P.step, R.U[2].fbos[j], Nx, Ny, {
        Up: [gl.TEXTURE_2D_ARRAY, R.U[0].t], Uc: [gl.TEXTURE_2D_ARRAY, R.U[1].t],
        D0a: R.D[0][0].t, D0b: R.D[0][1].t, D1a: R.D[1][0].t, D1b: R.D[1][1].t,
      }, u => {
        gl.uniform2i(u.N, Nx, Ny); gl.uniform2i(u.F, FX, FY);
        gl.uniform1i(u.j, j); gl.uniform1i(u.J, LAYERS); gl.uniform1i(u.pad, PAD);
        gl.uniform1i(u.shP, row - lrow[0]); gl.uniform1i(u.shC, row - lrow[1]);
        gl.uniform1i(u.dsh0, row - drow[0]); gl.uniform1i(u.dsh1, row - drow[1]);
        gl.uniform1f(u.fade, fade); gl.uniform1f(u.t, simT); gl.uniform3fv(u.w, OMEGA);
        gl.uniform1f(u.dt, dt); gl.uniform1f(u.c2, phys.c2);
        gl.uniform1f(u.g0, GAMMA); gl.uniform1f(u.gEdge, GAMMA_EDGE);
      });
    }
    // The new level is at time simT + dt (it read the drive at simT for layer J's neighbour).
    R.U = [R.U[1], R.U[2], R.U[0]];      // one step ago, now, two steps ago
    lrow = [lrow[1], row, lrow[0]];
    simT += dt;
  }
}

function see() {
  const row = rowNow();
  run(P.see, R.see.f, Nx, Ny, {
    U0: [gl.TEXTURE_2D_ARRAY, R.U[2].t], U1: [gl.TEXTURE_2D_ARRAY, R.U[0].t], U2: [gl.TEXTURE_2D_ARRAY, R.U[1].t],
  }, u => {
    gl.uniform2i(u.N, Nx, Ny);
    gl.uniform1i(u.s0, row - lrow[2]); gl.uniform1i(u.s1, row - lrow[0]); gl.uniform1i(u.s2, row - lrow[1]);
    gl.uniform1f(u.dt, phys.dt); gl.uniform3fv(u.w, OMEGA);
    gl.uniformMatrix3fv(u.see, false, SEE_MAT);
  });
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
  run(P.disp, null, canvas.width, canvas.height, { F: R.see.t }, u => {
    gl.uniform2i(u.N, Nx, Ny);
    gl.uniform2f(u.res, canvas.width, canvas.height);
    gl.uniform2f(u.cells, W, H);
    gl.uniform1i(u.pad, PAD);
    gl.uniform1f(u.fy, scrollY * d - rowNow());
  });
}

const settled = () => simT - changedAt > SETTLE;
function hud() {
  const x = Math.min(1, (simT - changedAt) / SETTLE);
  hudText.textContent =
    `colour = what the eye sees of light at ${LAMBDA.join(', ')} nm, run at ${FREQ.map(f => f.toFixed(1)).join(', ')} Hz · ` +
    `driven only from planes above and below · wavelength at 10 Hz ${Math.round(phys.wl)} px · ${x >= 1 ? 'standing' : `settling ${Math.round(100 * x)}%`}`;
}

// ---------------------------------------------------------------- animation
function loop() { if (!raf) raf = requestAnimationFrame(tick); }

let lastFrame = 0, slowFrames = 0;
function tick(now) {
  raf = 0;
  if (!R) return;                             // waves switched off
  const frame = lastFrame ? Math.min(1 / 30, (now - lastFrame) / 1000) : 1 / 60;
  if (lastFrame && now - lastFrame < 250) {   // adaptive resolution for slow GPUs
    slowFrames = now - lastFrame > 30 ? slowFrames + 1 : Math.max(0, slowFrames - 1);
    if (slowFrames > 24 && Nx * Ny * LAYERS > 4e5) {
      slowFrames = 0;
      maxCells = Nx * Ny * LAYERS / 2;
      layout();
      changeDrive(now);
    }
  }
  let more = false;
  if (moved && now - lastStep >= STEP_GAP) { changeDrive(now); moved = false; }
  if (moved) more = true;
  if (!settled()) advance(frame);
  see();
  draw();
  hud();
  if (!settled()) more = true;
  lastFrame = more ? now : 0;
  if (more) loop();
}

// Start already standing (reduced motion): run the field forward until it has settled.
function warmUp() {
  const steps = Math.ceil(SETTLE / phys.dt);
  for (let k = 0; k < steps; k += MAX_SUB) advance(MAX_SUB * phys.dt);
  pending = 0;
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
// The grid resolves the finest (blue) light with PPW points per wavelength; if the slab would have
// too many points, the wavelength is made longer instead.
function layout() {
  Lx = canvas.clientWidth; Ly = canvas.clientHeight;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(Lx * dpr);
  canvas.height = Math.round(Ly * dpr);
  const blue = wl => wl * F_REF / FREQ[0];                         // blue wavelength, CSS px
  d = PPW / blue(wavelength);
  for (;;) {
    W = Math.max(8, Math.round(Lx * d)); H = Math.max(8, Math.round(Ly * d));
    if ((W + 2 * PAD) * (H + 2 * PAD) * LAYERS <= maxCells || d < 0.05) break;
    d *= 0.95;
  }
  d = W / Lx;
  Nx = W + 2 * PAD; Ny = H + 2 * PAD;
  FX = pow2AtLeast(Nx); FY = pow2AtLeast(Ny);
  const wl = Math.max(wavelength, PPW / d * FREQ[0] / F_REF);
  const c = wl * F_REF * d;                                         // grid points / s
  const dt = CFL / c;
  // Per frequency: the scheme's own w^2 and damping, over c^2 (for the steady state), and the
  // band limit (a fraction of the discrete wavenumber).
  const W2re = [0, 0, 0, 0], W2im = [0, 0, 0, 0], kc = [1, 1, 1, 1];
  OMEGA.forEach((w, i) => {
    const wd2 = (2 - 2 * Math.cos(w * dt)) / (dt * dt), wg = Math.sin(w * dt) / dt;
    W2re[i] = wd2 / (c * c); W2im[i] = GAMMA * wg / (c * c);
    kc[i] = BAND * Math.sqrt(W2re[i]);
  });
  Object.assign(phys, { wl, c, c2: c * c, dt, W2re, W2im, kc });
  allocate();
  lrow = [rowNow(), rowNow(), rowNow()];
  drow = [rowNow(), rowNow()];
}

// Waves on: the canvas draws the page, formed from scratch by the drive. Off: the plain page,
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
