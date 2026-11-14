// The page as light: the middle plane z = 0 of a deep waveguide filled with a scalar wave field,
// seen the way an eye sees light. The only inputs are boundary conditions, at the guide's left
// and right ends.
//
// The field is three waves, at the frequencies of visible light at 460 nm (blue), 540 nm (green)
// and 620 nm (red), scaled down so that 550 nm would be 200 Hz: 239, 204 and 177 Hz. Three are
// enough: the eye has three kinds of colour receptor, so three spectral lines can match any colour
// on the page, as a laser projector does. Each pixel's colour is what those frequencies look like
// together: the intensity of each on the middle plane, weighted by the eye's colour matching
// functions (CIE 1931, analytic fit) and turned into sRGB. Nothing else sets a colour.
//
// The guide: x across the screen, between two open ends just beyond the screen's left and right
// edges; y along the page (uniform); z deep (uniform, mirror-symmetric about the screen). Inside,
// the wave equation u_tt = c^2 lap u, the same c for every frequency (set by
// WAVELENGTH, the wavelength at 200 Hz in CSS px). Each end sends in a prescribed wave and lets whatever arrives
// leave. Nothing else is driven: the screen is lit only by light that has come in from the ends.
//
// The guide is uniform in y and z, so its light is a sum of modes e^{i(kx x + ky y)} cos(kz z),
// each crossing in x at its own sideways wavenumber kx = sqrt(k^2 - ky^2 - kz^2): the left end
// sends those with kx > 0, the right end those with kx < 0. On the middle plane they add up to any
// pattern with |q| < k, so the ends can hold the page there (to about half a wavelength), as long
// as the light also moves sideways: light with |kx| near 0 runs almost along y or z and takes
// forever to get in from the ends, so the page is fitted without it (|kx| > GRAZE k).
//
// The fit: for each frequency, the field on the middle plane whose amplitude is the page's, with
// its phase free (Gerchberg-Saxton, with FFTs), seeded with light slanted toward the middle (sin =
// TILT): the left half of the page carried mostly from the left end, the right half from the
// right. What each end sends follows from that field mode by mode.
//
// Arrival, exactly: for each mode, the crossing from an end is a 1D Klein-Gordon problem in x
// (u_tt = c^2 u_xx - c^2 (k^2 - kx^2) u), whose response to a wave switched on at the end is known
// (computed here through its spectrum): nothing until the light-speed front, then the wave arriving
// at its group velocity c |kx| / k, dispersed. The ends switch on sharply (RAMP), so the front rings:
// precursors ahead of it, overshoot and beats behind it, travelling in from the ends. It depends only on |kx| / k and the distance from
// the end. So the light is split by |kx| / k (NODES smooth bands per end, on a log scale of
// k / |kx|) and each band is stored as an image; each frame, each band is weighted by its
// response at each point's distance from its end, and they add up to the field now.
//
// The page changes when it loads and while it scrolls. The field moves with the page; the ends are
// refitted as it scrolls (each refit keeps the light already there and fits only what is new),
// and each row of the page is lit from the moment it first came into view: light from the ends
// has been arriving there since. Light that scrolls off the screen is gone.
(() => {
'use strict';

const WAVELENGTH = 5;      // CSS px: the wavelength at 200 Hz (?wavelength= overrides)
// The frequencies: visible light at these wavelengths (nm), scaled so that 550 nm is 200 Hz.
const LAMBDA = [460, 540, 620];
const F_REF = 200, LAMBDA_REF = 550;
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
const PPW = 2.5;           // grid points per wavelength of the finest (blue) light
const MAX_TEXELS = 1 << 20;   // grid points at most (else the wavelength is made longer)
const BAND = 0.9;          // patterns fitted up to this fraction of each frequency's wavenumber
const TAPER = 0.75;        // ... rolling off smoothly from this fraction of that
const GRAZE = 0.1;         // no light with |kx| below this fraction of k ...
const GRAZE_FULL = 0.22;   // ... full light above this
const TILT = 0.3;          // sin of the slant toward the middle the fit starts from
const GS_ITERS = 8;        // phase-fit rounds for a new page ...
const GS_ITERS_SCROLL = 3; // ... and for a scrolled one (most of it kept)
const STEPS = 2;           // steps of a scrolled fit per frame (it is spread over frames)
const RING = 12;           // grid points around the screen left free by the fit; the ends sit at its edge
const RAMP = 0.04;         // s: the ends switch a row's light on over this (8 cycles at 200 Hz):
                           // sharp, so the light arrives as a ringing front, not a fade
const NODES = 8;           // bands of k / |kx| per end
const LS0 = Math.log(1 / BAND), LS1 = Math.log(1 / GRAZE), DLS = (LS1 - LS0) / (NODES - 1);
const ND = 64, NA = 1024;  // response table: distances 0..tmax; ages from the light-speed front to settled
const FS = 100;            // Hz: the response is computed at this many samples per s
const STEP_GAP = 100;      // ms: start a refit at most this often while scrolling

const root = document.documentElement;
const canvas = document.getElementById('field');
const doc = document.getElementById('content');
const hudText = document.getElementById('hud-text');
const modeButton = document.getElementById('mode');
const reduce = matchMedia('(prefers-reduced-motion: reduce)');
const params = new URLSearchParams(location.search);

let gl = null;
try {
  gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: false });
} catch (e) { gl = null; }
if (!gl || !gl.getExtension('EXT_color_buffer_float')) return;   // the plain HTML page stays
const LINEAR = !!gl.getExtension('OES_texture_float_linear');     // filtered table lookups

// ---------------------------------------------------------------- shaders
// The grid (F.x x F.y, powers of two, periodic) holds the screen at (OX, OY) and a free ring
// around it; the window is the screen and its ring. Fields and spectra hold two complex numbers
// per texel: frequencies 2p and 2p + 1 of part p (the fourth is spare).
const HEAD = '#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\nprecision highp sampler2DArray;\nout vec4 o;\n';
const VS = '#version 300 es\nin vec2 p;\nvoid main(){ gl_Position = vec4(p, 0.0, 1.0); }';
const CPLX = `
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
vec2 cis(float a) { return vec2(cos(a), sin(a)); }`;
// The wave vector (radians per grid point) of spectral index n, and the light the ends can
// deliver: crossing patterns (|q| < BAND k, tapered) that also move sideways (|kx| > GRAZE k).
const Q = `
uniform ivec2 F;
vec2 wave(ivec2 n) {
  return 6.28318531 * vec2(n.x < F.x / 2 ? n.x : n.x - F.x, n.y < F.y / 2 ? n.y : n.y - F.y) / vec2(F);
}
vec2 band(ivec2 n, vec2 k) {
  vec2 v = wave(n);
  vec2 t = clamp((length(v) / (${BAND} * k) - ${TAPER}) / (1.0 - ${TAPER}), 0.0, 1.0);
  vec2 g = clamp((abs(v.x) / k - ${GRAZE}) / ${(GRAZE_FULL - GRAZE).toFixed(4)}, 0.0, 1.0);
  return (0.5 + 0.5 * cos(3.14159265 * t)) * (0.5 - 0.5 * cos(3.14159265 * g));
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
// light), with the phase of light slanted toward the middle; rows kept from the last fit start
// (and stay) as they were.
const FS_GS_INIT = HEAD + CPLX + `
uniform sampler2D T, C;        // the page (linear sRGB); the last fit, made cs rows ago
uniform int cs;
uniform ivec2 lit;             // rows [lit.x, lit.y) keep the last fit
uniform vec3 share[2];         // this part's two rows of MIX
uniform vec2 k;                // this part's two wavenumbers, radians per grid point
uniform float xc;              // the screen's middle column
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  if (n.y >= lit.x && n.y < lit.y) { o = texelFetch(C, ivec2(n.x, n.y + cs), 0); return; }
  vec3 t = texelFetch(T, n, 0).rgb;
  float a0 = sqrt(max(dot(share[0], t), 0.0)), a1 = sqrt(max(dot(share[1], t), 0.0));
  vec2 tilt = -${TILT} * k * abs(float(n.x) + 0.5 - xc);
  o = vec4(a0 * cis(tilt.x), a1 * cis(tilt.y));
}`;
// Keep only the light the ends can deliver, conjugated and scaled for the inverse transform.
const FS_GS_BAND = HEAD + Q + `
uniform sampler2D X;
uniform vec2 k;
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec2 w = band(n, k) / float(F.x * F.y);
  vec4 x = texelFetch(X, n, 0) * vec4(w.x, w.x, w.y, w.y);
  o = vec4(x.x, -x.y, x.z, -x.w);
}`;
// Keep that field's phase; put the target amplitude back on the screen (the kept rows as they
// were), leave the ring free and keep the rest dark.
const FS_GS_PHASE = HEAD + `
uniform sampler2D Y, T, C;
uniform int cs;
uniform ivec2 lit;
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
  if (n.y >= lit.x && n.y < lit.y) { o = texelFetch(C, ivec2(n.x, n.y + cs), 0); return; }
  vec3 t = texelFetch(T, n, 0).rgb;
  float a0 = sqrt(max(dot(share[0], t), 0.0)), a1 = sqrt(max(dot(share[1], t), 0.0));
  o = vec4(a0 * unit(z0), a1 * unit(z1));
}`;
// One band of the fitted light, half transformed (x still spectral): the light from end `side`
// (0 left: kx > 0; 1 right) in band `node` of log(k / |kx|) (smooth hats that add up to one).
// Conjugated for the inverse transform.
const FS_HAT = HEAD + Q + `
uniform sampler2D X;
uniform vec2 k;
uniform int side, node;
float hat(float kx, float k) {
  if (side == 0 ? kx <= 0.0 : kx >= 0.0) return 0.0;
  float ls = clamp(log(k / abs(kx)), ${LS0.toFixed(6)}, ${LS1.toFixed(6)});
  return max(0.0, 1.0 - abs(ls - ${LS0.toFixed(6)} - float(node) * ${DLS.toFixed(6)}) / ${DLS.toFixed(6)});
}
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  float kx = wave(n).x;
  vec4 x = texelFetch(X, n, 0) * vec4(vec2(hat(kx, k.x)), vec2(hat(kx, k.y)));
  o = vec4(x.x, -x.y, x.z, -x.w);
}`;
// A region of a (conjugated) transform, conjugated back.
const FS_TAKE = HEAD + `
uniform sampler2D X;
uniform ivec2 org;
void main() {
  vec4 x = texelFetch(X, ivec2(gl_FragCoord.xy) + org, 0);
  o = vec4(x.x, -x.y, x.z, -x.w);
}`;

// The field now, on the window: every band of light from each end, weighted by its response at
// that point's distance from the end and the age of the light in that row of the page.
const FS_COMBINE = HEAD + CPLX + `
uniform sampler2DArray L;      // the bands: layer (part * 2 + side) * NODES + node; row: page row mod hh
uniform sampler2D E, A;        // the response (this part's two frequencies); each row's age
uniform int part, base, hh, ww;   // the bands' slot for the window's first row; window size
uniform float cps, tmax, amax; // light speed (grid points per s); distances in the table; all arrived
uniform float slow[${NODES}], lead[${NODES}], span[${NODES}];   // per band: k / |kx|; the table's ages
vec4 at(int b, float T, float a) {   // the table, bilinear, by age from the band's group arrival
  float tau = a - T * slow[b];
  if (tau <= -lead[b]) return vec4(0.0);
  if (tau >= span[b]) return vec4(1.0, 0.0, 1.0, 0.0);
  float x = clamp(T / tmax, 0.0, 1.0) * ${ND - 1}.0, y = (tau + lead[b]) / (lead[b] + span[b]) * ${NA - 1}.0;
  ${LINEAR ? `
  return texture(E, vec2((float(${ND} * b) + x + 0.5) / ${NODES * ND}.0, (y + 0.5) / ${NA}.0));` : `
  int x0 = int(floor(x)), y0 = int(floor(y)); float fx = x - float(x0), fy = y - float(y0);
  int x1 = min(x0 + 1, ${ND - 1}), y1 = min(y0 + 1, ${NA - 1});
  x0 += ${ND} * b; x1 += ${ND} * b;
  return mix(mix(texelFetch(E, ivec2(x0, y0), 0), texelFetch(E, ivec2(x1, y0), 0), fx),
             mix(texelFetch(E, ivec2(x0, y1), 0), texelFetch(E, ivec2(x1, y1), 0), fx), fy);`}
}
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  int ys = (base + n.y) % hh;
  float a = texelFetch(A, ivec2(n.y, 0), 0).r;
  if (a < 0.0) { o = vec4(0.0); return; }
  bool done = a >= amax;
  vec4 sum = vec4(0.0);
  for (int s = 0; s < 2; s++) {
    float T = (s == 0 ? float(n.x) + 0.5 : float(ww - n.x) - 0.5) / cps;
    for (int b = 0; b < ${NODES}; b++) {
      vec4 f = texelFetch(L, ivec3(n.x, ys, (part * 2 + s) * ${NODES} + b), 0);
      if (done) { sum += f; continue; }
      vec4 e = at(b, T, a);
      sum += vec4(cmul(e.xy, f.xy), cmul(e.zw, f.zw));
    }
  }
  o = sum;
}`;

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
    hat: program(FS_HAT), take: program(FS_TAKE), combine: program(FS_COMBINE),
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
// Draw with `prog` into fbo (w x h), or only into its rows [y0, y0 + rows).
function run(prog, fbo, w, h, tex, set, y0 = 0, rows = h) {
  gl.useProgram(prog.p);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.viewport(0, y0, w, rows);
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
let WW = 0, HH = 0;            // the window: the screen and its ring
let d = 1;                     // grid points per CSS px
let wavelength = WAVELENGTH;
const phys = { wl: 0, k: [0, 0, 0, 0], cps: 1, tmax: 1, amax: 1, bands: null };
let R = null;
let t = 0;                     // s of time
let fitRow = 0, hasFit = false;   // the page row the last fit was made at
let lit = null, LIT0 = 0;      // per page row (offset LIT0): the time its light was switched on, or NaN
let ageBuf = null;
let raf = 0, moved = false, lastStep = -1e9, colors = null;
const SEE_MAT = new Float32Array([SEE[0][0], SEE[1][0], SEE[2][0], SEE[0][1], SEE[1][1], SEE[2][1], SEE[0][2], SEE[1][2], SEE[2][2]]);
const pow2AtLeast = n => 1 << Math.ceil(Math.log2(Math.max(2, n)));

function freeResources() {
  if (!R) return;
  for (const r of [...R.pair, R.mix, R.page, ...R.fit, ...R.cur, R.see]) { gl.deleteTexture(r.t); gl.deleteFramebuffer(r.f); }
  gl.deleteTexture(R.L.t); R.L.fbos.forEach(f => gl.deleteFramebuffer(f));
  R.E.forEach(e => gl.deleteTexture(e));
  gl.deleteTexture(R.A);
  R = null;
}

// ---------------------------------------------------------------- the response from an end
// For each frequency, band node (|kx| / k = 1 / s) and distance T (in light-seconds, 0..tmax):
// the field at that distance from an end that switches the mode on over RAMP, relative to its
// steady state. The mode crosses as a Klein-Gordon wave, kx(w) = sqrt(w^2 - kappa^2) (c = 1,
// kappa^2 = k^2 - kx^2: its wavenumber across the guide), so
//   E(T, t) = (1 / 2 pi) int R^(W) e^{i (kx(w + W) - kx(w)) T} e^{-i W t} dW,
// R^ the spectrum of the switch-on; evaluated by FFT (the switch-on made a long pulse, so it
// is periodic, and read before it ends). Nothing arrives before the light-speed front (t = T);
// the wave itself comes at the group velocity (t = T s), ringing, its precursors ahead of it.
// Tabulated by age from the group arrival, tau = t - T s, from the front at the farthest
// distance (-lead) until it has settled (span), when it is blended into 1.
const twiddles = {};
function fftInPlace(re, im, sign) {
  const n = re.length;
  if (!twiddles[n]) {
    const c = new Float64Array(n / 2), si = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { c[i] = Math.cos(2 * Math.PI * i / n); si[i] = Math.sin(2 * Math.PI * i / n); }
    twiddles[n] = [c, si];
  }
  const [C, S] = twiddles[n];
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let x = re[i]; re[i] = re[j]; re[j] = x; x = im[i]; im[i] = im[j]; im[j] = x; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const step = n / len, h = len >> 1;
    for (let i = 0; i < n; i += len) {
      for (let j = 0; j < h; j++) {
        const cr = C[j * step], ci = sign * S[j * step], p = i + j, q = p + h;
        const xr = re[q] * cr - im[q] * ci, xi = re[q] * ci + im[q] * cr;
        re[q] = re[p] - xr; im[q] = im[p] - xi;
        re[p] += xr; im[p] += xi;
      }
    }
  }
}
// Per band: k / |kx|, and the table's ages before and after the group arrival.
function bandTimes(tmax) {
  const slow = [], lead = [], span = [];
  for (let b = 0; b < NODES; b++) {
    const sb = Math.exp(LS0 + b * DLS);
    slow.push(sb); lead.push(tmax * (sb - 1) + 0.05); span.push(1.5 + 0.5 * tmax * sb);
  }
  return { slow, lead, span };
}
function computeResponse(tmax) {
  const ramp = x => x <= 0 ? 0 : x >= RAMP ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * x / RAMP);
  const DT = 1 / FS;
  const { slow, lead, span } = bandTimes(tmax);
  const tables = [new Float32Array(NODES * ND * NA * 4), new Float32Array(NODES * ND * NA * 4)];
  for (let b = 0; b < NODES; b++) {
    const last = tmax * slow[b] + span[b];             // the latest age needed
    const pulse = last + 1;
    const N = pow2AtLeast(Math.ceil((2 * last + 4) * FS));
    const Rr = new Float64Array(N), Ri = new Float64Array(N);
    for (let n = 0; n < N; n++) Rr[n] = (ramp(n * DT) - ramp(n * DT - pulse)) * DT;
    fftInPlace(Rr, Ri, 1);                             // R^(W_m), W_m = 2 pi m / (N DT)
    const er = new Float64Array(N), ei = new Float64Array(N);
    const dkr = new Float64Array(N), dki = new Float64Array(N);
    for (let f = 0; f < 3; f++) {
      const w = 2 * Math.PI * FREQ[f];
      const tab = tables[f >> 1], ch = (f & 1) * 2;
      const kx0 = w / slow[b], kap2 = w * w - kx0 * kx0;
      for (let m = 0; m < N; m++) {                     // kx(w + W) - kx(w), causal branch
        const W = 2 * Math.PI * (m < N / 2 ? m : m - N) / (N * DT), om = w + W;
        const q = om * om - kap2;
        if (q >= 0) { dkr[m] = Math.sign(om) * Math.sqrt(q) - kx0; dki[m] = 0; }
        else { dkr[m] = -kx0; dki[m] = Math.sqrt(-q); }
      }
      for (let iT = 0; iT < ND; iT++) {
        const T = tmax * iT / (ND - 1);
        for (let m = 0; m < N; m++) {
          const mag = Math.exp(-dki[m] * T), ph = dkr[m] * T;
          const c = mag * Math.cos(ph), si = mag * Math.sin(ph);
          er[m] = Rr[m] * c - Ri[m] * si; ei[m] = Rr[m] * si + Ri[m] * c;
        }
        fftInPlace(er, ei, -1);
        const norm = 1 / (N * DT);
        for (let ia = 0; ia < NA; ia++) {
          const tau = -lead[b] + (lead[b] + span[b]) * ia / (NA - 1), a = T * slow[b] + tau;
          let rr = 0, ri = 0;
          if (a > 0) {
            const x = a / DT, n0 = Math.min(N - 2, Math.floor(x)), fx = x - n0;
            rr = (er[n0] * (1 - fx) + er[n0 + 1] * fx) * norm;
            ri = (ei[n0] * (1 - fx) + ei[n0 + 1] * fx) * norm;
          }
          const fade = Math.min(1, Math.max(0, (tau - 0.6 * span[b]) / (0.4 * span[b])));
          const sm = fade * fade * (3 - 2 * fade);
          rr += (1 - rr) * sm; ri -= ri * sm;
          const o = ((ia * NODES + b) * ND + iT) * 4 + ch;
          tab[o] = rr; tab[o + 1] = ri;
        }
      }
    }
  }
  return tables;
}
let tableKey = '', tableData = null;
function responseTables() {
  const key = phys.tmax.toFixed(4);
  if (key !== tableKey) { tableData = computeResponse(phys.tmax); tableKey = key; }
  return tableData.map(data => {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    nearest(gl.TEXTURE_2D);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, NODES * ND, NA, 0, gl.RGBA, gl.FLOAT, data);
    if (LINEAR) {
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    }
    return tex;
  });
}

function allocate() {
  freeResources();
  if (Math.max(FX, FY, NODES * ND) > gl.getParameter(gl.MAX_TEXTURE_SIZE)) throw new Error('grid too large');
  const A = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, A);
  nearest(gl.TEXTURE_2D);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, HH, 1, 0, gl.RED, gl.FLOAT, null);
  R = {
    pair: [target(FX, FY), target(FX, FY)],      // transforms and fitting
    mix: target(FX, FY),                         // the fitted light, transformed in y only
    page: target(FX, FY, true),                  // the page (linear sRGB)
    fit: [target(FX, FY, true), target(FX, FY, true)],   // the last fit (steady field), per part
    L: layers(WW, HH, 2 * 2 * NODES),            // the bands: part, end, node
    cur: [target(WW, HH, true), target(WW, HH, true)],   // the field now, on the window
    see: target(WW, HH, true),                   // what it looks like (linear sRGB)
    E: responseTables(),                         // the response of each band
    A,                                           // the age of each window row's light
  };
  ageBuf = new Float32Array(HH);
  hasFit = false; job = null;
  t = 0;
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
    if (el.closest('svg')) continue;               // a figure's text: drawn with its figure
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

// Figures (inline svg.figure, coloured with var(--fg), var(--muted), var(--neg) and var(--bg)) are
// light too: each is drawn into the page as an image of itself in the current colours. A figure
// whose image cannot be made (or would taint the capture) is shown as it is, over the field.
const figs = new Map();
let figKey = '';
function prepareFigures() {
  const key = [colors.fgCss, colors.mutedCss, colors.negCss, colors.bgCss].join();
  if (key === figKey) return Promise.resolve(false);
  figKey = key;
  return Promise.all([...doc.querySelectorAll('svg.figure')].map(svg => {
    const src = svg.outerHTML.replace(/var\(--(fg|muted|neg|bg)\)/g, (_, n) => colors[`${n}Css`]);
    const img = new Image();
    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(src)}`;
    return img.decode().then(() => {
      const t = document.createElement('canvas').getContext('2d');
      t.drawImage(img, 0, 0);
      t.getImageData(0, 0, 1, 1);
      figs.set(svg, img);
      svg.classList.remove('shown');
    }).catch(() => { figs.delete(svg); svg.classList.add('shown'); });
  })).then(() => true);
}
function paintFigures(c, h) {
  for (const [svg, img] of figs) {
    const b = svg.getBoundingClientRect();
    if (b.bottom < -40 || b.top > h + 40 || !b.width) continue;
    c.drawImage(img, b.left, b.top, b.width, b.height);
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
  paintFigures(ctx, Ly);
  ctx.restore();
  const img = ctx.getImageData(0, 0, FX, FY).data;
  if (!pageBuf || pageBuf.length !== FX * FY * 4) pageBuf = new Float32Array(FX * FY * 4);
  for (let i = 0, n = FX * FY * 4; i < n; i++) pageBuf[i] = LIN[img[i]];
  gl.bindTexture(gl.TEXTURE_2D, R.page.t);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, FX, FY, gl.RGBA, gl.FLOAT, pageBuf);
}

// Forward FFT in place over the ping-pong pair, starting from pair[a], along x, y or both (x only
// over grid rows [rows[0], rows[1]) if given); the index of the result.
function fft(a, axes = 'xy', rows = null) {
  const pair = R.pair;
  const passes = axes === 'x' ? [[1, FX]] : axes === 'y' ? [[0, FY]] : [[1, FX], [0, FY]];
  for (const [horiz, size] of passes) {
    for (let sub = 2; sub <= size; sub *= 2) {
      run(P.fft, pair[1 - a].f, FX, FY, { src: pair[a].t }, u => {
        gl.uniform1i(u.size, size); gl.uniform1i(u.sub, sub); gl.uniform1i(u.horiz, horiz);
      }, ...(rows && horiz ? [rows[0], rows[1] - rows[0]] : []));
      a = 1 - a;
    }
  }
  return a;
}
const kOf = part => [phys.k[2 * part], phys.k[2 * part + 1] || phys.k[2]];
const mod = (a, n) => ((a % n) + n) % n;

// The page rows lit, and when.
const litAt = row => { const i = row + LIT0; return i >= 0 && i < lit.length ? lit[i] : NaN; };
function setLit(row, v) { const i = row + LIT0; if (i >= 0 && i < lit.length) lit[i] = v; }

// Fit the page (captured at page row `row`) and split what the ends must send into bands, a step
// at a time (a generator: the steps can be spread over frames). Rows still on the screen since the
// last fit keep their light; the rest of the screen is new, and only it (and a margin, where the
// band limit lets the two meet) is split again. The bands are kept by page row, modulo HH.
function* fitJob(row, iters) {
  const pair = R.pair, T = R.page.t;
  const cs = row - fitRow;
  let l0 = 0, l1 = 0;                                // grid rows kept
  if (hasFit) {
    const a = Math.max(row, fitRow), b = Math.min(row + H, fitRow + H);
    if (b > a) { l0 = OY + a - row; l1 = OY + b - row; }
  }
  // Grid rows to split again: all of the window, or the new rows and a margin.
  let s0 = OY - RING, s1 = OY + H + RING;
  if (l1 > l0) {
    const M = RING;
    const top = l0 > OY, bottom = l1 < OY + H;
    s0 = top ? OY - RING : l1 - M;
    s1 = bottom ? OY + H + RING : l0 + M;
    if (top && bottom) { s0 = OY - RING; s1 = OY + H + RING; }
    if (!top && !bottom) { s0 = s1 = 0; }
  }
  const rows = [s0, s1];
  for (let part = 0; part < 2; part++) {
    const share = new Float32Array([...MIX[2 * part], ...MIX[2 * part + 1]]);
    const k = kOf(part);
    const bandPass = a => run(P.gsBand, pair[1 - a].f, FX, FY, { X: pair[a].t }, u => { gl.uniform2i(u.F, FX, FY); gl.uniform2f(u.k, k[0], k[1]); });
    run(P.gsInit, pair[0].f, FX, FY, { T, C: R.fit[part].t }, u => {
      gl.uniform3fv(u.share, share); gl.uniform2f(u.k, k[0], k[1]); gl.uniform1f(u.xc, OX + W / 2);
      gl.uniform1i(u.cs, cs); gl.uniform2i(u.lit, l0, l1);
    });
    let a = 0;
    for (let it = 0; it < iters; it++) {
      a = fft(a);
      bandPass(a);
      yield;
      a = fft(1 - a);
      run(P.gsPhase, pair[1 - a].f, FX, FY, { Y: pair[a].t, T, C: R.fit[part].t }, u => {
        gl.uniform3fv(u.share, share); gl.uniform4i(u.win, OX, OY, W, H); gl.uniform1i(u.ring, RING);
        gl.uniform1i(u.cs, cs); gl.uniform2i(u.lit, l0, l1);
      });
      a = 1 - a;
      yield;
    }
    // The light the ends deliver: band-limited, transformed back in y (x still spectral).
    a = fft(a);
    bandPass(a);
    a = fft(1 - a, 'y');
    run(P.take, R.mix.f, FX, FY, { X: pair[a].t }, u => gl.uniform2i(u.org, 0, 0));   // (conjugated back)
    yield;
    // The steady field, kept to start the next fit from.
    const win = [OY - RING, OY + H + RING];
    run(P.take, pair[0].f, FX, FY, { X: R.mix.t }, u => gl.uniform2i(u.org, 0, 0), win[0], HH);
    a = fft(0, 'x', win);
    run(P.take, R.fit[part].f, FX, FY, { X: pair[a].t }, u => gl.uniform2i(u.org, 0, 0), win[0], HH);
    yield;
    // Each band from each end, into the rows it belongs to.
    if (s1 > s0) {
      for (let side = 0; side < 2; side++) {
        for (let node = 0; node < NODES; node++) {
          run(P.hat, pair[0].f, FX, FY, { X: R.mix.t }, u => {
            gl.uniform2i(u.F, FX, FY); gl.uniform2f(u.k, k[0], k[1]);
            gl.uniform1i(u.side, side); gl.uniform1i(u.node, node);
          }, s0, s1 - s0);
          a = fft(0, 'x', rows);
          const layer = R.L.fbos[(part * 2 + side) * NODES + node];
          // grid row g is page row row + g - OY, kept at slot mod(that, HH); in at most two runs
          for (let g = s0; g < s1;) {
            const slot = mod(row + g - OY, HH), n = Math.min(s1 - g, HH - slot);
            run(P.take, layer, WW, HH, { X: pair[a].t }, u => gl.uniform2i(u.org, OX - RING, g - slot), slot, n);
            g += n;
          }
          if (node % 4 === 3) yield;
        }
      }
    }
  }
  // Done: the new fit stands. Rows on the screen keep their time; new ones are switched on now;
  // the rest are gone.
  const keep = new Float64Array(H);
  for (let i = 0; i < H; i++) keep[i] = litAt(row + i);
  lit.fill(NaN);
  for (let i = 0; i < H; i++) setLit(row + i, Number.isNaN(keep[i]) ? t : keep[i]);
  fitRow = row; hasFit = true;
}

const rowNow = () => Math.floor(scrollY * d);
let job = null;                // a fit in progress

// The ends switch to the page as it is now: at once (fresh: everything forms again from
// darkness), or spread over the next frames (scrolling).
function changeDrive(now, fresh) {
  const row = rowNow();
  if (fresh) { lit.fill(NaN); hasFit = false; }
  capturePage(row);
  job = fitJob(row, hasFit ? GS_ITERS_SCROLL : GS_ITERS);
  if (fresh) { while (!job.next().done); job = null; }
  lastStep = now;
}
// Some steps of the fit in progress; true when it has finished.
function stepJob(n) {
  for (let i = 0; i < n && job; i++) if (job.next().done) job = null;
  return !job;
}

// The field now, and what it looks like.
function render() {
  const row = rowNow();
  for (let j = 0; j < HH; j++) {
    const s = litAt(row + j - RING);
    ageBuf[j] = Number.isNaN(s) ? -1 : t - s;
  }
  gl.bindTexture(gl.TEXTURE_2D, R.A);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, HH, 1, gl.RED, gl.FLOAT, ageBuf);
  for (let part = 0; part < 2; part++) {
    run(P.combine, R.cur[part].f, WW, HH, { L: [gl.TEXTURE_2D_ARRAY, R.L.t], E: R.E[part], A: R.A }, u => {
      gl.uniform1i(u.part, part); gl.uniform1i(u.base, mod(row - RING, HH));
      gl.uniform1i(u.hh, HH); gl.uniform1i(u.ww, WW);
      gl.uniform1f(u.cps, phys.cps); gl.uniform1f(u.tmax, phys.tmax); gl.uniform1f(u.amax, phys.amax);
      gl.uniform1fv(u.slow, phys.bands.slow); gl.uniform1fv(u.lead, phys.bands.lead); gl.uniform1fv(u.span, phys.bands.span);
    });
  }
  run(P.see, R.see.f, WW, HH, { C0: R.cur[0].t, C1: R.cur[1].t }, u => gl.uniformMatrix3fv(u.see, false, SEE_MAT));
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
    gl.uniform2i(u.F, WW, HH); gl.uniform2i(u.off, RING, RING);
    gl.uniform2f(u.res, canvas.width, canvas.height);
    gl.uniform2f(u.cells, W, H);
    gl.uniform1f(u.fy, scrollY * d - rowNow());
  });
}

// How far the light on the screen has arrived: its youngest row's age, of amax.
function arrived() {
  const row = rowNow();
  let x = 1;
  for (let i = 0; i < H; i++) {
    const s = litAt(row + i);
    x = Math.min(x, Number.isNaN(s) ? 0 : (t - s) / phys.amax);
  }
  return Math.max(0, x);
}
function hud(x) {
  hudText.textContent =
    `colour = what the eye sees of light at ${LAMBDA.join(', ')} nm, run at ${FREQ.map(f => Math.round(f)).join(', ')} Hz · ` +
    `sent in only from the left and right ends of a deep waveguide · wavelength at ${F_REF} Hz ${Math.round(phys.wl)} px · ` +
    `${x >= 1 ? 'standing' : `arriving ${Math.round(100 * x)}%`}`;
}

// ---------------------------------------------------------------- animation
function loop() { if (!raf) raf = requestAnimationFrame(tick); }

let lastFrame = 0;
function tick(now) {
  raf = 0;
  if (!R) return;                             // waves switched off
  t += lastFrame ? Math.min(1 / 30, (now - lastFrame) / 1000) : 1 / 60;
  if (job) stepJob(STEPS);
  else if (moved && now - lastStep >= STEP_GAP) { changeDrive(now, false); moved = false; stepJob(STEPS); }
  render();
  draw();
  const x = arrived();
  hud(x);
  const more = moved || job || x < 1;
  lastFrame = more ? now : 0;
  if (more) loop();
}

// Start already standing (reduced motion): everything has arrived.
function warmUp() {
  for (let i = 0; i < lit.length; i++) if (!Number.isNaN(lit[i])) lit[i] = t - phys.amax - 1;
}

addEventListener('scroll', () => { if (R) { moved = true; loop(); } }, { passive: true });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (!R) return;
  readColors();
  prepareFigures().then(() => {
    if (!R) return;
    changeDrive(performance.now(), true);   // a new palette is a new page
    loop();
  });
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
    if (FX * FY <= MAX_TEXELS || d < 0.05) break;
    d *= 0.95;
  }
  d = W / Lx;
  OX = (FX - W) >> 1; OY = (FY - H) >> 1;
  WW = W + 2 * RING; HH = H + 2 * RING;
  const wl = Math.max(wavelength, PPW / d * FREQ[0] / F_REF);
  phys.wl = wl;
  phys.k = FREQ.map(f => 2 * Math.PI / (wl * F_REF / f * d));     // radians per grid point
  phys.cps = wl * d * F_REF;                                       // light speed, grid points per s
  phys.tmax = WW / phys.cps;                                       // s: light across the window
  phys.bands = bandTimes(phys.tmax);
  phys.amax = Math.max(...phys.bands.slow.map((sb, b) => phys.tmax * sb + phys.bands.span[b]));   // s: all arrived
  LIT0 = RING + 8;
  lit = new Float64Array(Math.ceil(document.documentElement.scrollHeight * d) + 2 * (H + RING) + 32).fill(NaN);
  allocate();
}

// Waves on: the canvas draws the page, sent in from the ends from scratch. Off: the plain page,
// with the simulation stopped and its GPU memory freed. The choice is remembered.
function setWaves(on, settledStart) {
  modeButton.setAttribute('aria-pressed', String(on));
  modeButton.textContent = on ? 'Waves: on' : 'Waves: off';
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
    return;
  }
  moved = false; lastFrame = 0;
  changeDrive(performance.now(), true);
  if (settledStart || reduce.matches) warmUp();
  loop();
  prepareFigures().then(changed => {             // the palette changed while the waves were off
    if (!changed || !R) return;
    changeDrive(performance.now(), true);
    loop();
  });
}

function start() {
  let on = true;
  try { on = localStorage.getItem('waves') !== 'off'; } catch (e) { /* default on */ }
  modeButton.hidden = false;
  modeButton.addEventListener('click', () => setWaves(!R));
  wavelength = Math.min(24, Math.max(4, +params.get('wavelength') || WAVELENGTH));
  setWaves(on, params.has('still'));   // the drive switches on at load
  enableGlide();

  let timer = 0;
  addEventListener('resize', () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (!R || (canvas.clientWidth === Lx && canvas.clientHeight === Ly)) return;
      layout();
      changeDrive(performance.now(), true);         // a new box: the page forms again
      loop();
    }, 180);
  });
}

canvas.addEventListener('webglcontextlost', e => {
  e.preventDefault();
  root.classList.remove('live');
  R = null;
  modeButton.hidden = true;
});

(document.fonts ? document.fonts.ready : Promise.resolve())
  .then(() => { readColors(); return prepareFigures(); })
  .then(start);
})();
