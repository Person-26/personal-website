// The page as the 2D slice z = 0 of an electromagnetic wave in open 3D space.
// Colour is the field: (R, G, B) - background = (E_x, E_y, E_z) on the slice.
//
// The screen x in [0,Lx], y in [0,Ly] is a window on the slice, and space is open all round:
// every edge of the screen and both sides of the slice in z lead out to the far field.
//
// Nothing is drawn directly. Light is massless, omega = c |k|, and transverse, k . E = 0, so
// Gauss's law holds for every wave. Each in-plane mode q stands for a bundle of 3D plane waves
// (q, |q| s) with the tilt s Gaussian (width SIGMA), which is enough to carry any colour below
// the band limit: across q any tilt carries the field; the part along q and E_z need a tilt.
// Among the waves with that much spread, the one meeting the page with least energy is used.
// The waves are sent by currents: one filling each side margin of the screen (outside the text
// column) and two sheets at z = +-h, parallel to the slice, all shaped to send waves only inward
// (by the equivalence theorem such currents can produce any field inside the region they
// bound, and none outside). They are timed so that every wave is in step on the slice T after a
// change: tau = T - age before that. On the slice a mode is then c (|q| - K) tau out of step
// with the frame (Omega = c K), and the tilts disagree further by c |q| s^2 tau / 2; the sum
// over s is exact in closed form (paraxially): a power-law fade, (1 + u^2)^(-1/4) for the
// across part and (1 + u^2)^(-3/4) for the along part, u = c |q| tau SIGMA^2. So the slice
// holds the change evolved back by tau: waves that come in across the edges of the screen and
// interfere, and sharpen in from out of the plane, focusing into
// the page at tau = 0. While a change is on its way in, the field is cut off past the margin
// currents and builds up across them; and the slice stays dark for the light time from the sheets,
// SHEET. c is set so the fastest waves cross the width in T. (A change let go before it arrived
// is approximated as still arriving in full.)
//
// The transforms: x and y are zero-padded to twice the screen (open: what leaves goes into the
// padding, not back in). E_x + i E_y
// share one complex FFT; their spectra are separated with Z(-k) for the along/across split.
// The field scrolls with the page: every stored field keeps the grid row it was made at and is
// read shifted by how far the page has moved since. A drive change sends in the whole current
// page, moved text included, and stops sending the old one. A page that had arrived was being
// held by the sheets sending it continuously at the carrier frequency (every mode q crossing the
// slice at k_z = sqrt(K^2 - q^2), which is why the band limit sits below K); nothing of it is
// stored, so it lasts only until the last wavefront sent has crossed the gap, broad shapes first,
// fine detail (grazing) last. So a drive change while the drive is running is a switch-over: the
// currents send the new page at once, and each mode of it reaches the slice just as the old one's
// last wavefront passes, the same delay (only from nothing, at load or for a new wavelength, is
// the page sent as a focusing pulse). A pulse still on its way when let go runs on freely by the same
// formula, tau now negative: it spreads out of the slice in z and runs out
// across the edges of the screen, the time mirror of how it arrived. (A change let go before it arrived
// keeps going too: its waves still focus, then run on out.) Once a mode has moved a whole screen
// in x or y it is all in the padding and is dropped, before the periodic transform could bring it back;
// whatever of a released field is still left after D_MAX is dropped too.
// Text scrolled past the top or bottom simply leaves the window.
// Every change in flight is transformed back each frame with its modes' current phases.
//
// Colour is amplitude: the field is the complex envelope of light at the carrier frequency, and
// each pixel shows (R, G, B) = background +- (|E_x|, |E_y|, |E_z|). The phase is free, so for each
// page the sheets' field is fitted to it (Gerchberg-Saxton, from a phase pattern fixed to the
// page) within the band |q| < K: the knob is the carrier wavelength 2 pi / K. A field that fine can
// only approximate the page; what is left over is what a light field really does when it is made
// to draw: speckle in the strokes and dark vortices, points where the amplitude is zero and the
// phase winds around them. A shorter wavelength makes the approximation as good as wanted.
(() => {
'use strict';

const WAVELENGTH = 8;      // CSS px: the carrier wavelength (?wavelength= overrides): the knob
const RHO = 0.8;           // the amplitude's band (twice the field's) may use this much of the grid
const GS_ITERS = 6;        // phase-fit iterations per drive change
const EPS = 0.06;          // how close the finest in-slice mode sits to cutoff
const T_ARRIVE = 4.0;      // s: from a drive change to the slice holding it
const SIGMA = 0.25;        // spread of the tilt out of the slice (k_z / |q|)
const D_MAX = 8.0;         // s: drop what is left of a released field after this
const TAPER = 0.75;        // the band limit rolls off smoothly from this fraction of it
const SHEET = 0.4;         // s: light time from the z current sheets to the slice
const SLOTS = 3;           // changes arriving, held or leaving at once
const STEP_GAP = 220;      // ms: at most one drive change per this interval while scrolling
const MAX_POINTS = 1 << 19;
let maxPoints = MAX_POINTS;

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

// ---------------------------------------------------------------- shaders
const HEAD = '#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\nprecision highp sampler2DArray;\nout vec4 o;\n';
const VS = '#version 300 es\nin vec2 p;\nvoid main(){ gl_Position = vec4(p, 0.0, 1.0); }';

// A fitted field (two complex numbers per texel), zero-padded to twice the size each way (open on
// every side).
const FS_EXT_RAW = HEAD + `
uniform ivec2 N;
uniform sampler2D U;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  o = p.x < N.x && p.y < N.y ? texelFetch(U, p, 0) : vec4(0.0);
}`;

// Fitting the sheets' field to the page's amplitude, on the screen grid (periodic). Start: the
// page's amplitude with a phase pattern fixed to the page (it moves with the text), per component.
// Part 0 holds (E_x, E_y), part 1 holds E_z, each complex.
const FS_GS_INIT = HEAD + `
uniform sampler2DArray T;
uniform int part, row;
float hash(int x, int y, int c) {
  uint h = uint(x) * 1664525u ^ uint(y) * 22695477u ^ uint(c) * 2654435761u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
  return float(h) / 4294967296.0;
}
vec2 cis(float a) { return vec2(cos(a), sin(a)); }
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec3 t = texelFetch(T, ivec3(n, 0), 0).rgb;
  int y = n.y + row;
  const float TAU = 6.28318531;
  o = part == 0 ? vec4(t.r * cis(TAU * hash(n.x, y, 0)), t.g * cis(TAU * hash(n.x, y, 1)))
                : vec4(t.b * cis(TAU * hash(n.x, y, 2)), 0.0, 0.0);
}`;
// Keep only what the sheets can send (|q| below the band limit), conjugated and scaled for the
// inverse transform.
const FS_GS_BAND = HEAD + `
uniform sampler2D X;
uniform ivec2 N;
uniform vec2 ks;               // (2 pi / Lx, 2 pi / Ly)
uniform float kc;
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec2 k = vec2(n.x < N.x / 2 ? n.x : n.x - N.x, n.y < N.y / 2 ? n.y : n.y - N.y) * ks;
  float t = clamp((length(k) / kc - ${TAPER}) / (1.0 - ${TAPER}), 0.0, 1.0);
  vec4 x = texelFetch(X, n, 0) * (0.5 + 0.5 * cos(3.14159265 * t)) / float(N.x * N.y);
  o = vec4(x.x, -x.y, x.z, -x.w);
}`;
// Keep that field's phase, put the page's amplitude back.
const FS_GS_PHASE = HEAD + `
uniform sampler2D Y;
uniform sampler2DArray T;
uniform int part;
vec2 unit(vec2 z) { float m = length(z); return m > 1e-20 ? z / m : vec2(1.0, 0.0); }
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec4 y = texelFetch(Y, n, 0);
  vec2 a = vec2(y.x, -y.y), b = vec2(y.z, -y.w);
  vec3 t = texelFetch(T, ivec3(n, 0), 0).rgb;
  o = part == 0 ? vec4(t.r * unit(a), t.g * unit(b)) : vec4(t.b * unit(a), 0.0, 0.0);
}`;

// Signed wavevector of spectrum index n.
const KVEC = `
uniform ivec2 N;
uniform vec2 kscale;           // (pi / Lx, pi / Ly)
vec2 kvec(ivec2 n) { return vec2(n.x < N.x ? n.x : n.x - 2 * N.x, n.y < N.y ? n.y : n.y - 2 * N.y) * kscale; }`;

// Forward FFT -> band-limited spectrum of a change: what the drives send in.
const FS_SPEC = HEAD + KVEC + `
uniform sampler2D G;
uniform float kc;
void main() {
  float m = sqrt(float(4 * N.x * N.y));      // stored / sqrt(M): fits half precision
  ivec2 n = ivec2(gl_FragCoord.xy);
  float r = length(kvec(n)) / kc;            // raised-cosine roll-off: no ringing at the band edge
  float t = clamp((r - ${TAPER}) / (1.0 - ${TAPER}), 0.0, 1.0);
  o = texelFetch(G, n, 0) * (0.5 + 0.5 * cos(3.14159265 * t)) / m;
}`;

// A change's spectrum on the slice now, ready for the inverse FFT (conjugated and scaled:
// IFFT(X) = conj(FFT(conj X)) / M). The slice z = 0 sits in open 3D space: each in-plane mode q
// stands for 3D plane waves (q, |q| s) tilted out of the slice by a Gaussian spread of s, all
// massless (omega = c |k|) and transverse (k . E = 0). Over the tilts the slice sees them in step
// only at the focus; away from it the paraxial sum over s is exact in closed form, a power-law
// fade as the field spreads out of the slice in z. Transversality splits the in-plane field into
// the part across q (any tilt carries it), the part along q and E_z (only tilted waves carry
// them, so they focus more sharply in time).
const FS_EVOLVE = HEAD + KVEC + `
uniform sampler2DArray Z;
uniform int la, part;          // the change's slot; 0: (E_x, E_y), 1: E_z
uniform float c, K, age, T, sheet, sig, dmax;
uniform float off;             // s after the start when its drive let go of it (>= 1e8: never)
uniform int cw;                // 1: switched over from a running drive, not sent as a pulse
vec2 conj(vec2 z) { return vec2(z.x, -z.y); }
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
vec2 cis(float a) { return vec2(cos(a), sin(a)); }
// The tilt averages <e^{i phase}> and <s^2 e^{i phase}> / sig^2 of the envelope, phase relative
// to the frame turning at c K, tau before the focus (negative after).
void tilt(float m, float tau, out vec2 t0, out vec2 t2) {
  float u = c * m * tau * sig * sig, a = atan(u), r = 1.0 + u * u;
  float p = c * (m - K) * tau;
  t0 = pow(r, -0.25) * cis(p + 0.5 * a);
  t2 = pow(r, -0.75) * cis(p + 1.5 * a);
}
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec2 k = kvec(n);
  float m = length(k);
  float tau = T - age, env = 1.0;
  // Each mode's light crosses the gap from the sheets along z at c k_z / K.
  float cross = sheet * K / sqrt(max(K * K - m * m, 1e-4 * K * K));
  if (cw == 1) {
    // The currents switched to this page while running: its light comes right behind the old
    // page's, each mode taking over as the old one's last wavefront passes, broad shapes first.
    tau = 0.0;
    env = smoothstep(cross - 0.15, cross + 0.15, age);
    if (off < 1e8) env *= 1.0 - smoothstep(cross - 0.15, cross + 0.15, age - off);
  } else if (age < T) {
    // Light from the current sheets above and below needs sheet s to reach the slice.
    env = smoothstep(0.5 * sheet, 1.5 * sheet, age);
  } else if (off < T) {
    // Let go while still on its way: the pulse already sent keeps going, focuses, and runs on
    // freely, out of the slice in z and out of the screen in x and y.
    tau = T - age;
  } else {
    // Arrived and held: from T on the sheets send it continuously at the carrier frequency, each
    // mode q as waves crossing the slice at k_z = sqrt(K^2 - q^2). Nothing is stored; once the
    // currents stop, the field lasts until the last wavefront they sent has crossed the gap,
    // whose edge moves along z at c k_z / K: broad shapes go first, fine detail (grazing) last.
    tau = 0.0;
    env = 1.0 - smoothstep(cross - 0.15, cross + 0.15, age - off);
  }
  // Moved a whole screen away in x or y it is all in the padding, off the screen: drop it there,
  // before the periodic transform could bring it back. Whatever of a released field is left
  // after dmax goes too.
  vec2 v = c * abs(k) / max(m, 1e-6);
  float te = min(3.14159265 / kscale.x / max(v.x, 1e-6), 3.14159265 / kscale.y / max(v.y, 1e-6));
  if (tau < 0.0) te = min(te, dmax);
  env *= 1.0 - smoothstep(te - 1.0, te, abs(tau));
  vec2 f0, f2;
  tilt(m, tau, f0, f2);
  f0 *= env; f2 *= env;
  vec4 z = texelFetch(Z, ivec3(n, 2 * la + part), 0);
  float M = sqrt(float(4 * N.x * N.y));
  if (part == 1) {
    float s2 = sig * sig;
    o = vec4(conj(cmul((f0 - s2 * f2) / (1.0 - s2), z.xy)), 0.0, 0.0) / M;
    return;
  }
  vec2 q = m > 0.0 ? k / m : vec2(0.0);
  vec2 tx = z.xy, ty = z.zw;
  vec2 l = q.x * tx + q.y * ty;                                          // along q
  vec2 sx = cmul(f0, tx) + q.x * cmul(f2 - f0, l), sy = cmul(f0, ty) + q.y * cmul(f2 - f0, l);
  o = vec4(conj(sx), conj(sy)) / M;
}`;

// One radix-2 Stockham pass on two complex numbers per texel.
const FS_FFT = HEAD + `
uniform sampler2D src;
uniform int size, sub, horiz;
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int idx = horiz == 1 ? p.x : p.y;
  int hs = sub / 2;
  int ev = (idx / sub) * hs + (idx % hs);
  int od = ev + size / 2;
  vec4 e = texelFetch(src, horiz == 1 ? ivec2(ev, p.y) : ivec2(p.x, ev), 0);
  vec4 q = texelFetch(src, horiz == 1 ? ivec2(od, p.y) : ivec2(p.x, od), 0);
  float a = -6.283185307179586 * float(idx % sub) / float(sub);
  vec2 w = vec2(cos(a), sin(a));
  o = e + vec4(cmul(w, q.xy), cmul(w, q.zw));
}`;

// Inverse FFT -> the complex field on the screen (one part).
const FS_UNPACK = HEAD + `
uniform sampler2D F;
void main() {
  vec4 g = texelFetch(F, ivec2(gl_FragCoord.xy), 0);
  o = vec4(g.x, -g.y, g.z, -g.w);
}`;

// The slice now: every drive change still on it, each read where the page has moved it since it
// was made, added as complex fields (they interfere), then shown as amplitude.
const FS_COMPOSITE = HEAD + `
uniform sampler2DArray F, G;   // each change now, and once arrived (layers 2 slot, 2 slot + 1)
uniform ivec2 N;
uniform vec3 sgn;              // ink lighter (+1) or darker (-1) than the background
uniform int count;
uniform int slot[${SLOTS}];
uniform int shift[${SLOTS}];   // rows the page has moved since each was made
uniform float held[${SLOTS}];  // 1: arrived and held, its field is G
uniform float gpart[${SLOTS}]; // how much of its arrived field to use where its waves are cut off
uniform float coming[${SLOTS}];// 1: still on its way in, from the margin currents
uniform vec2 margin;          // grid x where the text column starts and ends
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec4 a = vec4(0.0); vec2 b = vec2(0.0);
  // The margin currents send their waves inward only: none outside them, building up across them.
  float x = float(n.x) + 0.5;
  float inside = smoothstep(0.0, margin.x, x) * (1.0 - smoothstep(margin.y, float(N.x), x));
  for (int j = 0; j < ${SLOTS}; j++) {
    if (j >= count) break;
    int y = n.y + shift[j];
    if (y < 0 || y >= N.y) continue;
    ivec3 q0 = ivec3(n.x, y, 2 * slot[j]), q1 = q0 + ivec3(0, 0, 1);
    vec4 ga = texelFetch(G, q0, 0); vec2 gb = texelFetch(G, q1, 0).xy;
    if (held[j] > 0.5) { a += ga; b += gb; continue; }
    // Waves were made in this change's own frame; where the page has moved that frame's edge
    // onto the screen, let them fade out instead of stopping at a line.
    float edge = shift[j] > 0 ? float(N.y - 1 - y) : shift[j] < 0 ? float(y) : 1e9;
    float t = smoothstep(0.0, 40.0, edge);
    float w = mix(1.0, inside, coming[j]);
    a += w * (t * texelFetch(F, q0, 0) + (1.0 - t) * gpart[j] * ga);
    b += w * (t * texelFetch(F, q1, 0).xy + (1.0 - t) * gpart[j] * gb);
  }
  o = vec4(sgn * vec3(length(a.xy), length(a.zw), length(b)), 0.0);
}`;

// The screen. The field between grid points is what the band-limited mode sum gives there, so
// it is reconstructed with a Lanczos-3 (windowed sinc) kernel rather than bilinearly (the
// amplitude's band is twice the field's, and the grid is sized for it).
const FS_DISPLAY = HEAD + `
uniform sampler2D F;           // background-relative colour on the grid
uniform ivec2 N;
uniform vec2 res;
uniform vec3 bg;
uniform float fy;              // fraction of a grid row the page is past the composite
vec3 E(int x, int y) {
  return texelFetch(F, clamp(ivec2(x, y), ivec2(0), N - 1), 0).rgb;
}
float lz(float x) {
  x = abs(x);
  if (x < 1e-4) return 1.0;
  if (x >= 3.0) return 0.0;
  float p = 3.14159265 * x;
  return 3.0 * sin(p) * sin(p / 3.0) / (p * p);
}
void main() {
  vec2 g = vec2(gl_FragCoord.x / res.x * float(N.x), (1.0 - gl_FragCoord.y / res.y) * float(N.y) + fy);
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
  e /= sx * sy;
  o = vec4(clamp(bg + e, 0.0, 1.0), 1.0);
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
    extRaw: program(FS_EXT_RAW), spec: program(FS_SPEC), evolve: program(FS_EVOLVE), fft: program(FS_FFT),
    gsInit: program(FS_GS_INIT), gsBand: program(FS_GS_BAND), gsPhase: program(FS_GS_PHASE),
    unpack: program(FS_UNPACK), composite: program(FS_COMPOSITE), disp: program(FS_DISPLAY),
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
let Lx = 0, Ly = 0, Nx = 0, Ny = 0;
let wavelength = WAVELENGTH;
let margins = [0, 0];   // CSS px: where the text column starts and ends (the drive currents fill the rest)
const phys = { kc: 0, K: 0, c: 0, done: 0, swap: 0 };
let R = null;
// Every change the drives have made that is still on the slice, oldest first: its page was
// captured with the page scrolled to grid row `row`, it started at t, and `off` is when a
// newer change let go of it (Infinity: it is the one being driven). Its spectrum is in R.Z,
// its field this frame in R.F and once arrived in R.G, all at layers 2 slot (E_x, E_y) and
// 2 slot + 1 (E_z).
let steps = [];        // { row, t: ms, slot, off: ms }
let raf = 0, moved = false, lastStep = -1e9, colors = null;

function pow2AtLeast(n) { return 1 << Math.ceil(Math.log2(Math.max(2, n))); }

function freeResources() {
  if (!R) return;
  for (const r of [...R.ext, ...R.gs[0], ...R.gs[1], R.comp]) { gl.deleteTexture(r.t); gl.deleteFramebuffer(r.f); }
  for (const s of [R.T, R.Z, R.F, R.G]) { gl.deleteTexture(s.t); s.fbos.forEach(f => gl.deleteFramebuffer(f)); }
  R = null;
}

function allocate() {
  freeResources();
  if (2 * Math.max(Nx, Ny) > gl.getParameter(gl.MAX_TEXTURE_SIZE)) throw new Error('grid too large');
  R = {
    ext: [target(2 * Nx, 2 * Ny), target(2 * Nx, 2 * Ny)],
    gs: [0, 1].map(() => [target(Nx, Ny), target(Nx, Ny)]),   // phase fit, per part
    T: layers(Nx, Ny, 1),                        // the target page (|colour - background|)
    Z: layers(2 * Nx, 2 * Ny, 2 * SLOTS),        // spectrum of each change, / sqrt(M)
    F: layers(Nx, Ny, 2 * SLOTS),                // each change on the slice, this frame
    G: layers(Nx, Ny, 2 * SLOTS),                // ... and once arrived
    comp: target(Nx, Ny, true),                  // the slice now, moved
  };
  steps = [];                                    // drives off
}

// The field's band limit for a carrier wavelength (K a little above it, so that every mode the
// sheets hold crosses the slice at a real angle).
const bandFor = lam => 2 * Math.PI / lam / Math.sqrt(1 + EPS);
function setPhysics() {
  // The amplitude has twice the field's band; the grid is sized for it, or caps it if too coarse.
  const kc = Math.min(bandFor(wavelength), 0.5 * RHO * Math.PI * Math.min(Nx / Lx, Ny / Ly));
  const K = Math.sqrt(1 + EPS) * kc;
  const c = 0.95 * Lx / T_ARRIVE;                  // the fastest waves cross the width in T
  // done: a pulse has arrived; swap: every mode of a switch-over has crossed (finest last).
  Object.assign(phys, { kc, K, c, done: T_ARRIVE + 0.05, swap: SHEET * K / Math.sqrt(K * K - kc * kc) + 0.2 });
}

// ---------------------------------------------------------------- page -> drive
const rc = document.createElement('canvas');
const ctx = rc.getContext('2d', { willReadFrequently: true });
const range = document.createRange();
let rasterBuf = null;

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

// The page as it would look scrolled to grid row `row` (within a row of where it is), on the grid.
function rasterize(row) {
  rc.width = Nx; rc.height = Ny;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = colors.bgCss;
  ctx.fillRect(0, 0, Nx, Ny);
  ctx.setTransform(Nx / Lx, 0, 0, Ny / Ly, 0.5, 0.5 + (scrollY - row * Ly / Ny) * Ny / Ly);   // grid point n at x = n Lx / Nx
  paintText(ctx, Ly);
  const img = ctx.getImageData(0, 0, Nx, Ny).data;
  if (!rasterBuf || rasterBuf.length !== Nx * Ny * 4) rasterBuf = new Float32Array(Nx * Ny * 4);
  const [br, bgc, bb] = colors.bg;
  for (let i = 0, n = Nx * Ny; i < n; i++) {        // the amplitude is |colour - background|
    rasterBuf[i * 4] = Math.abs(img[i * 4] / 255 - br);
    rasterBuf[i * 4 + 1] = Math.abs(img[i * 4 + 1] / 255 - bgc);
    rasterBuf[i * 4 + 2] = Math.abs(img[i * 4 + 2] / 255 - bb);
  }
  return rasterBuf;
}

// Forward FFT in place over the ping-pong pair (w x h), starting from pair[a]; the index of the result.
function fft(pair, w, h, a) {
  for (const [horiz, size] of [[1, w], [0, h]]) {
    for (let sub = 2; sub <= size; sub *= 2) {
      run(P.fft, pair[1 - a].f, w, h, { src: pair[a].t }, u => {
        gl.uniform1i(u.size, size); gl.uniform1i(u.sub, sub); gl.uniform1i(u.horiz, horiz);
      });
      a = 1 - a;
    }
  }
  return a;
}
function fft2() { return R.ext[fft(R.ext, 2 * Nx, 2 * Ny, 0)].t; }

function rowNow() { return Math.round(scrollY * Ny / Ly); }

// Capture the page at grid row `row` as target layer tl.
function capture(tl, row) {
  const data = rasterize(row);
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, R.T.t);
  gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, tl, Nx, Ny, 1, gl.RGBA, gl.FLOAT, data);
}

// The sheets' field for the captured page (R.T layer 0, captured at grid row `row`): its phase
// fitted so that its amplitude comes as close to the page as the band allows, then its
// band-limited spectrum into R.Z at the slot's two layers.
function coefFor(slot, row) {
  const T = [gl.TEXTURE_2D_ARRAY, R.T.t];
  for (const part of [0, 1]) {
    const pair = R.gs[part];
    run(P.gsInit, pair[0].f, Nx, Ny, { T }, u => { gl.uniform1i(u.part, part); gl.uniform1i(u.row, row); });
    let a = 0;
    for (let it = 0; it < GS_ITERS; it++) {
      a = fft(pair, Nx, Ny, a);
      run(P.gsBand, pair[1 - a].f, Nx, Ny, { X: pair[a].t }, u => {
        gl.uniform2i(u.N, Nx, Ny);
        gl.uniform2f(u.ks, 2 * Math.PI / Lx, 2 * Math.PI / Ly);
        gl.uniform1f(u.kc, phys.kc);
      });
      a = fft(pair, Nx, Ny, 1 - a);
      run(P.gsPhase, pair[1 - a].f, Nx, Ny, { Y: pair[a].t, T }, u => gl.uniform1i(u.part, part));
      a = 1 - a;
    }
    run(P.extRaw, R.ext[0].f, 2 * Nx, 2 * Ny, { U: pair[a].t }, u => gl.uniform2i(u.N, Nx, Ny));
    const G = fft2();
    run(P.spec, R.Z.fbos[2 * slot + part], 2 * Nx, 2 * Ny, { G }, u => {
      gl.uniform2i(u.N, Nx, Ny);
      gl.uniform2f(u.kscale, Math.PI / Lx, Math.PI / Ly);
      gl.uniform1f(u.kc, phys.kc);
    });
  }
}

// The change in `slot`, `age` seconds after it started, into the slot's layers of `dst` (R.F or R.G).
function fieldOf(slot, age, dst, off = 1e9, cw = false) {
  for (const part of [0, 1]) {
    run(P.evolve, R.ext[0].f, 2 * Nx, 2 * Ny, { Z: [gl.TEXTURE_2D_ARRAY, R.Z.t] }, u => {
      gl.uniform2i(u.N, Nx, Ny);
      gl.uniform1i(u.la, slot); gl.uniform1i(u.part, part);
      gl.uniform2f(u.kscale, Math.PI / Lx, Math.PI / Ly);
      gl.uniform1f(u.c, phys.c); gl.uniform1f(u.K, phys.K); gl.uniform1f(u.age, age);
      gl.uniform1f(u.T, T_ARRIVE); gl.uniform1f(u.sheet, SHEET);
      gl.uniform1f(u.sig, SIGMA); gl.uniform1f(u.dmax, D_MAX); gl.uniform1f(u.off, off);
      gl.uniform1i(u.cw, cw ? 1 : 0);
    });
    const F = fft2();
    run(P.unpack, dst.fbos[2 * slot + part], Nx, Ny, { F });
  }
}

const ageOf = (s, now) => (now - s.t) / 1000;
const isCur = (s, now) => s.off > now;
const doneOf = s => s.cw ? phys.swap : phys.done;
const isHeld = (s, now) => isCur(s, now) && ageOf(s, now) > doneOf(s);
const offOf = s => s.off === Infinity ? 1e9 : (s.off - s.t) / 1000;
// Let go of, and every mode out through the ends (or dropped) by now.
// (A steady field, or one switched over, is gone once its last mode has crossed.)
const isGone = (s, now) => !isCur(s, now) && (s.cw || offOf(s) >= T_ARRIVE
  ? now - s.off > 1000 * phys.swap + 50
  : now - Math.max(s.off, s.t + 1000 * T_ARRIVE) > 1000 * D_MAX + 50);

// Switch the drives to the current page. False if there is no room yet (try again later).
function changeDrive(now, settled) {
  steps = steps.filter(s => !isGone(s, now));
  const row = rowNow();
  if (settled || reduce.matches) {          // jump straight to the standing wave
    const s = { row, t: now - 1e5, slot: 0, off: Infinity, cw: false };
    capture(0, row);
    coefFor(0, row);
    fieldOf(0, 1e6, R.G);
    steps = [s];
  } else {
    const last = steps[steps.length - 1];
    const used = new Set(steps.map(s => s.slot));
    let slot = 0;
    while (used.has(slot)) slot++;
    const young = last && isCur(last, now) && ageOf(last, now) < 0.5 * doneOf(last);
    if (young || (slot >= SLOTS && last && isCur(last, now) && !isHeld(last, now))) {
      // The change being driven has only just started (or there is no room for another): it
      // takes the latest page and keeps its start time.
      last.row = row;
      capture(0, row);
      coefFor(last.slot, row);
      fieldOf(last.slot, 1e6, R.G);
    } else if (slot >= SLOTS) {
      return false;
    } else {
      // With a drive running, the currents switch over and the new light follows the old;
      // from nothing (first load, new wavelength), the page is sent in as a focusing pulse.
      const cw = steps.some(o => isCur(o, now));
      for (const o of steps) if (isCur(o, now)) o.off = now;   // let go of what it held
      const s = { row, t: now, slot, off: Infinity, cw };
      capture(0, row);
      coefFor(slot, row);                   // the whole page, sent in by the currents
      fieldOf(slot, 1e6, R.G);
      steps.push(s);
    }
  }
  lastStep = now;
  return true;
}

// The slice with the page at grid row rowC (fields of changes not held already in R.F).
function compositeInto(fbo, rowC, now) {
  run(P.composite, fbo, Nx, Ny, { F: [gl.TEXTURE_2D_ARRAY, R.F.t], G: [gl.TEXTURE_2D_ARRAY, R.G.t] }, u => {
    gl.uniform2i(u.N, Nx, Ny);
    gl.uniform3fv(u.sgn, colors.sgn);
    const slot = new Int32Array(SLOTS), shift = new Int32Array(SLOTS);
    const held = new Float32Array(SLOTS), gpart = new Float32Array(SLOTS), coming = new Float32Array(SLOTS);
    steps.forEach((s, i) => {
      slot[i] = s.slot; shift[i] = rowC - s.row;
      held[i] = isHeld(s, now) ? 1 : 0;
      const a = ageOf(s, now), x = Math.min(1, Math.max(0, (a - 0.75 * T_ARRIVE) / (phys.done - 0.75 * T_ARRIVE)));
      gpart[i] = isCur(s, now) && !s.cw ? x * x * (3 - 2 * x) : 0;
      coming[i] = !s.cw && a < T_ARRIVE ? 1 : 0;
    });
    gl.uniform1i(u.count, steps.length);
    gl.uniform1iv(u.slot, slot); gl.uniform1iv(u.shift, shift);
    gl.uniform1fv(u.held, held); gl.uniform1fv(u.gpart, gpart); gl.uniform1fv(u.coming, coming);
    gl.uniform2f(u.margin, margins[0] * Nx / Lx, margins[1] * Nx / Lx);
  });
}

let fy = 0;            // fraction of a row the page is past the composite
function composite(now) {
  const rows = scrollY * Ny / Ly, rowC = Math.floor(rows);
  fy = rows - rowC;
  steps = steps.filter(s => !isGone(s, now));
  for (const s of steps) if (!isHeld(s, now)) fieldOf(s.slot, ageOf(s, now), R.F, offOf(s), s.cw);
  compositeInto(R.comp.f, rowC, now);
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
  const fg = hex(v('--fg'));
  colors.sgn = colors.bg.map((b, i) => fg[i] >= b ? 1 : -1);   // ink lighter or darker than the background
}

function draw() {
  run(P.disp, null, canvas.width, canvas.height, { F: R.comp.t }, u => {
    gl.uniform2i(u.N, Nx, Ny);
    gl.uniform2f(u.res, canvas.width, canvas.height);
    gl.uniform3fv(u.bg, colors.bg);
    gl.uniform1f(u.fy, fy);
  });
}

function hud(now) {
  const last = steps[steps.length - 1];
  const since = last ? ageOf(last, now) : Infinity;
  const arrived = Math.min(1, since / (last ? doneOf(last) : 1));
  hudText.textContent =
    `(R, G, B) = background ± (|Eₓ|, |E_y|, |E_z|) on the slice z = 0 · wavelength ${Math.round(2 * Math.PI / phys.K)} px · ` +
    `${arrived >= 1 ? 'standing' : `${Math.round(100 * arrived)}% arrived`}`;
}

// ---------------------------------------------------------------- animation
function loop() { if (!raf) raf = requestAnimationFrame(tick); }

let lastFrame = 0, slowFrames = 0;
function tick(now) {
  raf = 0;
  if (!R) return;                             // waves switched off
  if (lastFrame && now - lastFrame < 250) {   // adaptive resolution for slow GPUs
    slowFrames = now - lastFrame > 30 ? slowFrames + 1 : Math.max(0, slowFrames - 1);
    if (slowFrames > 24 && Nx * Ny > (1 << 18)) {
      slowFrames = 0;
      maxPoints = (Nx * Ny) >> 1;
      layout();
      changeDrive(now, true);
    }
  }
  let more = false;
  if (moved) {
    if (now - lastStep >= STEP_GAP && changeDrive(now)) moved = false;
    else more = true;
  }
  composite(now);
  draw();
  hud(now);
  if (steps.some(s => !isHeld(s, now))) more = true;   // until settled
  lastFrame = more ? now : 0;
  if (more) loop();
}

addEventListener('scroll', () => { if (R) { moved = true; loop(); } }, { passive: true });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (!R) return;
  readColors();
  changeDrive(performance.now());   // a new palette is a new target
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
function layout() {
  Lx = canvas.clientWidth; Ly = canvas.clientHeight;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(Lx * dpr);
  canvas.height = Math.round(Ly * dpr);
  const cap = Math.min(2048, pow2AtLeast(+params.get('grid') || 2048));   // ?grid=512 caps resolution
  const pow2Near = v => 1 << Math.max(1, Math.round(Math.log2(v)));
  const density = 2 * bandFor(wavelength) / (RHO * Math.PI);      // grid points per CSS px
  let nx = Math.min(cap, pow2Near(density * Lx)), ny = Math.min(cap, pow2Near(density * Ly));
  while (nx * ny > maxPoints) { if (nx / Lx > ny / Ly) nx /= 2; else ny /= 2; }
  Nx = nx; Ny = ny;
  const box = doc.getBoundingClientRect(), cs = getComputedStyle(doc);
  const left = box.left + parseFloat(cs.paddingLeft), right = box.right - parseFloat(cs.paddingRight);
  margins = [Math.max(1, left), Math.min(Lx - 1, right)];
  allocate();
  setPhysics();
}

// Waves on: the canvas draws the page, formed from scratch by the drives. Off: the plain page,
// with the simulation stopped and its GPU memory freed. The choice is remembered.
function setWaves(on, settled) {
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
  changeDrive(performance.now(), settled);
  loop();
}

// The wavelength slider: a new carrier is a new drive, so the page comes in again from scratch.
function showWavelength() {
  knobInput.value = String(wavelength);
  knobOut.textContent = `${R ? Math.round(2 * Math.PI / phys.K) : wavelength} px`;   // as used (the grid may cap it)
}
let knobTimer = 0;
function setWavelength(v) {
  wavelength = v;
  try { localStorage.setItem('wavelength', String(v)); } catch (e) { /* not stored */ }
  if (!R) return;
  try { layout(); } catch (e) { console.warn('waves: grid for this wavelength failed', e); return; }
  moved = false;
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
  setWaves(on, params.has('still'));   // the drives switch on at load
  enableGlide();

  let timer = 0;
  addEventListener('resize', () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (!R || (canvas.clientWidth === Lx && canvas.clientHeight === Ly)) return;
      layout();
      changeDrive(performance.now(), true);   // a new box: start already standing
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
