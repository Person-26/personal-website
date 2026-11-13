// The page as a 2D slice of a boundary-driven electromagnetic standing wave in a 4D box.
// Colour is the field: (R, G, B) - background = (E_x, E_y, E_z) on the slice.
//
// Box: x in [0,Lx], y in [0,Ly] (the screen), z, w hidden; the screen is the slice z = z0,
// w = w0. The top and bottom (y = 0, Ly) are conducting walls, so there E_x and E_z vanish:
//   E_x ~ sin(ky y),  E_y ~ cos(ky y),  E_z ~ sin(ky y).
// The z walls are far field: open, nothing comes in from them. The slice is a waveguide
// along x, and its left and right ends (x = 0, Lx) are open and driven.
//
// Nothing is drawn directly. The only inputs are those drives at one frequency Omega = c K,
// on the faces x = 0 and x = Lx. Each colour component rides its own carrier, leaning into
// the hidden dimensions by its own amount, with hidden wavenumber mu_c: Gauss's
// law is met by E_w (not displayed), so any colour below the band limit is reachable, and
// the drives are chosen so the standing wave on the slice is the page.
//
// In the slice a mode of in-plane wavevector k has omega_c(k) = c sqrt(k^2 + mu_c^2), so it
// moves at c k / omega_c and, against the drive, slips in phase at delta_c = omega_c - Omega.
// When the page changes, the drives send the change in from both ends, timed so that every
// mode falls into step T after the change: tau = T - age before that, a mode is delta_c tau
// out of step, and the slice holds the change evolved back by tau, which is waves that
// entered from the left and right, cross the screen, bounce off the top and bottom walls and
// interfere, focusing into the page at tau = 0. The leans differ, so R, G and B disperse
// differently and only coincide once settled. c is set so the fastest waves cross the width
// in T, so at the start of a change it is still outside the screen.
//
// The transforms: x is zero-padded to twice the width (open ends: what leaves goes into the
// padding, not back in), y is mirrored (odd for E_x, E_z; even for E_y: the walls). E_x + i E_y
// share one complex FFT; their spectra are separated with Z(-k) when each gets its own phase.
// The field scrolls with the page: every stored field keeps the grid row it was made at and is
// read shifted by how far the page has moved since. The box is open (left, right and the far
// field), so a standing wave only lasts while the drives feed it. A drive change stops feeding
// everything the old drive held, which then leaks away with time constant TAU_LOSS, and sends in
// the whole current page, moved text included: after a scroll the moved text fades and is
// rebuilt by the new waves along with whatever came into view.
// Moving a field cuts it at the top and bottom walls, which no field in the box can do: there
// the tangential components E_x, E_z must vanish. So after every move the field is projected
// back onto the box's modes in y (sine for E_x, E_z; cosine for E_y), band-limited. Away from
// the walls that changes nothing (the field is already band-limited), so it is done locally:
// a low-pass filter applied to the odd/even mirror image of the field across each wall.
// Every change in flight is transformed back each frame with its modes' current phases.
//
// The grid is coarser than the screen and band-limited, so the settled field is a little soft.
// Where the picture has formed, it hands over to the same page drawn at full screen resolution.
// "Formed" is judged on the images, not the field: the colours on the slice against the page
// they are heading for, by local structural similarity (SSIM) per colour channel, the worst
// channel counting. Locking in eases over LOCK_RISE; wherever the picture stops matching (new
// waves arriving) it eases back to the simulation over LOCK_FALL.
(() => {
'use strict';

const RHO = 0.8;           // band limit: fraction of the grid's Nyquist wavenumber kept
const DENSITY = 0.55;      // grid points per CSS px wanted (rounded to a power of two)
const EPS = 0.06;          // how close the finest in-slice mode sits to cutoff
// Each component's carrier has its own hidden wavenumber mu_c (a fraction of K): it acts as a
// mass in the slice, so the colours disperse differently. Kept small so that even broad
// shapes travel at close to c and really arrive from the sides.
const MU = [0.08, 0.2, 0.34];  // E_x, E_y, E_z
const T_ARRIVE = 4.0;      // s: from a drive change to the slice holding it
const TAU_LOSS = 2.0;      // s: an unfed standing wave leaks out of the open box
const T_LATE = 1.0;        // s: the most a slow mode may arrive after T_ARRIVE
const TAPER = 0.75;        // the band limit rolls off smoothly from this fraction of it
const SLEW = 0.6;          // s: time constant with which the drives take up a change
const IN_FLIGHT = 2;       // drive changes travelling at once
const STEP_GAP = 220;      // ms: at most one drive change per this interval while scrolling
const LOCK_LO = 0.85, LOCK_HI = 0.97;   // SSIM over which an area goes from unlocked to locked
const LOCK_RISE = 1.5;     // s: fade from the simulation to the full-resolution page
const LOCK_FALL = 0.4;     // s: and back
const WALL = 16;           // rows: half-width of the filter that makes moved fields meet the walls
const MAX_POINTS = 1 << 19;
let maxPoints = MAX_POINTS;

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

// ---------------------------------------------------------------- shaders
const HEAD = '#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\nprecision highp sampler2DArray;\nout vec4 o;\n';
const VS = '#version 300 es\nin vec2 p;\nvoid main(){ gl_Position = vec4(p, 0.0, 1.0); }';

// Target page colour (minus background), or the new target minus the old one moved by the
// scroll in between, extended: zero-padded in x, mirrored in y (E_x, E_z odd; E_y even).
// Packed as two complex numbers: (E_x + i E_y, E_z).
const FS_EXT_RAW = HEAD + `
uniform ivec2 N;
uniform sampler2DArray T;
uniform int la, lb, shift;     // T[la](n) - T[lb](n + shift rows); lb < 0 means just T[la]
vec3 src(ivec2 n) {
  vec3 a = texelFetch(T, ivec3(n, la), 0).rgb;
  int y = n.y + shift;
  if (lb >= 0 && y >= 0 && y < N.y) a -= texelFetch(T, ivec3(n.x, y, lb), 0).rgb;
  return a;
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (p.x >= N.x) { o = vec4(0.0); return; }
  int y = p.y; float s = 1.0;
  if (y > N.y) { y = 2 * N.y - y; s = -1.0; }
  vec3 v = src(ivec2(p.x, min(y, N.y - 1)));
  float odd = (y == 0 || y == N.y) ? 0.0 : s;
  o = vec4(odd * v.r, v.g, odd * v.b, 0.0);
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

// A change's spectrum with each mode's current phase slip, ready for the inverse FFT
// (conjugated and scaled: IFFT(X) = conj(FFT(conj X)) / M).
const FS_EVOLVE = HEAD + KVEC + `
uniform sampler2DArray Z;
uniform int la;
uniform float c, K, age, T, late, slew;
uniform vec3 mu;               // per component: carrier hidden wavenumber
vec2 conj(vec2 z) { return vec2(z.x, -z.y); }
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec2 k = kvec(n);
  vec3 om = sqrt(dot(k, k) + mu * mu);                    // omega / c
  // A mode crosses at c kx / omega; one too slow to reach the middle from the sides within T
  // arrives when it can (at most late s after T) and only appears then.
  vec3 travel = 0.5 * 3.14159265 / kscale.x * om / (c * max(abs(k.x), 1e-6));
  vec3 Tn = clamp(travel, T, T + late);
  vec3 tau = max(Tn - age, 0.0);                          // time left until it is in step
  vec3 slip = c * (om - K) * tau;
  vec3 env = mix(1.0 - exp(-age / slew) * tau / Tn,       // entered from outside the screen
                 smoothstep(Tn - 1.0, Tn, vec3(age)),      // slow: appears as it arrives
                 step(T + 1e-3, Tn));
  vec3 w = cos(slip) * env;
  vec4 z = texelFetch(Z, ivec3(n, la), 0);
  vec2 zm = texelFetch(Z, ivec3((2 * N - n) % (2 * N), la), 0).xy;    // Z1(-k)
  vec2 x1 = 0.5 * ((w.x + w.y) * z.xy + (w.x - w.y) * conj(zm));  // E_x, E_y separated
  vec2 x2 = w.z * z.zw;
  o = vec4(conj(x1), conj(x2)) / sqrt(float(4 * N.x * N.y));
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

// Inverse FFT -> field (E_x, E_y, E_z) on the screen.
const FS_UNPACK = HEAD + `
uniform sampler2D F;
void main() {
  vec4 g = texelFetch(F, ivec2(gl_FragCoord.xy), 0);
  o = vec4(g.x, -g.y, g.z, 0.0);
}`;

// The slice now: the settled field plus every drive change still travelling, each read where
// the page has moved it since it was made.
const FS_COMPOSITE = HEAD + `
uniform sampler2D fed, fade;   // settled field the drive holds; unfed fields leaking away
uniform sampler2DArray F, G;   // each change now, and once arrived
uniform ivec2 N;
uniform int goal;              // 1: what it will settle to instead of now
uniform int fedShift, fadeShift;   // rows the page has moved since each was made
uniform float fedAmp, fadeAmp;
uniform int count;             // drive changes in flight
uniform int slot[${IN_FLIGHT}];
uniform int shift[${IN_FLIGHT}];
uniform float age[${IN_FLIGHT}];
uniform float amp[${IN_FLIGHT}];   // 1 while its drive is on, then leaking away
uniform float arrive, done;
vec3 at(sampler2D t, int s, ivec2 n) {
  int y = n.y + s;
  return y >= 0 && y < N.y ? texelFetch(t, ivec2(n.x, y), 0).rgb : vec3(0.0);
}
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec3 e = fedAmp * at(fed, fedShift, n);
  if (goal == 0) e += fadeAmp * at(fade, fadeShift, n);
  for (int j = 0; j < ${IN_FLIGHT}; j++) {
    if (j >= count) break;
    int y = n.y + shift[j];
    if (y < 0 || y >= N.y) continue;
    ivec3 q = ivec3(n.x, y, slot[j]);
    vec3 g = texelFetch(G, q, 0).rgb;
    if (goal == 1) { e += amp[j] < 1.0 ? vec3(0.0) : g; continue; }
    // The waves still in flight were made in this change's own frame; where the page has moved
    // that frame's edge onto the screen, let them fade out instead of stopping at a line
    // (its arrived part comes in there as it arrives).
    float edge = shift[j] > 0 ? float(N.y - 1 - y) : shift[j] < 0 ? float(y) : 1e9;
    float t = smoothstep(0.0, 40.0, edge);
    e += amp[j] * (t * texelFetch(F, q, 0).rgb + (1.0 - t) * smoothstep(0.75 * arrive, done, age[j]) * g);
  }
  o = vec4(e, 0.0);
}`;

// Projection of a moved field onto the box's modes in y, near the top and bottom walls.
const FS_WALLS = HEAD + `
uniform sampler2D F;
uniform ivec2 N;
uniform float h[${WALL + 1}];  // symmetric low-pass kernel, cutoff above the band limit
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec4 e0 = texelFetch(F, n, 0);
  if (n.y >= ${WALL} && n.y < N.y - ${WALL}) { o = e0; return; }
  vec3 e = vec3(0.0);
  for (int k = -${WALL}; k <= ${WALL}; k++) {
    int y = n.y + k;
    float s = 1.0;                        // sign of the odd (sine) components' mirror image
    if (y < 0) { y = -y; s = -1.0; }
    if (y > N.y) { y = 2 * N.y - y; s = -1.0; }
    float odd = (y == 0 || y == N.y) ? 0.0 : s;   // E_x, E_z vanish on the walls
    vec3 v = texelFetch(F, ivec2(n.x, min(y, N.y - 1)), 0).rgb;
    e += h[k < 0 ? -k : k] * vec3(odd * v.r, v.g, odd * v.b);
  }
  o = vec4(e, e0.a);
}`;

// Has the picture formed here? Local SSIM between the colours on the slice and the colours
// it is heading for, over a 13 x 13 cell window, per channel; the worst channel counts.
const FS_SSIM = HEAD + `
uniform sampler2D S, P;        // field now; settled field
uniform ivec2 N;
uniform vec3 bg;
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  vec3 mx = vec3(0.0), my = vec3(0.0), xx = vec3(0.0), yy = vec3(0.0), xy = vec3(0.0);
  for (int j = -3; j <= 3; j++) for (int i = -3; i <= 3; i++) {
    ivec2 q = clamp(n + 2 * ivec2(i, j), ivec2(0), N - 1);
    vec3 x = clamp(bg + texelFetch(S, q, 0).rgb, 0.0, 1.0);
    vec3 y = clamp(bg + texelFetch(P, q, 0).rgb, 0.0, 1.0);
    mx += x; my += y; xx += x * x; yy += y * y; xy += x * y;
  }
  mx /= 49.0; my /= 49.0;
  vec3 vx = max(xx / 49.0 - mx * mx, 0.0), vy = max(yy / 49.0 - my * my, 0.0), cxy = xy / 49.0 - mx * my;
  const float C1 = 0.0001, C2 = 0.0009;
  vec3 ssim = (2.0 * mx * my + C1) * (2.0 * cxy + C2) / ((mx * mx + my * my + C1) * (vx + vy + C2));
  o = vec4(min(ssim.r, min(ssim.g, ssim.b)), 0.0, 0.0, 0.0);
}`;

// How locked each grid point is to the full-resolution page: moved with the scroll, then
// eased toward how well the picture has formed there.
const FS_LOCK = HEAD + `
uniform sampler2D A, Q;        // previous lock; SSIM this frame
uniform ivec2 N;
uniform int shift;             // rows moved since the previous lock
uniform float rise, fall;      // largest change this frame
void main() {
  ivec2 n = ivec2(gl_FragCoord.xy);
  int y = n.y + shift;
  float a = y >= 0 && y < N.y ? texelFetch(A, ivec2(n.x, y), 0).r : 0.0;
  float want = smoothstep(${LOCK_LO}, ${LOCK_HI}, texelFetch(Q, n, 0).r);
  o = vec4(a + clamp(want - a, -fall, rise), 0.0, 0.0, 0.0);
}`;

// The screen. The field between grid points is what the band-limited mode sum gives there, so
// it is reconstructed with a Lanczos-3 (windowed sinc) kernel rather than bilinearly, across the
// walls by the field's own mirror symmetry (E_x, E_z odd; E_y even).
const FS_DISPLAY = HEAD + `
uniform sampler2D F, A, H;     // field, lock, full-resolution page
uniform ivec2 N;
uniform vec2 res;
uniform vec3 bg;
uniform float fy;              // fraction of a grid row the page is past the composite
uniform float hshift;          // device px the page has moved since H was drawn
vec3 E(int x, int y) {
  x = clamp(x, 0, N.x - 1);
  float s = 1.0;
  if (y < 0) { y = -y; s = -1.0; }
  if (y > N.y) { y = 2 * N.y - y; s = -1.0; }
  float odd = y == N.y ? 0.0 : s;
  vec3 v = texelFetch(F, ivec2(x, min(y, N.y - 1)), 0).rgb;
  return vec3(odd * v.r, v.g, odd * v.b);
}
float lz(float x) {
  x = abs(x);
  if (x < 1e-4) return 1.0;
  if (x >= 3.0) return 0.0;
  float p = 3.14159265 * x;
  return 3.0 * sin(p) * sin(p / 3.0) / (p * p);
}
float L(ivec2 i) { return texelFetch(A, clamp(i, ivec2(0), N - 1), 0).r; }
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
  vec3 sim = clamp(bg + e, 0.0, 1.0);
  float lock = mix(mix(L(i), L(i + ivec2(1, 0)), f.x), mix(L(i + ivec2(0, 1)), L(i + ivec2(1, 1)), f.x), f.y);
  vec2 q = vec2(gl_FragCoord.x, res.y - gl_FragCoord.y + hshift);
  lock = q.y >= 0.0 && q.y < res.y ? clamp(lock, 0.0, 1.0) : 0.0;
  vec3 hi = texture(H, q / res).rgb;
  o = vec4(mix(sim, hi, lock * lock * (3.0 - 2.0 * lock)), 1.0);
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
    unpack: program(FS_UNPACK), composite: program(FS_COMPOSITE), walls: program(FS_WALLS), ssim: program(FS_SSIM), lock: program(FS_LOCK), disp: program(FS_DISPLAY),
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
const phys = { kc: 0, K: 0, c: 0, mu: new Float32Array(3), done: 0 };
let R = null;
// The drive changes in flight, oldest first. Each has a target page (layer tl of R.T) captured
// with the page scrolled to grid row `row`; `off` is when a newer change stopped feeding it.
let steps = [];        // { tl, row, t: ms, slot, off: ms }
// R.fed: the arrived field the current drive holds (made at fedRow; fedOn: is there one).
// R.fade: fields no drive holds any more, leaking away: amplitude exp(-(now - fadeT) / TAU_LOSS).
let fedRow = 0, fedOn = false, fadeRow = 0, fadeT = -1e12;
let raf = 0, moved = false, lastStep = -1e9, colors = null;

function pow2AtLeast(n) { return 1 << Math.ceil(Math.log2(Math.max(2, n))); }

function freeResources() {
  if (!R) return;
  for (const r of [R.ext[0], R.ext[1], R.fed, R.fed2, R.fade, R.fade2, R.comp, R.comp2, R.goal, R.goal2, R.score, R.lock[0], R.lock[1]]) { gl.deleteTexture(r.t); gl.deleteFramebuffer(r.f); }
  for (const s of [R.T, R.Z, R.F, R.G]) { gl.deleteTexture(s.t); s.fbos.forEach(f => gl.deleteFramebuffer(f)); }
  gl.deleteTexture(R.hi);
  R = null;
}

function allocate() {
  freeResources();
  if (2 * Math.max(Nx, Ny) > gl.getParameter(gl.MAX_TEXTURE_SIZE)) throw new Error('grid too large');
  R = {
    ext: [target(2 * Nx, 2 * Ny), target(2 * Nx, 2 * Ny)],
    T: layers(Nx, Ny, IN_FLIGHT + 2),            // target pages (colour minus background)
    Z: layers(2 * Nx, 2 * Ny, IN_FLIGHT + 1),    // spectrum of each change (+ scratch), / sqrt(M)
    F: layers(Nx, Ny, IN_FLIGHT),                // each change on the slice, this frame
    G: layers(Nx, Ny, IN_FLIGHT),                // ... and once settled
    fed: target(Nx, Ny, true), fed2: target(Nx, Ny, true),     // arrived field the drive holds
    fade: target(Nx, Ny, true), fade2: target(Nx, Ny, true),   // unfed fields leaking away
    comp: target(Nx, Ny, true),                  // the slice now, moved
    comp2: target(Nx, Ny, true),                 // ... and meeting the walls
    goal: target(Nx, Ny, true),                  // the settled picture, moved
    goal2: target(Nx, Ny, true),                 // ... and meeting the walls
    score: target(Nx, Ny, true),                 // how well the picture has formed (SSIM)
    lock: [target(Nx, Ny, true), target(Nx, Ny, true)],   // handover to full resolution
    hi: hiTexture(),                             // the page at full resolution
  };
  steps = []; fedOn = false; fadeT = -1e12;           // drives off
}

function setPhysics() {
  const kc = RHO * Math.PI * Math.min(Nx / Lx, Ny / Ly);
  const K = Math.sqrt(1 + EPS) * kc;
  const mu = new Float32Array(MU.map(m => K * m));
  const c = 0.95 * Lx / T_ARRIVE;                  // the fastest waves cross the width in T
  Object.assign(phys, { kc, K, mu, c, done: T_ARRIVE + T_LATE + 0.05 });
}

// Blackman-windowed sinc with cutoff 0.8 of the grid's Nyquist wavenumber: flat up to the band
// limit's roll-off, so it leaves band-limited fields alone and only reshapes cuts.
const wallKernel = (() => {
  const fc = 0.93, h = new Float32Array(WALL + 1);
  let sum = 0;
  for (let k = 0; k <= WALL; k++) {
    const sinc = k === 0 ? fc : Math.sin(Math.PI * fc * k) / (Math.PI * k);
    const t = (k + WALL + 1) / (2 * WALL + 2);
    h[k] = sinc * (0.42 - 0.5 * Math.cos(2 * Math.PI * t) + 0.08 * Math.cos(4 * Math.PI * t));
    sum += k === 0 ? h[k] : 2 * h[k];
  }
  return h.map(v => v / sum);
})();
function walls(src, fbo) {
  run(P.walls, fbo, Nx, Ny, { F: src }, u => { gl.uniform2i(u.N, Nx, Ny); gl.uniform1fv(u.h, wallKernel); });
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
  for (let i = 0, n = Nx * Ny; i < n; i++) {        // the field is colour minus background
    rasterBuf[i * 4] = img[i * 4] / 255 - br;
    rasterBuf[i * 4 + 1] = img[i * 4 + 1] / 255 - bgc;
    rasterBuf[i * 4 + 2] = img[i * 4 + 2] / 255 - bb;
  }
  return rasterBuf;
}

// The same page at full screen resolution, for areas that have settled.
const hc = document.createElement('canvas');
const hctx = hc.getContext('2d');
let hiScroll = 0;
function hiTexture() {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  return t;
}
function captureHi() {
  hc.width = canvas.width; hc.height = canvas.height;
  hctx.setTransform(1, 0, 0, 1, 0, 0);
  hctx.fillStyle = colors.bgCss;
  hctx.fillRect(0, 0, hc.width, hc.height);
  hctx.setTransform(hc.width / Lx, 0, 0, hc.height / Ly, 0, 0);
  paintText(hctx, Ly);
  gl.bindTexture(gl.TEXTURE_2D, R.hi);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, hc);
  hiScroll = scrollY;
}

function fft2() {
  let a = 0;
  for (const [horiz, size] of [[1, 2 * Nx], [0, 2 * Ny]]) {
    for (let sub = 2; sub <= size; sub *= 2) {
      run(P.fft, R.ext[1 - a].f, 2 * Nx, 2 * Ny, { src: R.ext[a].t }, u => {
        gl.uniform1i(u.size, size); gl.uniform1i(u.sub, sub); gl.uniform1i(u.horiz, horiz);
      });
      a = 1 - a;
    }
  }
  return R.ext[a].t;
}

function rowNow() { return Math.round(scrollY * Ny / Ly); }

// Capture the page at grid row `row` as target layer tl.
function capture(tl, row) {
  const data = rasterize(row);
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, R.T.t);
  gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, tl, Nx, Ny, 1, gl.RGBA, gl.FLOAT, data);
}

// Band-limited spectrum (into layer `layer` of R.Z) of target `s`, or of its difference from
// the target `p` moved by the scroll between them.
function coefFor(s, p, layer) {
  run(P.extRaw, R.ext[0].f, 2 * Nx, 2 * Ny, { T: [gl.TEXTURE_2D_ARRAY, R.T.t] }, u => {
    gl.uniform2i(u.N, Nx, Ny);
    gl.uniform1i(u.la, s.tl); gl.uniform1i(u.lb, p ? p.tl : -1);
    gl.uniform1i(u.shift, p ? s.row - p.row : 0);
  });
  const G = fft2();
  run(P.spec, R.Z.fbos[layer], 2 * Nx, 2 * Ny, { G }, u => {
    gl.uniform2i(u.N, Nx, Ny);
    gl.uniform2f(u.kscale, Math.PI / Lx, Math.PI / Ly);
    gl.uniform1f(u.kc, phys.kc);
  });
}

// The change with spectrum in layer `layer`, `age` seconds after it started, into fbo.
function fieldOf(layer, age, fbo) {
  run(P.evolve, R.ext[0].f, 2 * Nx, 2 * Ny, { Z: [gl.TEXTURE_2D_ARRAY, R.Z.t] }, u => {
    gl.uniform2i(u.N, Nx, Ny);
    gl.uniform1i(u.la, layer);
    gl.uniform2f(u.kscale, Math.PI / Lx, Math.PI / Ly);
    gl.uniform1f(u.c, phys.c); gl.uniform1f(u.K, phys.K); gl.uniform1f(u.age, age);
    gl.uniform1f(u.T, T_ARRIVE); gl.uniform1f(u.late, T_LATE); gl.uniform1f(u.slew, SLEW);
    gl.uniform3fv(u.mu, phys.mu);
  });
  const F = fft2();
  run(P.unpack, fbo, Nx, Ny, { F });
}

const fadeAmp = now => Math.exp(-Math.max(0, now - fadeT) / 1000 / TAU_LOSS);
const ampOf = (s, now) => s.off > now ? 1 : Math.exp(-(now - s.off) / 1000 / TAU_LOSS);

// Store the field `parts` (see compositeInto) at the current row into R.fed or R.fade.
function store(into, parts, now) {
  const row = Math.floor(scrollY * Ny / Ly);
  compositeInto(R.comp.f, row, parts.live || [], { fed: parts.fed ? 1 : 0, fade: parts.fade ? fadeAmp(now) : 0, now: 1e15 });
  if (into === 'fed') { walls(R.comp.t, R.fed2.f); [R.fed, R.fed2] = [R.fed2, R.fed]; fedRow = row; }
  else { walls(R.comp.t, R.fade2.f); [R.fade, R.fade2] = [R.fade2, R.fade]; fadeRow = row; fadeT = now; }
}

// The oldest change in flight has arrived. If its drive is still on, it is now the field the
// drive holds; if not, it joins the fields leaking away.
function settleOldest(now) {
  const s = steps[0];
  fieldOf(s.slot, 1e6, R.F.fbos[s.slot]);
  if (s.off > now) { store('fed', { live: [s] }, now); fedOn = true; }
  else store('fade', { fade: true, live: [{ ...s, ampNow: ampOf(s, now) }] }, now);
  steps.shift();
}
const arrived = (s, now) => (now - s.t) / 1000 > phys.done;

// Switch the drives to the current page.
function changeDrive(now, settled) {
  while (steps.length && arrived(steps[0], now)) settleOldest(now);
  const row = rowNow();
  if (settled || reduce.matches) {          // jump straight to the standing wave
    const s = { tl: 0, row, t: -1e12, slot: -1 };
    capture(0, row);
    steps = [];
    coefFor(s, null, IN_FLIGHT);
    fieldOf(IN_FLIGHT, 1e6, R.fed.f);
    fedRow = row; fedOn = true; fadeT = -1e12;
    for (const l of R.lock) { gl.bindFramebuffer(gl.FRAMEBUFFER, l.f); gl.clearColor(1, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
    lockRow = Math.floor(scrollY * Ny / Ly);
  } else if (steps.length >= IN_FLIGHT && steps[steps.length - 1].off > now) {
    // Every slot is busy: the newest change, still the one being driven, takes the latest page
    // and keeps its start time.
    const s = steps[steps.length - 1];
    s.row = row;
    capture(s.tl, row);
    coefFor(s, null, s.slot);
    fieldOf(s.slot, 1e6, R.G.fbos[s.slot]);
  } else {
    // The old drive stops: what it held, arrived or on its way, starts leaking away.
    if (fedOn) { store('fade', { fed: true, fade: true }, now); fedOn = false; }
    for (const o of steps) if (o.off > now) o.off = now;
    while (steps.length >= IN_FLIGHT) settleOldest(now);
    const usedT = new Set(steps.map(s => s.tl)), usedS = new Set(steps.map(s => s.slot));
    let tl = 0, slot = 0;
    while (usedT.has(tl)) tl++;
    while (usedS.has(slot)) slot++;
    const s = { tl, row, t: now, slot, off: Infinity };
    capture(tl, row);
    coefFor(s, null, slot);                 // the whole page, sent in from the sides
    fieldOf(slot, 1e6, R.G.fbos[slot]);
    steps.push(s);
  }
  captureHi();
  lastStep = now;
}

// The slice with the page at grid row rowC: the held field, the fields leaking away, and the
// changes `live` (their fields already in R.F). opt: { goal, fed, fade (amplitude), now }.
function compositeInto(fbo, rowC, live, opt) {
  run(P.composite, fbo, Nx, Ny, { fed: R.fed.t, fade: R.fade.t, F: [gl.TEXTURE_2D_ARRAY, R.F.t], G: [gl.TEXTURE_2D_ARRAY, R.G.t] }, u => {
    gl.uniform2i(u.N, Nx, Ny);
    gl.uniform1i(u.goal, opt.goal ? 1 : 0);
    gl.uniform1i(u.fedShift, rowC - fedRow); gl.uniform1i(u.fadeShift, rowC - fadeRow);
    gl.uniform1f(u.fedAmp, opt.fed); gl.uniform1f(u.fadeAmp, opt.fade);
    const slot = new Int32Array(IN_FLIGHT), shift = new Int32Array(IN_FLIGHT);
    const age = new Float32Array(IN_FLIGHT), amp = new Float32Array(IN_FLIGHT);
    live.forEach((s, i) => {
      slot[i] = s.slot; shift[i] = rowC - s.row;
      age[i] = Math.min(1e6, (opt.now - s.t) / 1000);
      amp[i] = s.ampNow !== undefined ? s.ampNow : ampOf(s, opt.now);
    });
    gl.uniform1fv(u.age, age); gl.uniform1fv(u.amp, amp);
    gl.uniform1f(u.arrive, T_ARRIVE); gl.uniform1f(u.done, phys.done);
    gl.uniform1i(u.count, live.length);
    gl.uniform1iv(u.slot, slot);
    gl.uniform1iv(u.shift, shift);
  });
}

let fy = 0;            // fraction of a row the page is past the composite
let lockRow = 0, lockTime = 0, lockBusy = 0;
function composite(now) {
  const rows = scrollY * Ny / Ly, rowC = Math.floor(rows);
  fy = rows - rowC;
  const live = steps;
  for (const s of live) fieldOf(s.slot, (now - s.t) / 1000, R.F.fbos[s.slot]);
  const fa = fadeAmp(now);
  compositeInto(R.comp.f, rowC, live, { fed: fedOn ? 1 : 0, fade: fa > 0.003 ? fa : 0, now });
  walls(R.comp.t, R.comp2.f);
  // Judge where the picture has formed, and lock toward full resolution there. Not while a
  // drive change is pending (the full-resolution page is only redrawn with it), nor while a
  // change is in its first half, before its waves have reached everywhere they will pass.
  compositeInto(R.goal.f, rowC, live, { goal: true, fed: fedOn ? 1 : 0, fade: 0, now });
  walls(R.goal.t, R.goal2.f);
  run(P.ssim, R.score.f, Nx, Ny, { S: R.comp2.t, P: R.goal2.t }, u => {
    gl.uniform2i(u.N, Nx, Ny); gl.uniform3fv(u.bg, colors.bg);
  });
  const young = live.some(s => s.off > now && now - s.t < 500 * T_ARRIVE);
  const dt = lockTime ? Math.min(0.1, (now - lockTime) / 1000) : 0;
  run(P.lock, R.lock[1].f, Nx, Ny, { A: R.lock[0].t, Q: R.score.t }, u => {
    gl.uniform2i(u.N, Nx, Ny);
    gl.uniform1i(u.shift, rowC - lockRow);
    gl.uniform1f(u.rise, moved || young ? 0 : dt / LOCK_RISE); gl.uniform1f(u.fall, dt / LOCK_FALL);
  });
  R.lock.reverse();
  lockRow = rowC; lockTime = now;
  if (live.length || moved) lockBusy = now;
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
  run(P.disp, null, canvas.width, canvas.height, { F: R.comp2.t, A: R.lock[0].t, H: R.hi }, u => {
    gl.uniform2i(u.N, Nx, Ny);
    gl.uniform2f(u.res, canvas.width, canvas.height);
    gl.uniform3fv(u.bg, colors.bg);
    gl.uniform1f(u.fy, fy);
    gl.uniform1f(u.hshift, (scrollY - hiScroll) * canvas.height / Ly);
  });
}

function hud(now) {
  const live = steps.length;
  const since = live ? (now - steps[steps.length - 1].t) / 1000 : Infinity;
  const arrived = Math.min(1, since / phys.done);
  hudText.textContent =
    `(R, G, B) − background = (Eₓ, E_y, E_z)(x, y, z₀, w₀) · driven from the left and right edges · ` +
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
    if (now - lastStep >= STEP_GAP) { moved = false; changeDrive(now); }
    else more = true;
  }
  while (steps.length && arrived(steps[0], now)) settleOldest(now);
  composite(now);
  draw();
  hud(now);
  if (steps.length || fadeAmp(now) > 0.003 || now - lockBusy < 2 * LOCK_RISE * 1000) more = true;   // until settled and eased in
  lastFrame = more ? now : 0;
  if (!more) lockTime = 0;
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
  let nx = Math.min(cap, pow2Near(DENSITY * Lx)), ny = Math.min(cap, pow2Near(DENSITY * Ly));
  while (nx * ny > maxPoints) { if (nx / Lx > ny / Ly) nx /= 2; else ny /= 2; }
  Nx = nx; Ny = ny;
  allocate();
  setPhysics();
}

// Waves on: the canvas draws the page, formed from scratch by the drives. Off: the plain page,
// with the simulation stopped and its GPU memory freed. The choice is remembered.
function setWaves(on, settled) {
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
  moved = false; lastFrame = 0; lockTime = 0;
  changeDrive(performance.now(), settled);
  loop();
}

function start() {
  let on = true;
  try { on = localStorage.getItem('waves') !== 'off'; } catch (e) { /* default on */ }
  modeButton.hidden = false;
  modeButton.addEventListener('click', () => setWaves(!R));
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
});

(document.fonts ? document.fonts.ready : Promise.resolve()).then(start);
})();
