// A worker for waves.js: the response tables, off the page's thread (they take a few hundred ms).
// Each request: { id, tmax, bands, c } (c: waves.js's constants); the reply: { id, tables } (two
// Float32Arrays, transferred) or { id, error }. The tables come from waves.wasm (Rust, in wasm/:
// about twice as fast) or, if it did not load, from computeResponse here (the reference).
'use strict';

// For each frequency, band node (|kx| / k = 1 / s) and distance T (in light-seconds, 0..tmax):
// the field at that distance from an end that switches the mode on over RAMP, relative to its
// steady state. The mode crosses as a Klein-Gordon wave, kx(w) = sqrt(w^2 - kappa^2) (c = 1,
// kappa^2 = k^2 - kx^2: its wavenumber along y and z), so
//   E(T, t) = (1 / 2 pi) int R^(W) e^{i (kx(w + W) - kx(w)) T} e^{-i W t} dW,
// R^ the spectrum of the switch-on; evaluated by FFT (the switch-on made a long pulse, so it
// is periodic, and read before it ends). Nothing arrives before the light-speed front (t = T);
// the wave itself comes at the group velocity (t = T s), ringing, its precursors ahead of it.
// Tabulated by age from the group arrival, tau = t - T s, from the front at the farthest
// distance (-lead) until it has settled (span), when it is blended into 1.
let RAMP, FS, NODES, ND, NA, FREQ;   // set from each request
const pow2AtLeast = n => 1 << Math.ceil(Math.log2(Math.max(2, n)));
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
function computeResponse(tmax, bands) {
  const ramp = x => x <= 0 ? 0 : x >= RAMP ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * x / RAMP);
  const DT = 1 / FS;
  const { slow, lead, span } = bands;
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

let wasm = null;
const wasmLoad = (typeof WebAssembly === 'object' ? fetch('/waves.wasm') : Promise.reject())
  .then(r => r.ok ? r.arrayBuffer() : Promise.reject())
  .then(b => WebAssembly.instantiate(b))
  .then(m => { wasm = m.instance.exports; }, () => {});
function computeResponseWasm(tmax, c) {
  const n = NODES * ND * NA * 4, p = wasm.alloc(2 * n);
  try {
    wasm.compute(p, tmax, RAMP, FS, NODES, ND, NA, c.LS0, c.DLS, ...FREQ);
    const all = new Float32Array(wasm.memory.buffer, p, 2 * n);
    return [all.slice(0, n), all.slice(n)];
  } finally { wasm.release(p, 2 * n); }
}

self.onmessage = async ({ data: { id, tmax, bands, c } }) => {
  try {
    await wasmLoad;
    ({ RAMP, FS, NODES, ND, NA, FREQ } = c);
    const tables = wasm ? computeResponseWasm(tmax, c) : computeResponse(tmax, bands);
    self.postMessage({ id, tables }, tables.map(t => t.buffer));
  } catch (error) {
    self.postMessage({ id, error: String(error) });
  }
};
