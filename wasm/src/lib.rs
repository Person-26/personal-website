// The response tables of waves.js (computeResponse in waves-response.js is the reference and the fallback;
// see its comments for the physics). For each band of k / |kx| and each frequency: the wave
// arriving at distance T from an end that was switched on (over RAMP) at time 0, as a function of
// its age from the group arrival, found through its spectrum by FFT and blended into 1 once settled.
//
// The same as the JS but for one thing: e^{i (kx(w + W) - kx(w)) T} is stepped from one distance to
// the next by a complex multiply instead of an exp, cos and sin per spectral point per distance.
//
// Interface (no bindings): alloc(n) gives room for n f32s; compute(out, ...) fills 2 tables of
// nodes * nd * na * 4 f32s each, laid out as in waves.js; release(out, n) frees it.

use std::f64::consts::PI;

#[unsafe(no_mangle)]
pub extern "C" fn alloc(n: usize) -> *mut f32 {
    let mut v = vec![0f32; n];
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn release(p: *mut f32, n: usize) {
    drop(unsafe { Vec::from_raw_parts(p, n, n) });
}

fn fft(re: &mut [f64], im: &mut [f64], cs: &[f64], sn: &[f64], sign: f64) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let step = n / len;
        let h = len >> 1;
        for i in (0..n).step_by(len) {
            for k in 0..h {
                let (cr, ci) = (cs[k * step], sign * sn[k * step]);
                let (p, q) = (i + k, i + k + h);
                let xr = re[q] * cr - im[q] * ci;
                let xi = re[q] * ci + im[q] * cr;
                re[q] = re[p] - xr;
                im[q] = im[p] - xi;
                re[p] += xr;
                im[p] += xi;
            }
        }
        len <<= 1;
    }
}

fn pow2_at_least(n: f64) -> usize {
    1usize << (n.max(2.0).log2().ceil() as u32)
}

/// # Safety
/// `out` must hold 2 * nodes * nd * na * 4 f32s (from alloc).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn compute(
    out: *mut f32,
    tmax: f64,
    ramp_s: f64,
    fs: f64,
    nodes: usize,
    nd: usize,
    na: usize,
    ls0: f64,
    dls: f64,
    f0: f64,
    f1: f64,
    f2: f64,
) {
    let size = nodes * nd * na * 4;
    let tables = unsafe { std::slice::from_raw_parts_mut(out, 2 * size) };
    tables.fill(0.0);
    let freq = [f0, f1, f2];
    let ramp = |x: f64| {
        if x <= 0.0 {
            0.0
        } else if x >= ramp_s {
            1.0
        } else {
            0.5 - 0.5 * (PI * x / ramp_s).cos()
        }
    };
    let dt = 1.0 / fs;
    for b in 0..nodes {
        let slow = (ls0 + b as f64 * dls).exp();
        let lead = tmax * (slow - 1.0) + 0.05;
        let span = 1.5 + 0.5 * tmax * slow;
        let last = tmax * slow + span;
        let pulse = last + 1.0;
        let n = pow2_at_least(((2.0 * last + 4.0) * fs).ceil());
        let cs: Vec<f64> = (0..n / 2).map(|i| (2.0 * PI * i as f64 / n as f64).cos()).collect();
        let sn: Vec<f64> = (0..n / 2).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();
        let mut rr = vec![0f64; n];
        let mut ri = vec![0f64; n];
        for (k, r) in rr.iter_mut().enumerate() {
            let x = k as f64 * dt;
            *r = (ramp(x) - ramp(x - pulse)) * dt;
        }
        fft(&mut rr, &mut ri, &cs, &sn, 1.0);
        let (mut er, mut ei) = (vec![0f64; n], vec![0f64; n]);
        // e^{i (kx(w + W) - kx(w)) T} at the current distance, and its step per distance
        let (mut zr, mut zi) = (vec![0f64; n], vec![0f64; n]);
        let (mut sr, mut si) = (vec![0f64; n], vec![0f64; n]);
        let dtt = tmax / (nd - 1) as f64;
        for (f, &fr) in freq.iter().enumerate() {
            let w = 2.0 * PI * fr;
            let base = (f >> 1) * size;
            let ch = (f & 1) * 2;
            let kx0 = w / slow;
            let kap2 = w * w - kx0 * kx0;
            for m in 0..n {
                let mm = if m < n / 2 { m as f64 } else { m as f64 - n as f64 };
                let om = w + 2.0 * PI * mm / (n as f64 * dt);
                let q = om * om - kap2;
                let (dkr, dki) = if q >= 0.0 { (om.signum() * q.sqrt() - kx0, 0.0) } else { (-kx0, (-q).sqrt()) };
                let mag = (-dki * dtt).exp();
                sr[m] = mag * (dkr * dtt).cos();
                si[m] = mag * (dkr * dtt).sin();
                zr[m] = 1.0;
                zi[m] = 0.0;
            }
            for it in 0..nd {
                let t = tmax * it as f64 / (nd - 1) as f64;
                if it > 0 {
                    for m in 0..n {
                        let (a, c) = (zr[m], zi[m]);
                        zr[m] = a * sr[m] - c * si[m];
                        zi[m] = a * si[m] + c * sr[m];
                    }
                }
                for m in 0..n {
                    er[m] = rr[m] * zr[m] - ri[m] * zi[m];
                    ei[m] = rr[m] * zi[m] + ri[m] * zr[m];
                }
                fft(&mut er, &mut ei, &cs, &sn, -1.0);
                let norm = 1.0 / (n as f64 * dt);
                for ia in 0..na {
                    let tau = -lead + (lead + span) * ia as f64 / (na - 1) as f64;
                    let a = t * slow + tau;
                    let (mut vr, mut vi) = (0.0, 0.0);
                    if a > 0.0 {
                        let x = a / dt;
                        let n0 = ((n - 2) as f64).min(x.floor()) as usize;
                        let fx = x - n0 as f64;
                        vr = (er[n0] * (1.0 - fx) + er[n0 + 1] * fx) * norm;
                        vi = (ei[n0] * (1.0 - fx) + ei[n0 + 1] * fx) * norm;
                    }
                    let fade = ((tau - 0.6 * span) / (0.4 * span)).clamp(0.0, 1.0);
                    let sm = fade * fade * (3.0 - 2.0 * fade);
                    vr += (1.0 - vr) * sm;
                    vi -= vi * sm;
                    let o = base + ((ia * nodes + b) * nd + it) * 4 + ch;
                    tables[o] = vr as f32;
                    tables[o + 1] = vi as f32;
                }
            }
        }
    }
}
