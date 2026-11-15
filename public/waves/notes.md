Claude

Yes, and the Fourier/DCT truncation version is easy to run in a browser. The wave-propagation version is also possible, but it’s better precomputed.

Truncated-mode style (the “sweet spot” look)

    Data size: At K×K = 12×12 coefficients per frame, 6,500 frames come to about 940k coefficients. Quantized to 8 bits and delta-coded against the previous frame, that’s roughly 0.5-1 MB. That’s smaller than the video.
    Render cost: Use a separable inverse DCT, which is two small matrix multiplies per frame. At 480×360 and K=12 that’s about 2M multiply-adds, which plain JS on a canvas handles at 30 fps. A WebGL fragment shader makes it effectively free.
    Crisp edges: Render at low resolution (say 160×120), upsample bilinearly, then threshold in a shader. You get smooth, sharp contours with the ringing preserved as ghost lines.
    Style controls: Cutoff K, hard vs. Lanczos taper, and threshold level can all be sliders. The threshold is a particularly fun one, since it moves the contour echoes around in real time.
    Sync: Drive the frame index from audio.currentTime so the audio and video don’t drift.
    Live mode: A forward DCT of a downsampled <video> frame to K×K coefficients is also cheap. A user could load a clip and see the effect live with no precomputation. The caveat is that you can’t ship the Bad Apple video itself, so users would need to supply their own file.

Boundary-to-slice wave version

    Realistic resolution: 128×128 to 256×256 with the angular spectrum method.
    Compute: Gerchberg-Saxton or gradient descent runs a few dozen FFT iterations per frame. WebGPU can do this in real time, but WebGL is borderline. Precomputing the phase maps offline and streaming them is more robust, though at 6,500 frames of 128×128 phases it’s much heavier on data (tens of MB).
    Warm starting: Starting each frame from the previous solution keeps flicker down. In the browser you’d need to run only a few iterations per frame.
    Visual payoff: The extras are speckle and diffraction fringes, plus you could show the boundary phase pattern next to the result, which makes a nice visualization of the idea.

My recommendation: Ship the DCT version first, with Chladni-style basis modes if you want a distinctive look, and add the wave-propagation view as a second mode if you want the physics story.

Want me to build a working demo? I can make a self-contained page that takes any video you load, applies mode truncation live with sliders for cutoff, taper, and threshold, and shows the coefficient grid next to the output.

what is minimum k for arbitrary approximation

There’s no finite minimum for arbitrary accuracy. It depends on the accuracy you demand, and exactness requires the full basis.

Exact reconstruction: For an N×M pixel image, k must reach the Nyquist limit, k_x = N/2 and k_y = M/2. That is every mode, so 480×360 needs all 172,800 coefficients. Truncating at any lower k can’t be exact for a generic image, because pixel-level detail lives in the highest modes.

Arbitrary (ε) approximation: For any ε you can find a k, but how fast it shrinks depends on the image’s smoothness.

    Smooth/band-limited images: The error falls off rapidly (exponentially for analytic ones), so a small k suffices.
    Images with sharp edges (Bad Apple): Convergence is slow. Mean-squared error scales roughly as (edge length)/k, so the L2 error goes as k^(-1/2). Halving the error needs about 4× the cutoff, which means about 16× the modes. This is the Gibbs penalty.
    After thresholding: What matters is edge position error, roughly L/(2k) as I estimated earlier. For under 1 pixel of error on a 480-px-wide frame, you need k ≈ 240, which is already Nyquist. So “visually pixel-perfect” is effectively the full basis, and the distinct-but-stylized regime (k ≈ 6-12) corresponds to edge errors of several pixels.

In the physical (boundary) setting: The ceiling comes from physics, not the image. Propagating waves carry spatial frequencies only up to 1/λ, so on a slice of width W the maximum is k_max ≈ W/λ cycles per frame (pixel pitch λ/2). Evanescent components decay exponentially with distance, so you can’t reach beyond this without near-field access. An arbitrary pattern is therefore achievable only if the pattern’s own k is below W/λ. For anything sharper, you get its low-passed version, which is just the truncated-mode look from before.

Practical rule: Pick the error tolerance first (say IoU > 0.95), find the k where the thresholded truncation hits it by sweeping k on real frames, and check it against your hardware ceiling W/λ. For Bad Apple that sweep will land well above 12, probably in the 30-60 range, for IoU around 0.95. I’d verify this on the actual frames rather than trust my estimate.

no, k as in pattern in n-k dimensions within n dimensional space

Ah, k as codimension. Then the minimum is k = 1 for static or fixed-frequency fields, and k = 0 is possible only with time dynamics.

Static or single-frequency fields (Laplace, Helmholtz):

    k = 0 (pattern fills the whole n-dimensional volume): Impossible. The boundary is (n−1)-dimensional, so it carries one dimension less data than the volume you want to fill. The field is also constrained by the PDE, so the interior is fully determined by the boundary.
    k = 1 (a hypersurface slice): This is the minimum. The counting works out exactly: n−1 boundary dimensions map to n−1 target dimensions. By unique continuation and Runge-type density, boundary data can approximate any reasonable pattern on the slice to arbitrary accuracy. It is only approximate, never exact, because the boundary-to-slice map is compact. The cost is ill-posedness: fine detail needs boundary amplitudes that blow up as the error shrinks.

I overstated the maximum-principle objection in my first answer. It forbids interior bumps in the full volume (k = 0), but a harmonic function restricted to a slice can have extrema, so k = 1 is fine.

k ≥ 2: Easier, with no new obstruction. The (n−1)-dimensional boundary has k−1 surplus dimensions of freedom relative to the target, so many boundary configurations produce the same pattern. You can pick the minimum-norm or lowest-amplitude one, which tames the instability somewhat. The smoothing and evanescent limits still apply.

With time dependence: The boundary control now spans (n−1)+1 = n dimensions, which equals the volume’s dimension, so k = 0 becomes possible.

    Waves: Exact controllability (HUM) lets you steer to an arbitrary finite-energy state throughout the volume at time T, given enough time and boundary coverage.
    Heat: You get approximate controllability only. Any target can be approached, but not hit exactly, and the amplitudes are extreme.

For Bad Apple: A 2D frame in 3D space is k = 1, so a plane in front of a 2D boundary surface works, as discussed. A 2D frame filling a 2D domain would be k = 0, which needs a time-driven wave system, for example a Chladni plate or water surface driven from its edge, and the frame would exist only at one instant per control sequence.

E&M, not simple waves (need rgb)

Switching to E&M with RGB doesn’t change the minimum, k = 1, but it changes how you get there.

Why Maxwell reduces to the earlier picture

In a linear, source-free medium, each monochromatic component obeys the vector Helmholtz equation, and each Cartesian field component obeys the scalar one. So:

    Each color is an independent k = 1 problem. Red, green, and blue are three separate frequencies with no coupling in linear media. A boundary field at each wavelength can approximate an arbitrary intensity pattern on a slice, with the same ill-posed amplitude blow-up.
    k = 0 is out for optics. Filling a 3D volume with an arbitrary RGB pattern isn’t possible. A monochromatic field is determined throughout the volume by its values on a surface, so you only ever get about one plane’s worth of freedom. The time-dependent route that gave k = 0 for waves needs boundary control at about 10¹⁵ Hz, which isn’t available. At microwave or RF frequencies it is, but then there’s no RGB.
    Polarization adds freedom, not dimensions. Each spatial frequency carries two transverse polarizations. For intensity targets this gives extra slack, but the vector coupling only matters at features near λ. Scalar propagation per color is accurate for display-scale patterns.

What RGB actually changes

    The resolution ceiling differs per color. k_max ≈ W/λ, so blue (~450 nm) resolves about 40% finer than red (~630 nm). The sharpest common pattern is set by red.
    Propagation scales with λ. The angular-spectrum phase is e^{i·kz·z}, with kz = √((2π/λ)² − kx² − ky²). One boundary phase map therefore produces three different patterns at the same plane, which is chromatic dispersion.
    The joint-design problem depends on the hardware:
        Three independent boundary channels: Three SLMs, or a color-sequential setup. Solve each color separately with the same GS or gradient method as before. This is the easy path.
        One fixed passive element (metasurface or hologram plate): The phase it imprints scales as 1/λ, so you must optimize one structure against three targets at once. Phase-only elements can’t match three arbitrary patterns exactly, so expect crosstalk. Dispersion-engineered metasurfaces or interleaved subpixels (each color gets about a third of the aperture, lowering its k_max) are the usual workarounds.
    Speckle is worse. Three coherent lasers each produce their own speckle. LEDs or short time-averaging reduce it at some cost to sharpness.

Practical recommendation

Use field-sequential color. One boundary device runs three exposures per frame, each at its own wavelength, with its own GS or gradient solution, warm-started from the previous frame. It needs 3× the frame rate but avoids the joint-optimization problem entirely. For a simulation, it’s just three independent angular-spectrum solves per frame.

Near-field exception

Sub-wavelength detail is possible only if the slice is within a fraction of λ of the boundary, where evanescent components are still alive (near-field plates, plasmonic superlenses). The cost is exponential loss, and the pattern is gone a few hundred nanometers away.

Is this a simulation, or are you thinking of real hardware (SLM, metasurface, LED array)? That decides whether the three colors are independent or coupled.

With R, G, B as E_x, E_y, E_z, the three channels are no longer independent. I’m assuming the image plane is spanned by the field’s x and y axes, so the slice is a 2D plane at fixed (z, w, ...) in an n-dimensional simulation. Components beyond x, y, z (E_w and any magnetic components) are discarded at render time.

The constraint that couples the channels

In a source-free region, every Fourier mode has to be transverse: k·E = 0. A single mode can therefore only supply the part of an RGB triple that is perpendicular to its wavevector. That makes the style a direct consequence of the geometry.
Setup	Independent channels at the slice	Result
n = 3, boundary on one side only	2 of 3	E_z is slaved to (E_x, E_y)
n = 3, boundary enclosing the slice	3 of 4 (one spare)	RGB fully independent, k = 1 still works
n = 3, static (E = −∇φ)	about 2	R and G must form a curl-free gradient field
n ≥ 4, boundary surrounding the slice in the extra dimensions	3	Fully independent, with lots of redundancy

Avoid the static case. It forces (E_x, E_y) to be a gradient field, and extra dimensions don’t relax that.

n = 3, one-sided (the most stylized case)

For a forward-going field, E_z = −(kx·E_x + ky·E_y)/kz at each spatial frequency.

    In smooth regions (k∥ ≪ ω), the wave travels almost along z, so E_z is almost zero. With a signed-to-color mapping of 0.5 + 0.5·E/E_max, blue sits at mid-level everywhere.
    Blue appears only where the in-plane field has divergence, which means at edges. As kz → 0 near the band edge, E_z grows relative to E_x and E_y, giving strong blue fringes on sharp contours.
    A grayscale target like Bad Apple is the worst case, because it asks for E ∥ (1,1,1) at every pixel. The best achievable field is the projection of the target onto the plane perpendicular to k̂. Big white areas come out yellow (1, 1, 0.5), black areas come out (0, 0, 0.5), and edges show blue/cyan/orange fringes. The part of the target the physics can’t make is exactly what produces the color.

n ≥ 4: discarded components as slack

The divergence constraint in n dimensions is the sum over all n components, kx·E_x + ky·E_y + kz·E_z + kw·E_w = 0. If you throw out E_w, it absorbs any mismatch, so (E_x, E_y, E_z) can be arbitrary. A gray target then renders gray, and the stylization has to come from other knobs. Some things to know:

    Boundary geometry matters. The boundary needs to surround the slice in the (z, w) directions so that wave directions span a full circle. A one-sided boundary reintroduces the n = 3 behavior.
    Band-edge attenuation. The extra directions collapse as |k∥| → ω, so near the cutoff the part of (R, G) along k∥ is hard to supply, and the cost diverges. Fine detail loses its gradient-aligned component, which gives a subtle chroma shift on thin features.
    Redundancy. The boundary has far more freedom than the target, so you choose the minimum-norm solution. A penalty weight on the discarded components (E_w and B) is a style knob. Weak penalties let energy hide in the discarded components, and the render dims.
    Resolution is unchanged. The slice still only carries |k∥| ≤ ω, so the earlier cutoff dial applies.

Sign and DC handling

Fields are signed and colors aren’t. Use the offset mapping (0.5 + 0.5·E/E_max), or render |E| or E² for a rectified look. A uniform E_z can’t exist in 3D, so black-to-white targets should be coded as −A to +A rather than 0 to A.

Cheap way to implement it

For a periodic box, the whole “physics” reduces to a per-frequency matrix:

    FFT the three channels of the target frame.
    At each (kx, ky), multiply the 3-vector by a 3×3 matrix M(k∥): the projection I − k̂k̂ᵀ for n = 3 one-sided, near-identity with band-edge attenuation for n ≥ 4, and zero beyond |k∥| = ω.
    Inverse FFT.

Minimum k stays 1 for the slice to carry the pattern at all, with n = 3 giving you the strongly constrained look and n = 4 the unconstrained one.

what is the first number of dimensions with arbitrary approximation

n = 3, provided the boundary surrounds the slice, so waves can arrive from both sides of the plane. A one-sided boundary pushes the first working dimension up to n = 4.

Why n = 3 works with a two-sided boundary

Take a 2D slice at z = 0 and fix an in-plane wavevector k∥ with |k∥| < ω. Two plane waves share that k∥: one with +kz and one with −kz. Each carries a polarization in the 2D plane perpendicular to its wavevector. Those two planes are different whenever kz ≠ 0, so together they span all of ℝ³. That means any 3-vector (E_x, E_y, E_z) can be produced at that k∥. The RGB channels are then independent, and you get arbitrary approximation up to the |k∥| ≤ ω cutoff, with the usual ill-posed amplitude blow-up for fine detail. This is also the k = 1 case, which is the minimum codimension.

Why nothing smaller works

    n = 2: There is no E_z component, so there are only two color channels. A 2D slice of a 2D space is also the whole domain (k = 0), which was ruled out for static or single-frequency fields.
    n = 3, one-sided: There is only the +kz branch, so the field at each k∥ is confined to a single transverse plane. E_z is slaved to (E_x, E_y), which gives the stylized blue-fringe look from before.

Why n = 4 works one-sided

With one extra dimension, (kz, kw) lies on a circle of radius √(ω² − |k∥|²). Even restricting to kz > 0, k_w takes both signs, giving a range of different transverse subspaces. Projecting onto (x, y, z) after discarding E_w, these span all three components, so it works without enclosing the slice.

Caveat for both: the degeneracy at the band edge (kz → 0 in n = 3, or the circle radius → 0 in n = 4) makes the required amplitude diverge. The independence is exact at every k∥ below cutoff, but the cost of the highest in-plane frequencies is what limits sharpness in practice.


Claude is AI and can make mistakes. Please double-check responses.
