# Personal website

Static site (plain HTML/CSS/JS, no build step on deploy). Site files live in `public/`.

`waves.js` gets its response tables from a Web Worker (`public/waves-response.js`), off the page's
thread. The worker computes them with `public/waves.wasm`, built from the Rust in `wasm/` and committed
(about twice as fast as the worker's JS fallback). After editing `wasm/src/lib.rs`, rebuild it with
`wasm/build.sh` (needs the `wasm32-unknown-unknown` target: `pacman -S rust-wasm`).
Deployed to Cloudflare Workers via `npx wrangler deploy` on every push to `main`.

Pages are drawn by `waves.js` as simulated light: a scalar wave field of three visible frequencies on a plane in open space,
driven only by open (non-reflecting) boundaries at its left and right and solved mode by mode with FFTs
(WebGL2), with a
button to switch to the plain page. Any page that includes `waves.css` and
`waves.js` and puts its content in `<main id="content">` gets the same mode.
Without WebGL2 the plain HTML is shown.
The wavelength is set in the screen's own pixels (so it is finer under display scaling), within a GPU
memory budget, and the grid is made coarser if the first frames are slow (remembered in
`localStorage` as `waves-quality`).

`cv.pdf` is the CV page printed (its print styles are at the end of `waves.css`). After editing
`cv.html`, regenerate it with any Chromium, with the site served locally:
`chromium --headless --no-pdf-header-footer --print-to-pdf=public/cv.pdf http://127.0.0.1:8765/cv`.
