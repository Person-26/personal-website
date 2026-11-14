# Personal website

Static site (plain HTML/CSS/JS, no build step). Site files live in `public/`.
Deployed to Cloudflare Workers via `npx wrangler deploy` on every push to `main`.

Pages are drawn by `waves.js` as simulated light: a scalar wave field of three visible frequencies in the middle of a deep 3D waveguide,
driven only by a boundary condition at its left and right ends and solved mode by mode with FFTs
(WebGL2), with a
button to switch to the plain page. Any page that includes `waves.css` and
`waves.js` and puts its content in `<main id="content">` gets the same mode.
Without WebGL2 the plain HTML is shown.
