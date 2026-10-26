# Atlas restoration staging verification — 20 October 2026

Source commit: `d5111e21`  
Proof: https://staging.universalmind.tv/informatics/

This correction supersedes the interface recorded in `atlas-areas-staging-verification-20261020.md` (source `3f8d9422`).

## Restored interface

- The plain Launch route opens the original homepage animation. Original homepage SVG, footer and `style.css` match `5e3e07d2` exactly.
- The original About dialog, including all research wording and resources, matches `5e3e07d2` verbatim.
- Research Atlas / Home occupies the original top-right position alongside the existing signature, music, fullscreen and return controls.
- About, Geometry, Genomics, Sound and Vision appear in a compact dropdown. Hover, touch and keyboard access are supported. The tagline appears beneath the brand on hover or keyboard focus.
- The permanent navigation row, permanent tagline, duplicated sequence labels and separate rewritten About page are removed.
- Sound constructions run automatically in 2.4–3.4-second drawing cycles with a short hold. Small pause/replay controls replace the full-width transport. Existing diagrams and explanatory workings remain available.
- Vision draws two equilateral triangles and reveals their shared central hexagon. Independent polygon clipping confirmed the intersection coordinates within 0.001 SVG units. The tracking-dot comparison and its entire control panel are removed. No new visible triangle or tetrahedron explanation was added.
- The supplied Vision Focus and Hypothesis and Sound research questions remain intact. The Vision construction is an SVG illustration; it does not introduce or define a clinical tracking protocol or a GIF stimulus library.

## Verification

- JavaScript syntax and Git whitespace checks passed.
- The complete candidate journey passed at 1470×956, 1024×768 and 390×844. A narrow-screen menu alignment correction then passed targeted touch checks at 320, 390 and 720 pixels.
- All existing staging preflights passed. The existing Panthur `informatics` deployment uploaded four changed assets and completed successfully.
- The complete journey passed again against served staging assets, without request interception, at 1470×956, 1024×768 and 390×844.
- Verified the original default launch, About content, hover/focus tagline, dropdown items and screen boundaries, Arrow/Home/End/Enter/Escape navigation and focus restoration.
- Served touch navigation passed at 320, 390 and 720 pixels, including opening the menu, selecting Vision, outside-tap dismissal and return to landing.
- Verified all five Sound views, automatic drawing progress, pause, replay, complete SVG downloads, scale/tuning changes, workings and exact research questions.
- Verified Vision triangle and hexagon construction, absence of tracking controls and exact hypothesis text. Reduced-motion mode presents complete static constructions.
- Verified existing Alpha, Beta, Gamma, Epsilon and Odom views, the 1001-record selector, retained record selection, Sound-to-M9/Beta navigation and Home return.
- Verified transcription, TFBS, publication register and First Principles dialogs, focus restoration and their placement inside the experience container.
- No JavaScript page exceptions or horizontal page overflow. Background film remained continuous through navigation and dialogs; Return to landing paused it.
- Inspected desktop, tablet and phone screenshots. Deployed CSS and JavaScript match committed bytes. HTML matches after removing the existing edge-injected Cloudflare analytics beacon.
- Every browser was a dedicated Chromium process with `--mute-audio` confirmed before navigation. Native fullscreen was mocked. Media was paused and all automated contexts closed after verification.

## Served asset proof

- `index.html` — HTTP 200; source SHA-256 `2d50958128436940ecee77ebcf666c9ed0f137609f0d26fe332669fc92bdd609`.
- `app.js` — HTTP 200; source SHA-256 `711e0d645cf764ae4fb1edd3535fe394861561143f8cd40f5c4874ae9f69aefc`.
- `atlas.css` — HTTP 200; source SHA-256 `0b7cd947dfc82cddc05d64a6b8b90fe53c40681a52fa6d516476654ad579f5ff`.
- `atlas-visuals.js` — HTTP 200; source SHA-256 `74c4e07b1cbc4f1be7b0f5d1432a167a13ee2a5778faa89e7034cf65a5de4c12`.

## Release boundary

Only the four Informatics page assets changed on staging. Existing Geometry records, genomic data, media behavior and production remain unchanged. No GPT or API changes were made.
