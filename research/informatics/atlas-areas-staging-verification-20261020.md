# Atlas areas staging verification — 20 October 2026

Source commit: `3f8d9422`  
Proof: https://staging.universalmind.tv/informatics/?area=sound

## Scope

Persistent About, Geometry, Genomics, Sound and Vision navigation, with a native area selector on narrow screens. Five Sound teaching views: sequence structure, melodic structure, circle of fifths, six around one, and tuning/frequency. Vision includes the supplied Focus and Hypothesis verbatim, plus stepped targets and continuous tracing for hexagonal/horizontal comparison. The bicyclic animation remains outside this release.

The original M1–M1001 records and genomic data are unchanged. Existing research dialogs remain within the experience. New diagrams use the existing Atlas colours, thin paths and progressive drawing. Playback, replay, scrubbing and SVG export are available for each construction.

## Verification

- JavaScript syntax and Git whitespace checks passed.
- Candidate browser checks passed at 1440×1000, 1024×768 and 390×844.
- All existing staging deployment preflights passed. The existing Panthur `informatics` workflow uploaded four changed assets and exited successfully.
- The same complete browser journey passed against served staging assets at all three sizes, without request interception.
- Verified exact Sound research questions, footnote and Vision hypothesis; arithmetic for phrase residues, octave ratios, tuning-reference scaling and G-major notes.
- Verified every Sound tab, keyboard tab navigation, playback/pause, replay, scrubbing, scale and tuning controls, expanded workings and SVG downloads.
- Verified Vision path selection, stepped/continuous target motion, duration, scrubbing and SVG download.
- Verified transcription, TFBS, publication register and First Principles dialogs; fullscreen-container placement; Gamma/Alpha/Beta/Epsilon and Odom views; 1001-record selector; preserved selected record on area return; Sound-to-M9/Beta navigation; animated overview and return to landing.
- No JavaScript page exceptions or horizontal page overflow. Background media continued through navigation and dialogs. Return to landing paused playback.
- Browser QA ran in dedicated headless Chromium with `--mute-audio` confirmed before navigation. Native fullscreen was mocked; no operating-system fullscreen was invoked. Test contexts and media sessions were closed.
- Visually inspected desktop, tablet and phone screenshots. The changed CSS and JavaScript responses match the committed bytes. HTML matches after removing the existing edge-injected Cloudflare analytics beacon.

## Served asset proof

- `index.html` — HTTP 200, source SHA-256 `f0a22ec0a980a176efe941d7a2c35b8e27355fa984134295b7ec9a3f00c25ba9`.
- `atlas.css` — HTTP 200, source SHA-256 `a86c91ace456b4348c7a0a9ad7ed7b41251ef0c65e17493b3ca118aecd2187f4`.
- `atlas-visuals.js` — HTTP 200, source SHA-256 `98df38e438bd556491e74e56e70cface98275d61804b357a3387b8c61fc8437f`.
- `app.js` — HTTP 200, source SHA-256 `cbb968593b98453d8b3815eb4c76be0d94d66afbd58379eceea4fc2c9bc14c52`.

## Boundaries

The animations illustrate the proposed relationships and tracking patterns. No eye-movement recording, biological assay, therapeutic outcome or new audio engine was introduced. No GPT/API behavior changed. Production was not modified.
