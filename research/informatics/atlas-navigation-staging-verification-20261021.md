# Shared Atlas navigation — staging verification, 21 October 2026

Source commit: `a327a5ca`  
Proof: https://staging.universalmind.tv/informatics/

## Scope

The header now contains a stable Research Atlas menu (About, Sound, Vision) and one shared M/P selector. M9 / P24 has two explicit destinations: Animation and Research view. All other entries open existing detailed records. The selector includes modulus/period search, an M9-family filter, the existing family comparison and an active-entry marker. The embedded Research Record selector is hidden; its existing data and control logic remain available to the shared navigation.

The original animation, About content, Sound and Vision content and all existing data records remain unchanged. No animations were generated for other moduli. The shorter brand tagline, “bioinformatics platform for mental health”, fades in over 700 milliseconds after a 100-millisecond entrance delay, and fades out on departure. Reduced-motion preferences retain the existing transition suppression.

## Verification

- JavaScript syntax checks and Git whitespace checks passed.
- Compared the original homepage main/footer, About dialog, Sound/Vision sections, base stylesheet, Sound animation source and record index against `8bfb0ffc`; all remain byte-identical.
- Candidate and served-staging journeys passed at 1470×956, 1024×768, 390×844 and 320×844. Phone contexts used touch input.
- Confirmed the plain launch opens the original M9 animation. M9 Research view opens the existing detailed page, and the selector returns to the animation.
- Verified M181/P90, M186/P120, M60/P120, M1001/P560 and M1/P1; the header and detailed view match the selection. The list contains all 1000 non-M9 records plus the pinned M9 research entry.
- Verified the M9-family filter's 110 non-reference members, existing family comparison and selection of M18 from that comparison.
- Verified retained record and tab selection when returning from Sound, Vision navigation, Sound-to-M9/Beta links and direct entry to M186/P120 Beta.
- Verified hover, touch, Arrow keys, Home/End, Escape, outside dismissal, focus restoration and dropdown bounds. Header controls stay on one row even at 320 pixels.
- Measured intermediate opacity during both tagline entrance and exit; confirmed the exact shorter wording.
- M181 CSV downloads matched local source bytes at every tested viewport.
- No JavaScript page exceptions or horizontal page overflow. Background film remained continuous through navigation. Every return to landing paused it.
- All browser QA used a dedicated Chromium process with `--mute-audio` confirmed before navigation. Native fullscreen was mocked. Media and browser contexts were closed after testing.
- Reviewed desktop and phone screenshots of the selector, animation, detailed records and tagline.

## Deployment

The first deployment preflight found the existing external shopfront video fixture missing from `/private/tmp`. Restored that exact fixture from the existing staging asset and verified the check's required SHA-256 (`6c444c7c66f806aff944d5eeba7d545d057e1edac3ee1a3c201ce3814678fb91`). No deployment scripts or preflight checks were changed.

The existing `./deploy-panthur-staging.sh informatics` then passed all preflights, uploaded seven changed assets and completed successfully. Live verification used served assets without interception.

## Served asset proof

- `index.html` — HTTP 200, byte-identical, SHA-256 `6746456343fa005c4b2b786b34fcdc45ed733dccd755acf8be1c2d43e68f3b9c`.
- `app.js` — HTTP 200, byte-identical, SHA-256 `59132bab3b50efb45af8ea53697411f13004f66464ed7de68e6d18b86e61b4f4`.
- `atlas.css` — HTTP 200, byte-identical, SHA-256 `d64d36961331ee4075a73027dd77a3ee5eb6d2be8c7c0b5d70296a50b90510d4`.
- `record-navigation.js` — HTTP 200, byte-identical, SHA-256 `07058c2083567bd181e1adafcd6ead3d3782f519825b783ff634e268e51bd856`.
- `research.html` — HTTP 200, byte-identical, SHA-256 `d9b711f674fa1f2b238da62613619ac7045e6f70f756262db56ec36388b6d99d`.
- `research.css` — HTTP 200, byte-identical, SHA-256 `62d34fbc42967c29ae4267de8a57b4e627f72e044302c18325c2fb664decc8bd`.
- `research.js` — HTTP 200, byte-identical, SHA-256 `0a3a15fe61cf34e357541b677ec1d79585ecb3a2539ab65c34e0ef4a4319f97e`.

## Boundaries

Production was not modified. No GPT/API, scientific wording, genomic data, sequence calculations or media engine changes were made.
