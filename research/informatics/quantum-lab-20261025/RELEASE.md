# Quantum Lab staging release — 25 October 2026

Quantum Lab follows Transcription Atlas in the shared navigation. It provides a bounded two-neutral-atom experiment, an independently checked browser solver, a simulated-time visualisation, the recorded GPU comparison, a dated Aquila API status, a teachable workflow, biological-method guidance and five primary research cards.

Scientists can create simulations, rerun with new IDs, retain local history, import parameters, export full run records and figures, and prepare a local hardware review file. The cloud access register distinguishes verified research submissions from activated website connections.

## Access decision

Universal Informatics funds approved researchers within individual budgets. See GATEWAY.md for the activation contract. Hardware submission, researcher sign-in, budget administration and shared server storage are pending. No live GPU or QPU job is submitted by this website release.

## Validation

- Independent SciPy DOP853 comparison over 25 baseline and boundary cases: maximum probability difference below 2.6e-9.
- Browser checks at 1470×956, 1024×768, 390×844 and 320×844: parameter changes, distinct reruns, deterministic reproduced probabilities, import/export, SVG figure download, persistent local history, history removal, rejected invalid model import and local-only hardware request.
- Embedded navigation order, area switching, focus, keyboard Escape and no horizontal overflow checked.
- Separate Chromium process with --mute-audio; no native fullscreen; automated contexts closed after testing.
- API refresh of the existing Aquila task: QUEUED, position 456 at 2026-10-24 18:18:38 UTC. No new hardware submission.
- No account identifiers, credentials or private provider task ARNs are included in public experiment data.

Production remains unchanged. Staging deployment and live evidence are recorded alongside this file after publication.

## Staging publication

- Source commit: ede9babf.
- Existing Panthur staging workflow uploaded 17 changed assets. The shared preflight required its documented external Mugler fixture; its SHA-256 matched the required value. No preflight was bypassed and no unrelated website asset was deployed.
- All 17 deployed website files matched their committed bytes over HTTPS, including HTML, module JavaScript, data and download files.
- Live four-viewport checks passed. The site's existing /cdn-cgi/rum analytics beacon is excluded from the no-experiment-submission assertion; no job API was called.
- Additional embedded checks at 390 and 320 pixels passed navigation from Quantum Lab into Transcription and back, with no overflow or native fullscreen.
- Proof: https://staging.universalmind.tv/informatics/?entry=research&area=quantum
- Production source baseline: 6d58d23e36fa2942e260a4eacd808c7eb68c3623; no production deployment invoked.
