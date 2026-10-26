# Quantum Atlas staging verification — 25 October 2026

## Deployed source

- Main implementation: `cb84b510`.
- Final source: `7cfd44b0` (embedded research media-control correction).
- Previous staging: `eed05594e824cf600698d26a98992b7403b278fb`.
- Staging frontend: existing Panthur `deploy-panthur-staging.sh informatics` workflow, 21 initial changed assets and three follow-up uploads.
- Staging API: existing `informatics-universal-one-staging` Worker, version `96152c07-ba00-43ea-8586-180b104fda3b`.
- Production was not deployed. Its remote ref before the staging audit push was `6d58d23e36fa2942e260a4eacd808c7eb68c3623`.

## Verified behavior

- All 21 changed frontend assets returned HTTP 200 and matched the final committed source byte for byte.
- Eleven primary destinations retain the same frame at widths 1470, 390 and 320. Global navigation, music, fullscreen control and the accessible Universal One orb remain present. Quantum follows Molecular. No JavaScript errors were recorded.
- Six direct routes resolve into the shared frame, including preserved research record and library collection parameters.
- Transcription section dropdown, music toggle and music-state persistence across navigation passed at 1024 × 768. Native fullscreen was not invoked.
- Embedded library and connection views hide their duplicate global and playback controls across video, black and white reading modes. Transcription background video decoded and advanced in the muted browser.
- Four saved, successful Quantum Atlas OpenAI responses cover binding evidence, a contextual follow-up, DNA reconstruction and a fresh-session repeat. Responses reached the server and rendered in the interface. The repeated result matches exactly; only its conversation run identifier differs.
- Two earlier successful AI calls preceded correction of a test expectation: recommending a classical baseline is a valid answer. Six total QA-pool questions were consumed; the four retained responses form the complete recorded sequence and repeat evidence.
- Downloaded model JSON exactly matches the visible DNA calculation. The repeat-question button restores the prompt. Optional historical verification opens and closes correctly.
- A simulated HTTP 503 retains the input without adding a fabricated answer. Invalid input returns 400; an unapproved origin returns 403.
- Existing Universal One completed a real OpenAI request from Transcription Atlas and rendered its source-backed M9 answer.
- Independent model checks cover all 512 three-fragment bit assignments, QUBO/Ising energy equivalence, input rejection, ambiguity and gaps. The demonstration reconstructs `ATGCCTAGGATTC` with total overlap 8.
- Shared question-budget tests cover concurrent admission, daily/weekly/monthly limits and period rollover. Staging limits are 30/120/300 AI questions, separate from the bounded QA pool.
- The existing two-atom solver passed its 25-case SciPy comparison, with maximum absolute difference `2.588711467410576e-9`.
- All automated browser work used a separate headless Chromium process with `--mute-audio` positively verified before navigation. Media was paused and browser sessions closed afterward.

## Interpretation and remaining integration

The live interface provides biological research guidance and a small exact classical reconstruction. It does not dispatch new GPU or QPU jobs. Wolfram Language and QUBO/Ising exports are model artifacts, not evidence of execution in Wolfram or on quantum hardware. Public GPU/QPU funding and dispatch remain inactive.

The shared question allowance is an initial staging safeguard, not a monetary hardware budget. The intended service uses public funding with daily, weekly and monthly shared capacity rather than individual researcher approval.

The five cited study summaries are curated evidence. This version does not retrieve new ENCODE, JASPAR or literature records live. DNA binding prediction and the D-Wave benchmark remain proposed integrations. Applying a trained model to OXTR requires appropriate measured data and independent validation.

## Proof

- https://staging.universalmind.tv/informatics/?entry=research&area=quantum
- https://staging.universalmind.tv/informatics/?entry=research&area=transcription

Machine-readable checks, deployment output, screenshots and the scoped secret scan accompany this record. The following audit commit records this evidence and the deployment manifest on `origin/staging`.
