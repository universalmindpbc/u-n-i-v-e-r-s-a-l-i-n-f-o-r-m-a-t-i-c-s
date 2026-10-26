# Quantum Atlas result access and timing — 25 October 2026

Deployed website source: `aba82988`.

Proof: https://staging.universalmind.tv/informatics/?entry=research&area=quantum

## Changes

- Recorded Study 01 results are available on first load, without a question or saved session. A prominent results panel directly after the technology diagram opens CUDA-Q, Aquila, or the combined interpretation.
- Result links switch to the relevant panel and focus the selected report. Shared study results remain identified as Study 01; unrelated questions, different study versions, and pending questions never inherit them as their own answers.
- Nature reference and geometric hypothesis moved after Biological challenges and research foundations, immediately before Science as a Public Service. Button text is exactly `Read the Nature Study`.
- Completed result tabs display availability immediately. Future quantum turnaround varies with device availability and queue, with recorded examples in the delivery section.
- The earlier two-atom hardware record now displays completed status, 100 returned shots and its archived measurements. No new quantum task was submitted.

## Timing evidence

Recorded Amazon Braket metadata is published in `Informatics/quantum/data/execution-timing.json` with account identifiers and storage locations omitted.

- Sixteen-run OXTR batch: first submission `2026-10-25T01:05:00.993Z`; final completion `2026-10-25T01:21:52.327Z`; elapsed **1,011.334 seconds** (16 minutes 51 seconds). First result after 196.056 seconds.
- Earlier two-atom test: submission `2026-10-24T04:07:26.498Z`; completion `2026-10-24T22:59:41.681Z`; elapsed **67,935.183 seconds** (18 hours 52 minutes).
- These durations include queue and execution. They are observed turnaround times, not QPU execution durations or guaranteed delivery times.

## Verification

- Local report-contract and routing checks passed, including rejection of mismatched study identity, missing receipts, unsupported genes, and pending-question fallback.
- Existing Panthur staging preflights and deployment completed; ten changed website assets uploaded. Ten HTTPS asset hashes match the committed source.
- Fresh staging session: all three results entry buttons opened the correct report, focused it and generated zero question requests. Both DNA windows and eight representation rows remained available.
- A real OpenAI question returned HTTP 200 and a provider response identifier; the on-page receipt and linked results displayed. Saved-study reopen generated no additional question request.
- Nature dialog opens and closes with Escape, returning focus. Earlier connection view confirms completed status and 100 returned shots.
- Desktop, 390-pixel and 320-pixel views checked, including expanded atom detail; no horizontal overflow or page errors.
- Separate headless Chromium process launched with `--mute-audio` confirmed in process arguments. Media paused and browser context/process closed after verification.
- The first test attempt reached result navigation, then checked chart rows before their data finished loading. The test now waits for the recorded rows; the complete rerun passed. Website source required no repair.

Production was not deployed or modified. Quantum simulation and hardware receipts are unchanged; the biological occupancy-prediction benchmark remains unexecuted.
