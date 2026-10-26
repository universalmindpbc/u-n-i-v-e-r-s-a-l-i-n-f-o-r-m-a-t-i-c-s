# Quantum Atlas: goal, verdict and next action

Staging source: `03469bb3`; principal implementation: `05de1e8e` and `5758cf9e`.
Base: `8558eb8402ad2351c1888a19cdb879a007708f1a`, including both completed Vision updates.
Proof: https://staging.universalmind.tv/informatics/?entry=research&area=quantum

## Product change

The yellow summary now starts with the biological question, the limited purpose of the first run, its recorded result, the signal sought in the next test, and one next-action button. It explicitly states that binding-site accuracy was not measured in the first 16 runs.

CUDA-Q and Aquila readers explain their goal and result before the charts. The term ideal was replaced in the visible readings with a defined calculated prediction. Lower/higher atom excitation and agreement with a calculation are explained without implying that either direction is a biological success.

A working Next test reader presents the public-data comparison, positive/inconclusive/discouraging outcomes, a separate quantum-versus-classical comparison, and the situations that call for later wet-lab work. It distinguishes proposed future decision rules from the numerical checks set for the completed run. Final sample size, uncertainty level and minimum useful gain are explicitly still to be fixed before prediction evaluation.

The reader opens the five saved CTCF-binding intervals and both original setup samples. This is a new coverage audit, not an occupancy prediction result. Source metadata and compressed peak-file hashes are preserved in `oxtr-next-test-evidence.json`. Neither setup sample overlaps a conservative peak in the selected ENCODE assay. The existing source remains a candidate assay with quality warnings; no data-admission approval or fitted model is claimed.

The post-question status now recognises the completed setup records instead of calling that phase active. It is gated by matching validated GPU and QPU receipts. Original API receipts and original measurement datasets remain unchanged.

## Evidence and checks

- Recomputed interpretation checks, integer-count checks and original routing/receipt checks passed.
- Source coverage audit: ENCSR932ZMX / ENCFF651DQL / GRCh38, five OXTR-area intervals, zero overlaps for each of the original two samples. Reproduce using `tools/build-quantum-next-evidence.py` with the research directory.
- Official source checks: ENCODE experiment and assay-standard search records; JASPAR MA0139.2 page. These confirm CTCF, cell identity and assay context. Peak coordinates came from the saved public BED source.
- Full live browser verification at 1470, 390 and 320 pixel widths: goal and verdict before charts; next-test reader; inspectable seven-row evidence table; positive/inconclusive/discouraging rules; public-data versus wet-lab explanation; both samples and all eight input versions; atom keyboard selection; no overflow or page errors.
- Close and Escape restore the saved question, scroll position and keyboard focus. Opening the recorded evidence made no question request.
- One real staging OpenAI request returned HTTP 200, a provider response ID and study_linked status. The receipt appeared in the interface, opened the hardware report and then the next-test evidence. No worker source changed.
- Separate headless Chromium had process-level --mute-audio confirmed before navigation; all automated media was paused and every automated browser process closed.
- Final one-word reading-order correction received a focused live reader check. All nine checked staging assets match the committed source hashes; Vision JS remains preserved.

## Release boundaries

Only the existing Panthur staging deployment process was used. All website changes were committed before deployment. No new CUDA-Q or Aquila jobs were submitted, and no new hardware spend occurred. No biological predictor was fitted or evaluated. No production deployment occurred.

Final cache versions: Quantum iframe 15, biology CSS 12, biology JS 13, insight module 3, response module 5 and study module 5. Vision JS version 2 remains in the main Informatics page.
