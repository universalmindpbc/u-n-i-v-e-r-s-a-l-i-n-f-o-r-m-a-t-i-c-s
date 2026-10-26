# Quantum Atlas insights and evidence readers — staging

Source commits: `442379a9` (insights and lightboxes), `e6b9e2f3` (chart-axis alignment).
Base staging: `db97bf1e200081d08bb10e8cce87efff7052847a`.
Proof: https://staging.universalmind.tv/informatics/?entry=research&area=quantum

## Experience

- A prominent yellow initial-insights panel leads the teal OXTR study card, explains the preliminary observations, and relates them to the occupancy prediction question.
- CUDA-Q, Aquila and comparison controls open closable native dialogs. Close and Escape restore page position, question and keyboard focus.
- Named representation findings precede the numerical detail. Blue and mint keys, ordinary-input baselines, DNA windows, excited-state percentages and sampling intervals have visible explanations.
- The six-atom diagram and table remain open. Keyboard and pointer selection provide an explanation of each atom's count and percentage.
- Trust is stated by purpose: checked simulation for the ideal model; hardware measurements for observed device behaviour; independent biological validation for occupancy prediction. No calibrated overall trust score is claimed.
- Higher or lower excitation alone is not biological success. The independent held-out prediction benchmark remains unmeasured.

## Verification

- Local source and data checks recomputed the editorial observations from all 16 recorded runs. Existing routing and receipt checks passed.
- Dedicated headless Chromium launched with process-level `--mute-audio` before navigation. Automated media was paused and the browser process closed afterward.
- Exact staging journey verified at 1470, 390 and 320 pixel widths: all three readers, both DNA windows, all eight representations, keyboard atom selection, persistent close control, no horizontal overflow, scroll/focus/question restoration, and no page errors.
- One real staging GPT request returned HTTP 200 and an OpenAI response ID, displayed its study receipt, and opened the recorded hardware results. `live-openai-receipt.json` preserves that response. No worker source changed.
- Final layout-only correction verified on staging for GPU and Aquila charts at all three widths; zero and endpoint alignment errors were below one pixel.
- All six deployed source assets match local SHA-256 hashes in `staging-assets.json`.
- The 16-run data and receipts are unchanged. No new GPU or QPU jobs were submitted. No additional hardware spending occurred.

## Deployment boundary and integration

Deployment used only the existing `deploy-panthur-staging.sh informatics` Panthur transport. Source was committed before each deployment. The deployment manifest belongs to this release.

Quantum iframe version: 12. Quantum CSS: 11. Biology module: 10. Response module: 4. Insights module: 1.

The parallel Vision Atlas integration can preserve these Quantum files and the iframe version while adding its separate assets. No research.html or research.js change belongs to this release.

Production was not deployed or modified. Remote main remained `6d58d23e36fa2942e260a4eacd808c7eb68c3623` before pushing the staging record.
