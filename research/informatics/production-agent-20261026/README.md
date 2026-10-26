# Production release — 26 October 2026

Ben explicitly approved all staging for production. This release publishes the approved Informatics website, research repository, exports, Quantum Atlas, Gemini review interface and robots/discovery updates through the established deployment process.

Production: https://universalmind.tv/informatics/?area=quantum&entry=research

Staging: https://staging.universalmind.tv/informatics/?area=quantum&entry=research

## Deployment evidence

- `release-audit.json`: scope, source revisions, deployment totals and verification summary.
- `cloud-deployment.json`: existing AWS worker image, successful activation and Cloudflare Worker versions.
- `.panthur-deploy-manifest.tsv` at the repository root: deployed file hashes.
- `staging-index.json` and `production-index.json`: 70 approved research additions; 891 publications, 4,268 passages and 1,001 Atlas records. Existing full-text rows were preserved.
- `production/asset-proof.json`: 1,928 changed public text assets checked against source. HTML comparisons account for the existing CDN analytics and email-obfuscation transforms; both received and normalized hashes are recorded.
- `production/complete-export-verification.json`: all five complete-dataset parts match their declared lengths and hashes, totaling 219,849,744 bytes.

## Live agent verification

Production task `ac5191ae-7466-47ff-8532-7f36e2136bf1` completed a real browser request, Astra planning, Gemini design approval, public genomic analysis, 16 local CPU CUDA-Q checks, report generation, Gemini revision feedback, Astra revision and Gemini final approval. The reviewed insights appeared in the interface. Both download buttons worked; the CSV reconstructed every JSON field. Starting a new question cleared the preceding task statuses.

Gemini identified identical Fibonacci and Phi Ratio programs for one comparison window. Astra corrected the report to disclose that limitation. The compact review handoff retains program hashes, numerical checks, deterministic identical-program groups and a hash of the complete measurements. Full arrays remain in the downloadable study. The saved task resumed from its checkpoint without repeating the calculations.

Gemini review records name the provider model, timestamp, evidence hash, decision and requested changes. An approved model review records a scientific review of the supplied evidence; it is not independent experimental confirmation of a biological hypothesis.

Evidence: `production/live-proof.json`, `production/live-task.json`, `production/live-result.json`, `production/live-result.csv`, `production/live-export-proof.json`, and `production/live-reviewed-insights.png`.

## Boundaries and checks

- No new paid Aquila or GPU jobs ran. Prior measurements were retained; new CUDA-Q checks used the existing CPU worker.
- Astra used 10,282 accounted tokens across three calls, within the existing 24,000-token study limit.
- Shared Gemini test accounting reached US$0.904624 of the US$1 allowance. The ledger was preserved across deployments.
- The existing monthly worker attempt cap remains 120. Hosting stays within the approved US$25 ceiling.
- Both study exports are present in encrypted S3 storage (`AES256`).
- Production and staging session tokens are audience-bound; cross-environment use returned 401. The private session credential returned HTTP 403.
- Universal One returned a real, cited OpenAI answer in the production interface. MCP, the saved-study API and the A2A card passed read-only checks.
- Browser checks used a separate Chromium process with `--mute-audio`, then paused media and closed the process. Desktop, 390-pixel and 320-pixel layouts passed.
- Secret scanning covered changed source and release evidence without printing secret values. Historical public identifiers in expired 2019 publisher citation URLs are recorded separately from credentials.

Detailed evidence is stored beside this file. The release commit and annotated production tag identify the final audit snapshot.
