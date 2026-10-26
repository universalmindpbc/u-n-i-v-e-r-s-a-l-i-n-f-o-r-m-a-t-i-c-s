# Approved Informatics production release — 26 October 2026

Ben approved the Vision research, Quantum Computing research and Quantum Atlas for production after the parallel crawler release completed. That crawler work completed at `433b38a0`. The three highlighted proofing notes were removed in `323c2687`, tested on staging, and recorded in `f3a11991` before production deployment.

## Deployed scope

- Approved website source: `f3a119913` and its ancestors; 1,897 changed Informatics files uploaded through the existing Panthur publisher.
- The approved robots, sitemap and LLM access files were deployed through the existing redirects target.
- Production research database: 821 publications, including three Vision Atlas sources and seven quantum-geometry additions; 1,001 Atlas records and 149 full-text articles. Existing cached embeddings were reused.
- Two immutable experiment records and four research-memory records imported with matching archive hashes. The Nature 2022 stimulus is indexed.
- Production Universal One Worker version: `c50980a0-bcea-48b6-a82b-81e3bc07ac8d`; dataset `research-20261026-geometry-v1`.

## Verification

- Quantum Atlas desktop/mobile: all three requested notes absent, four results dialogs open and close, no relevant browser errors, exact `clean-3` JavaScript URLs loaded.
- Research Repository: all ten additions, 821-record JSON and CSV downloads, Vision source panel, mobile layout and MCP retrieval pass.
- Two real GPT-6 Astra requests completed through the production interface and rendered answers with the new research citations.
- JavaScript-free reading journeys and layouts at 1470, 390 and 320 pixels pass. The generated catalogue contains 1,831 pages. Thirty-one reading/archive assets and the root robots/sitemap match local hashes.
- Thirty-two additional changed website and experiment assets match their production file hashes. The versioned complete-export manifest remains `43d49bbc4b8805d8f8995ddb53a27032791299be2d7c09ddfc0e735b95fce743`.
- Secret scan: 2,107 release files checked, zero matches. All browser checks used a separate headless Chromium process with `--mute-audio`, then closed the sessions.

The initial publication-index check counted the DOM before page loading completed. Waiting for the navigation load event resolved that test timing issue; all 821 entries and the complete file hash passed. No website change was needed.

## Separate ongoing work

The new Astra–Argon swarm prototype is not part of this production release. Real Gemini reviews and its stored credential are verified separately. The current AWS Astra execution service remains staging-only; this release publishes the approved research, recorded findings and existing production features, not new public autonomous execution. No new paid quantum or GPU jobs ran.

During verification, staging advanced to `f2cc53c2` with a later favicon, landing-link and DeSci support update, followed by new About wording in the parallel chat. Those later staging changes remain separate from this exact production version. Production and staging refs therefore intentionally differ. No parallel work was overwritten.
