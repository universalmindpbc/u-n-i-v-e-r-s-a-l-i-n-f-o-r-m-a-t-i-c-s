# Informatics production release — 24 October 2026

Ben explicitly approved publication of the entire completed Informatics staging website, including the Molecular Atlas animations. Approved staging: `6411df6646853c7da118ca8edb74ac3ee4b049b0`. Previous production: `d82c7ae080a5236cd1029bd9605d7b4e3988feb4`. Published source and staging audit: `85ecfba46449d93fa90c954ad69194f99388268e`.

## Production configuration

Source `a3e4ab3c` enables the existing Universal One interface on the production hostname, adds the dedicated production Worker configuration, and makes Atlas citation links use the active site origin. Approved research wording, grades, data and visual behavior are unchanged. Staging and production retain separate origins, rate-limit namespaces and usage counters. Both read the same research release.

The necessary credential transfer was initially blocked by automatic approval review. Ben then explicitly approved transferring the existing database credential, OpenAI API key and private QA token to `informatics-universal-one` in the existing Cloudflare account. The transfer succeeded; secret values never entered Git or public assets, and the temporary credential bundle was deleted. Active production Worker version: `6c14d473-4f77-4280-a999-61a4e8153116`. Research release: `research-20261024-v2`.

## Staging verification

The existing Panthur workflow uploaded the three configuration-related frontend changes after all preflights passed. Staging Worker version: `8d9cd994-0cf0-42bd-95d4-54c5a1de48e9`. The real AI test passed 17 layouts across four widths, including black/white/video backgrounds, Research entry from the Atlas and network, paper overlays and standalone Research. A real request reached the research service and Astra, and a cited answer appeared in the interface. Voice connected and greeted with Hi. The full live Quantum Teaching suite passed at 1470, 1024, 390 and 320 pixels. All 33 checked staged assets matched source bytes. Reports: `staging-ai.json`, `staging-assets.json`, `staging-molecular.json`.

## Production publication and verification

`./deploy-panthurlive.sh informatics` completed successfully through the existing Panthur FTP transport, publishing 27 changed assets within the 11,262-file Informatics scope. The scoped staging and production manifests match exactly. Unrelated production manifest entries are unchanged.

The production browser test passed all 17 Research layouts, including the theme modes, paper overlays, standalone Research and entry from the network. A real recreational music-making/gene-expression question reached the production API, received an Astra response and displayed two linked research sources. Voice connected through the production service, returned “Hi. What research question can I help you with?”, decoded remote audio, and preserved its session while Research opened and closed. Ending voice released its microphone stream. Automated QA did not consume the visitor allowance. Reports: `production-ai.json` and the desktop screenshot.

An additional production API check returned the M9 period with an Atlas citation pointing to `https://universalmind.tv/informatics/?record=fibonacci-m9-p24&area=geometry`. The live Quantum Teaching suite passed at four widths, verifying actual changing animation pixels, pause, mode separation, controls, camera interaction, state export, docking transitions and overflow. Desktop and phone screenshots were inspected. Reports: `production-atlas-citation.json`, `production-molecular.json` and the two Molecular Atlas screenshots.

All 33 downloaded production assets match committed source bytes, including changed Universal One and Molecular Atlas files, calculated fields, edition data and Atlas index. Reports: `production-assets.json`, `manifest-check.json`.

All browser checks used separate Chromium processes with `--mute-audio` before navigation. Synthetic microphone only; no native fullscreen. Automated media was paused and browsers closed. These checks verify functionality and decoding, not subjective voice quality.

## Release record

The production manifest and this audit are committed with the release. The annotated production tag identifies the final audit commit; `main`, `staging` and the peeled tag are verified against that same commit after the atomic push.
