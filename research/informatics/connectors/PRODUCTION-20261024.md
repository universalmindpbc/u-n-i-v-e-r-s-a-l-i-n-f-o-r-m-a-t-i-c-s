# Universal Informatics connectors — production release, 24 October 2026

Ben explicitly approved production publication after proofing staging commit `bd7fce6a69f7210b45c69f045bae7e13f5c7e346`.

## Published scope

- Page: https://universalmind.tv/informatics/connect.html
- API: https://informatics-universal-one.universalmind.workers.dev/api/v1/inventory
- MCP: https://informatics-universal-one.universalmind.workers.dev/mcp
- A2A: https://informatics-universal-one.universalmind.workers.dev/a2a/v1
- Agent discovery: https://informatics-universal-one.universalmind.workers.dev/.well-known/agent-card.json
- Source commit: `1422493ca9b28c71bb171b08c64f162ab83400ed`.
- Production Worker: `d006dfff-5b04-453b-b74d-f953a23efd9d`.

All website files remain identical to the approved staging version. Production configuration enables the same connector routes with its own 120-requests-per-minute IP rate limit. Existing production secrets were retained; no credential transfer was needed. QA scripts now accept environment-specific endpoint and output settings without changing the tested visitor journey.

The existing Panthur production publisher uploaded 14 changed files from its 11,270 tracked Informatics assets and completed successfully. This release adds no external research publication: the Synapse project and DeSci draft remain private.

## Verification

- Local connector boundaries, protocol lifecycle, rights filtering, pagination, CSV safety and rate-limit checks passed.
- All 38 research routing checks passed. The 100-turn conversation fixture remained bounded at 18,569 transmitted bytes from 10,887,374 original bytes, retaining source IDs without repeated evidence passages.
- Production MCP passed all eight tool checks through the official SDK. Inventory reports 811 publications, 1,001 Atlas records, 149 internal full texts, 146 articles eligible for public reusable passage access and 111 M9-family records. Complete M9 API data matched canonical JSON.
- Production A2A returned a cited research answer. An independent OpenAI Responses request successfully called the production MCP inventory and M9 probability tools.
- The live website sent a research request through the production server to OpenAI and rendered a successful cited answer. Main navigation opened Connect your AI in its existing lightbox. Provider guide dialogs, keyboard dismissal, theme selection, endpoint test, publication search and an 811-record download passed.
- Four browser widths passed without horizontal overflow: 1470, 1024, 390 and 320 pixels. No browser console errors were recorded. All automated browser sessions used process-level `--mute-audio`, paused media and closed on completion.
- All 14 changed website assets were compared against both production and staging responses and the approved Git source. The 28 comparisons matched exactly.

Evidence is retained in `production-verification/`: deployment identity, Panthur completion log, protocol results, A2A and OpenAI MCP responses, browser report, screenshots and asset hashes. The complete 811-publication/1,001-Atlas export audit remains in `verification/complete-export.json` and `verification/atlas-archives.json` from the approved staging release.

Provider buttons remain honest account-setup guides. Consumer-account installation, Synapse OAuth application onboarding, persistent A2A tasks and the future CORAL coordination work are not represented as completed by this release.

## Release record

The final release commit includes this audit, production evidence and the production deployment manifest. Its annotated production tag identifies the exact commit. Both remote `main` and `staging`, plus the dereferenced tag, are checked after the atomic GitHub push.

The preceding production baseline is `7f1de4ff9f75fbad0de3f3836ea273fe07182a87`, tag `production-2026-10-24-1232`; the preceding Worker version is `6c14d473-4f77-4280-a999-61a4e8153116`.
