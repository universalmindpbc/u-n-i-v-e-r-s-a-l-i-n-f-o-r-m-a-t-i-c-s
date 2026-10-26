# Approved Nature/Frontiers production release

Release source: c819836931015956a05f2805f0c30fb2f94a798d.
Previous production: 309367a9515024f2f2eeb72d519f0cbc4a5c6747.
Ben explicitly approved all current staging work for production in this turn. No website source changed after that approval.

The existing `./deploy-panthurlive.sh informatics` publisher completed all nine changed uploads. The isolated checkout excluded unrelated work. All nine changed assets match local, staging and production SHA-256 hashes at the actual versioned URLs.

Browser verification passed at 1470, 1024, 390 and 320 pixels in video, black and white modes. It checked 781 unique publication records, Nature Portfolio 49, Frontiers 42, their 91-record union, individual-journal overlap deduplication, gene filter conjunction, URL persistence and exports. New General and Vision chapters and the Language epigenetics trust score passed. The nonrandomized music study remains outside the randomized trial filter; the oxytocin fear study preserves its negative finding. No page errors occurred.

Desktop video, mobile white and General-page screenshots were visually reviewed. All automated media ran in an explicitly muted isolated Chromium process; pages and the browser were closed after verification. This release has no OpenAI application change and does not claim a fresh live Universal One test.

`tools/check-nature-expansion.py` passed: all 990 prior paper placements and the approved assessment copy remain unchanged. The pre-existing legacy extraction checker exception is documented in the staging review, not represented as resolved here.

`verification.json` records interaction tests. `assets.json` records the source commit, nine hashes from each environment and matching status. The production manifest is committed with this audit. The new implementation Markdown is planning only and is not deployed as a website feature. Full article files remain local.

The annotated production tag uses a UTC timestamp. Remote main, staging and the peeled tag are verified after the audit commit is pushed.
