# Universal One: Research access and greeting

24 October 2026 · staging only

## Change

The Informatics voice greeting is now exactly: “Hi. What research question can I help you with?” The I AM application is unchanged.

The existing Universal One instance now stays inside the Universal Research lightbox, with its planet at the lower right. The conversation panel is sized against the lightbox content area, including the header clearance. The same history and live voice session remain active across opening and closing Research. Dialog ownership follows opening order, including Research opened from the Phase Zulu network.

Source commits: 0079eba0, then 6cd83fad. Existing staging base 640859ac includes the Molecular Atlas / Quantum teaching work. No Molecular Atlas source was changed in this task.

## Verified

- Dedicated headless Chromium with --mute-audio before navigation; synthetic microphone only; no native fullscreen. Automated media was paused and browsers closed.
- Planet, chat panel and voice dock bounds at widths 1470, 1024, 390 and 320, with video, black and white reading backgrounds.
- Real UI text request through the staging server to OpenAI, with three displayed sources for recreational music making and gene expression. Active Sound library context was sent.
- Real voice connection, exact Hi greeting transcript and decoded remote audio. Silent verification does not establish subjective voice quality.
- Conversation and connected voice retained across closing and reopening Research; microphone released at End.
- One parent-owned assistant, including Research opened over the gene network, an open paper inside its iframe, and the standalone Research page at desktop and mobile sizes.
- Five frontend files matched live staging byte for byte. QA did not consume visitor allowances.
- Existing lifecycle, research routing and conversation-context checks passed. Panthur preflights passed.

The verification report records the initial click-handler conflict and its correction, plus a test-selector adjustment for All research. The separate nested-view continuation completed the remaining checks without repeating the paid AI requests.

Evidence: verification.json and the adjacent screenshots. The reusable browser harness is ../verify-research-access-staging.mjs.

## Release boundary

Ben will proof this version before authorizing the full production release, including the Molecular Atlas animations. Production remains at d82c7ae080a5236cd1029bd9605d7b4e3988feb4. No production deployment was performed.
