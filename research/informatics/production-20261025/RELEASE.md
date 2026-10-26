# Universal Informatics production release — 25 October 2026

Explicit authorisation: publish the entire approved Informatics website. Approved source: a29a7368e04852916b92b837c639d70742798fbc. Prior production: 6d58d23e36fa2942e260a4eacd808c7eb68c3623.

The existing Panthur `informatics` deployment published 45 changed Informatics assets. Other I AM website sections were untouched. The separate Chromatic staging work is excluded from this release.

The approved worker source was deployed to informatics-universal-one, version fc0158ff-c1b9-4f78-bcbb-b6357ce6526e. Runtime override QUANTUM_PLANNING_ENABLED=true matches staging activation; the approved production TOML does not yet persist that flag. This must be retained on future deployments until the config is updated.

Production browser proof used isolated Chromium with process-level --mute-audio and closed all sessions. Quantum results dialogs, all eight methods across two samples, responsive layouts, shared navigation, Transcription, Molecular, Geometry, Sound, Vision, Language, library, About and Connect loaded. A real Quantum question returned a linked recorded result; a real Universal One request answered the M9 inventory question. No quantum or GPU jobs were launched.

Local model, routing, budget and insight checks passed. Seven agent audit tool tests passed. Live asset comparisons and API receipts are in this directory. Full-data export and autonomous-agent additions remain new staging work.
