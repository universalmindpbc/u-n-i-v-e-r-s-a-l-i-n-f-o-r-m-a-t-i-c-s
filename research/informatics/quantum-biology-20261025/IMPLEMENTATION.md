# Quantum Atlas and shared Informatics frame

## Scope

Natural-language biological research replaces the technical-first Quantum Lab entry page. The accepted name is Quantum Atlas — Biological questions via Quantum Computing, immediately after Molecular Atlas. This release also fixes the user's screenshot issue and inconsistent main-page navigation.

One persistent parent owns global navigation, video/music controls, fullscreen and Universal One. Main destinations: About, Geometry, Transcription, Molecular, Quantum, Sound, Vision, Language, Universal Research, Connect your AI and Partners. Direct links to geometry, molecular, quantum, library and connection pages resolve into the same frame. Research and connection content no longer require an opaque primary modal. Transcription uses the original source-linked catalogue inside a translucent scrolling panel with a section dropdown. Secondary teaching/evidence dialogs remain optional.

Existing public research and raw observation content is preserved. JASPAR predictions remain distinct from ENCODE/BrainTF occupancy and defined sequence evidence. No GPU analysis was added to Transcription Atlas in this change.

## Architecture

`cloudflare/informatics-quantum.js` validates input, reserves a shared question, calls the existing server-side OpenAI Responses integration with a strict JSON schema, validates evidence IDs and runs a bounded deterministic compiler when explicit read fragments are supplied. No model-generated code is executed. Only user-supplied fragments can enter the exact solver. HTML output uses text nodes.

`informatics-quantum-model.js` uses n² binary variables x[read,position]. Squared one-hot penalties enforce one read per position and one position per read. Consecutive overlap rewards define a linear path. The penalty exceeds the entire maximum feasible overlap reward; QUBO and Ising preserve energies under x=(1-z)/2. Up to 720 feasible permutations are enumerated. All 512 assignments of the three-fragment fixture are independently checked against the constraint objective and Ising energy. This is a small error-free, oriented read model; it does not resolve reverse complements, sequencing errors or repeats.

`informatics-quantum-budget.js` uses the existing Durable Object class with isolated storage for public/QA planning. Transactional counters reserve all three UTC periods together. Monetary QPU limits are not configured and dispatch remains off. See the updated GATEWAY.md in the original Quantum Lab evidence directory.

The original two-atom numerical suite remains unchanged in its method and passes the 25-case SciPy comparison. Historical reference results are retained behind the optional verification view.

## Publication boundary

Only the existing staging Worker and Panthur staging website are targets. Production requires explicit approval after proofing. Record source, numerical and live browser/API verification, deployment manifest and origin/staging remote ref.
