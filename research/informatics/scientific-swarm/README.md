# Astra and Argon scientific review prototype

Owner direction: 26 October 2026. GPT-6 Astra performs bioinformatics and research. Gemini 4 Argon is Universal Mind's Chief Scientist. LangGraph owns execution, checkpoints, revisions, interruptions and publication state.

## Verified now

- Direct access to `gemini-4-argon` through the existing Google project, renamed **Universal Informatics**. The project identifier is unchanged.
- Gemini's design review of the saved OXTR investigation: approve.
- Gemini's final review of Astra's saved work: approve after the review packet included its cohort provenance and local CUDA-Q receipts. The first incomplete-packet review remains in the audit record.
- A bounded LangGraph design → review → execution → report → final review workflow. It supports scientific revisions and a mid-study request for Gemini's help. Unresolved decisions interrupt with saved state.
- Immutable JSON and CSV progress snapshots, queued separately for Synapse and DeSci. Each delivery requires a matching provider receipt; an uncertain upload cannot trigger an automatic duplicate.
- Nine automated checks covering review routing, bounded revisions, interruptions, restart-safe budgets, saved reviews and delivery reconciliation.
- The real Gemini review bundle is archived in Synapse folder `syn78381066`. All four downloaded files match their local SHA-256 hashes. The new DeSci delivery remains queued while its document service is reconciled.

The staging AWS worker now connects its fixed OXTR recipe to Gemini design and final review through `astra-service/review_flow.py`. The design review runs before fresh calculations; the final review runs before report delivery. The separate, general swarm prototype still provides the future mid-study support and publication adapters. Public API/MCP/A2A execution is a later activation.

## Execution contracts

`swarm.build(scientist, reviewer, checkpointer, publish)` accepts replaceable scientist and reviewer adapters. The scientist has plan, execute and report methods. An execution returns complete, support or human with actual tool receipts. The reviewer returns approve, revise or human with an evidence hash and provider response identifier. Neither model can grant itself permissions or change spending limits.

The scientist adapter must pass every evidence category used in its report: cohort selection and source hashes, quality warnings, analysis metrics, prior hardware receipts, new simulation receipts and actual spend. The first live review exposed an evidence-packet omission, not missing scientific work. `verify_live.py` retains these categories without resending large raw arrays. The full recorded study is hash-addressed.

A historical report retains its original review status and completion time. A later Gemini review is appended as a separate record. Cross-model review is an additional check; it is not an independent laboratory replication or proof of a biological mechanism.

## Funding and credentials

The connection tests have a single persistent US$1 reservation ledger. It includes failed or uncertain requests and never resets per study. Google prepaid credits were purchased by Ben: A$8, auto-reload off. No new Aquila or paid GPU jobs are enabled.

The Gemini key is stored as `gemini` in the existing `informatics-astra-staging/runtime` AWS Secrets Manager secret. Existing `openai` and `worker` entries are preserved. No credentials belong in source, report artifacts or browser pages. Persistent storage of the key does not authorise unlimited API spend.

The staging worker uses the existing US$1 test allowance, shared across all studies and persisted in encrypted S3. Every request reserves its maximum first; known usage settles against the provider receipt, while uncertain requests retain their full reservation. Reaching the allowance pauses work. Recurring Gemini funding still needs a new owner allowance; the existing US$25 monthly hosting ceiling covers hosting only. The general publication outbox remains separate from this fixed worker integration.

## Publishing as research progresses

`publication.Outbox.enqueue` saves preliminary design, execution and review snapshots, then a final record marked `model_review_complete` only after review approval. Every snapshot carries its own immutable revision. CSV uses JSON pointers and explicit value types, including empty containers and spreadsheet-formula escaping.

`ExistingConnector` reuses the repository's Synapse and private DeSci draft upload tools. It creates no new project, changes no sharing permissions and signs no blockchain transaction. Remote delivery is not the same as public publication. Destinations are configuration owned by Universal Informatics, never model-generated URLs.

The full website archive already has verified Synapse and DeSci/IPFS records. DeSci's document editor service still reports a missing document; this prevents claiming full editor/publishing health. Keep new DeSci snapshots queued until the target is reconciled. A receipt or successful HTTP response alone does not prove public availability.

## Staging integration and remaining scope

`astra-service/runner.py` is the live execution graph. Astra owns the question-specific plan and evidence-grounded report. Gemini can approve, request one design/report revision, or require human input. LangGraph enforces order and checkpoints. Neither provider can raise a budget, enable paid jobs or execute generated code. Unsupported research questions stop before the OXTR recipe runs.

Every delivered report includes the plan, provider receipts, both review records, reviewed evidence packets and hashes, actual calculations, source provenance and complete JSON/CSV exports. The page starts with a clean question and workflow. A current investigation never inherits the saved OXTR study's counts or review badges; the historical study retains its own findings and lightboxes below.

Remaining extensions: mid-study Gemini consultation in new scientific recipes; the general Synapse/DeSci progress outbox inside the worker; optional document destinations; and public API/MCP/A2A execution. New quantum hardware, GPU spend and recurring Gemini funding remain separately controlled.

Optional Notion, Google Docs and Prism reports remain destination adapters on the roadmap. The same evidence record feeds every format. Human review remains part of external collaboration and the future Drug Discovery Network and Open Source Genomic Insights.

## Tests

Run `python -m unittest -v test_swarm.py` in this directory using the existing Informatics agent environment. `verify_live.py` is an explicitly budgeted provider test against saved public work, not a new scientific experiment.
