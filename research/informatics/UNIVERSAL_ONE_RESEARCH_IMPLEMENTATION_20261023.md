# Universal One for Informatics — implementation plan

23 October 2026. Prepared after approval of the Nature/Frontiers production release and Ben's text/voice references. Initial plan. Implementation completed on staging on 24 October 2026; see universal-one/IMPLEMENTATION_20261024.md for the current inventory, architecture and verification. The original plan below is retained as the decision record.

## Decision

Reuse Universal One's existing identity, character, voice experience and connection code. Add an Informatics research profile to the shared system, with its own permitted tools, source collection and conversation namespace. Avoid a separate product fork and avoid changing the existing I Am experience while introducing the research profile.

The research service supplies relevant evidence at question time. This is retrieval, rather than a fine-tuning run. A model does not permanently learn the library from an upload. The approved public wording, quotations and assessments remain source records; generated interpretation is labelled and never silently replaces them.

## What exists, checked in this pass

- `i  a m/realtime-voice-assistant.js` mounts a character and voice panel within `[data-television-room]`. It uses WebRTC, microphone permission handling, captions, interruptions, audio ducking and cleanup. Its existing captions keep only eight visible lines. Text entry, durable research conversation history and source citations are new work.
- The production frontend points to the existing `iam-citizen-needs-inference` Worker; staging points to its existing Realtime staging deployment. The PHP endpoint is a fallback. Read-only code inspection, not a fresh live voice test, established this path.
- `cloudflare/iam-citizen-needs-worker.js`, `handleRealtimeClientSecret`, creates a `gpt-realtime-2.1` session with Marin voice and bundled I Am portal, wellbeing, programme and adaptive recommendation guides. That handler has no paper retrieval or pgvector integration.
- The connected Neon account has a project named `rmu-intelligence-demo` and a second generically named project. No Informatics project appeared in the account inventory. Project inventory does not establish which extensions or storage services are provisioned.
- Separately, the I Am repository records an older AWS citizen-data foundation with Aurora Postgres and encrypted S3 objects. Those documents concern consent and private citizen information, not a scientific article retrieval store. Its current AWS deployment was not checked in this turn. Do not conflate that foundation with the current voice path or with the Neon Re-Model project.
- The approved research export contains 781 unique publication records. The local catalogue contains 174 complete articles: 130 open-license files and 44 author manuscripts retained for text-mining review. All 174 retain source and rights metadata. No article files were sent to an AI provider in this work.
- The Atlas also contains versioned mathematical records and OXTR evidence. Preserve their own identifiers, versions, coordinate systems and native versus M9-reduced distinctions.

## Infrastructure recommendation

Use one dedicated Informatics Postgres database, with native full-text search and pgvector, plus private S3-compatible article storage. Reuse existing provider accounts after confirming the target project, region, access and incremental cost. Keep Re-Model and private I Am citizen data separate through database credentials, permissions and object boundaries.

Postgres holds publication identities, DOI/PMID/PMCID, journals and publisher groups, chapter placement, conditions, targets, study design, exact approved text, dated citation counts, assessment provenance, passage references, document permissions and versions. Full-text search and vector similarity search are two ways to search this same evidence. S3 holds original files, supplements and versioned extraction artifacts.

Pinecone and OpenAI File Search are alternative search implementations, not additional required databases. The first planned stack does not need either. Keep retrieval behind our own service contract so MCP and other AI clients can query the same collection later. Synapse remains the later collaboration destination, not a replacement for the serving database.

A first staging demonstration can use the existing versioned JSON and a local/server index before cloud provisioning. This enables interface and citation proof without making a database migration a prerequisite. Persistent shared research search is the reason to establish Postgres and object storage early in the full build.

## Work package 1 — reuse and prepare

1. Isolate the existing character/launcher, transport and transcript rendering behind a small host adapter. Remove assumptions about the music television room for the Informatics mount; keep the I Am adapter and its approved behaviour intact.
2. Define a server-selected Informatics profile. Client-supplied page names and route strings are context hints, never permission checks. Its allowed tasks include searching the library, explaining methods and findings, comparing studies, inspecting Atlas records, and identifying gaps. It does not receive private citizen-data tools, clinical-session tools or database write credentials.
3. Prepare a deterministic import from the 781-record export, narrative editions and full-text catalogue. Preserve every record and existing publication placement, exact approved passages and topic scores. Distinguish publication records, trial families, article versions, excerpts and full texts.
4. Normalize the three existing passage formats into stable passage IDs with article ID, section/page, text hash, extraction version, source URL and reuse permissions. Review rights at content level; an open metadata record does not automatically grant full-text provider upload. Keep the 44 manuscript texts outside general external-provider retrieval until permitted use is confirmed.
5. Prepare immutable SQL migrations, object manifests, import reports and rollback instructions. Do not execute a new billable infrastructure change until its concrete cost and target account have been reviewed.

## Work package 2 — research service

1. Load metadata and approved content into dedicated staging storage after the hosting decision. Reconcile counts and hashes against the production export. Preserve JSON/CSV exports as public access routes.
2. Implement shared read-only operations: `search_publications`, `get_publication`, `get_passages`, `compare_studies`, `get_atlas_record`, and deterministic filtered counts. Existing OR-within-filter and AND-across-filter semantics carry across.
3. Combine exact identifier/keyword search with semantic retrieval, then retain source context around each result. Include contradictory and null findings, population, species and measurement type. Never infer a gene-expression result from a hormone measurement or keyword alone.
4. Call the requested GPT-6 Astra model for research answers, subject to API account availability verification. Supply retrieved passages with stable citations and dataset version. Keep evidence from this repository distinct from general model knowledge and any explicitly requested external search.
5. Validate cited IDs and source links server-side. Return source cards containing publication title, journal, year, DOI/PMID, evidence location and accessible original link. Counts come from database queries, not model estimates.
6. Apply server-side access control, input validation, rate limits and restricted tool execution. A topic prompt guides conversation; it does not by itself create a guaranteed security boundary.

## Work package 3 — text and voice experience

Use Ben's two Codex pet screenshots as interaction references, while retaining Universal One's character and Informatics visual style. A compact launcher offers text and voice; the text composer also offers dictation.

- Text-to-text: type, submit, read the answer with citations.
- Voice-to-text: explicitly start dictation, review/edit the transcript, submit to the same research service, receive a written answer. No forced spoken output.
- Voice-to-voice: explicitly start a conversation, receive spoken replies, retain the transcript and source cards on screen.
- Keep one Informatics conversation across mode changes. Switching to text stops microphone capture and speech playback. Opening the page never requests microphone access.
- Reuse the existing Realtime voice transport and Marin identity; connect research questions to the same Astra research service through a restricted tool/backend path. The Realtime model and Astra are separate roles. A chained speech-to-text → Astra → speech path is an alternative if testing favours exact spoken/text agreement. A new voice stack is not required to begin.
- Preserve the selected Atlas record and current research filters as explicit context. Keep the launcher clear of navigation, downloads and reading controls at desktop and mobile sizes.
- Treat conversation persistence as a separate product/privacy decision. In-session continuity needs no automatic collection of private I Am conversations.

## Work package 4 — proof and delivery

Create a small representative evaluation set before ingestion is called complete. Cover exact counts, DOI lookup, condition/target combinations, full-text findings, unavailable full text, negative results, small studies, animal/human distinctions, direct gene expression versus other outcomes, and hypotheses versus measured Atlas findings. Include attempts to bypass the research remit, inject instructions from source documents, or request another project's private data.

Verify that citation passages support the answers and that unsupported questions produce a clear evidence gap. Test all three input/output combinations, microphone denial, interruptions, mode switching, transcript continuity, keyboard focus, mobile layout, all reading modes, error handling and a genuine staging request → server → OpenAI response → rendered answer. Browser automation runs in a dedicated process with `--mute-audio`; any audible listening test requires immediate approval.

Deploy the first complete candidate only to permanent Informatics staging with the existing Panthur workflow and the established backend deployment path. Ben proofs the result before any new AI feature reaches production. Expose the tested research service through a read-only public API and MCP in the next milestone. No new model account connector, collaboration publication or partner integration is implied by this first build.

## Concrete next deliverable

A staging Universal One that reuses the existing character and voice foundation, offers text and dictation, answers from approved research with clickable citations, retains Atlas context and uses an isolated research profile. Alongside it: a reproducible corpus import, database/storage design, permission register and an evaluation report. Cloud provisioning is a discrete step after account and cost verification; it does not prevent the reusable interface and ingestion work from beginning.

## Checked technical references

- OpenAI File Search: https://developers.openai.com/api/docs/guides/tools-file-search
- OpenAI Realtime: https://developers.openai.com/api/docs/guides/realtime
- OpenAI voice architectures: https://developers.openai.com/api/docs/guides/voice-agents
- pgvector: https://github.com/pgvector/pgvector
- PostgreSQL full-text search: https://www.postgresql.org/docs/current/textsearch.html
