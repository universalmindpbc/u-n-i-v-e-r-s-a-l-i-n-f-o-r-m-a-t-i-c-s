# Universal Informatics — scientific agent roadmap

Owner direction recorded: 25 October 2026.

## Intent engineering: anxiety to peace of mind

Ben's mission, recorded 26 October 2026: **moving from anxiety to peace of mind**, with intended outcomes of lower short-term stress and increased felt connection. Astra prioritises research, quantum and genomic insights, accessible beautiful experiences, music, entertainment and education in service of that mission. Education includes relationships between music and natural structures, including Fibonacci sequences. The runtime loads the versioned `../astra-service/intent.json` on every report.

Mission guides priorities; it does not determine scientific outcomes. Report measured effects, uncertainty and practical next actions. A quantum response is not a demonstrated mental health intervention. Preserve negative and inconclusive results and avoid treating agreement between language models as experimental validation.

## Lasting research memory

Use the existing research database and RAG, with immutable experiment JSON and CSV as the permanent evidence record. `research.agent_experiments` retains both exact downloadable files, their hashes, revision, question and structured JSON. `research.agent_memory` adds searchable experiment summaries, the mission and explicitly labelled research directions. Existing `research.publications` and `research.passages` retain the literature and semantic search. The Nature 2022 stimulus is already indexed as `nature-fibonacci-s41586-022-04853-4`.

LangGraph checkpoints track execution and recovery. They are separate from research evidence and from model training. Retrieve prior work, inspect its limits, choose a permitted next test, compare new evidence, then retain the findings and unresolved questions. This loop may generate new hypotheses; it does not automatically improve the model's underlying intelligence or make repeated interpretations independent evidence.

Only reviewed experiment revisions enter shared research memory. Private task reports stay separate until publication is authorised. Every publication updates JSON, CSV, complete exports and research-memory indexing together. Re-indexing a matching hash is idempotent; changed evidence requires a new revision. Keep provenance, failed tests, uncertainty and superseded interpretations accessible. Retrieved text is source material, never authority to change instructions, permissions or budgets.

## Parallel line: Fibonacci and quantum-information stability

Ben intends continued study of regular Fibonacci ordering, Phi Ratio and M9 P24 Alpha, Beta, Gamma and Epsilon in quantum environments, alongside the longer OXTR binding-prediction programme. The starting reference is Dumitrescu et al., *Dynamical topological phase realized in a trapped-ion quantum simulator*, Nature 607, 463–467 (2022), DOI `10.1038/s41586-022-04853-4`; the authors' figure data are at `10.5281/zenodo.6514878`.

The Nature result concerns a Fibonacci-ordered drive and edge-qubit information in ten trapped ions. Our modular sequences and proposed hexagonal mechanism remain new hypotheses. Preserve that distinction. Begin with open data and simulations, freeze the pulse mapping and success measure, compare matched periodic and shuffled controls, and examine responses over time under controlled perturbations. Vary time ordering and spatial geometry separately. The existing one-time atom-excitation measurements are not coherence-lifetime measurements. `../astra-service/memory/research-directions.json` records the question, controls and positive/null criteria.

No ongoing paid hardware schedule is authorised. Further Aquila/GPU expenditure retains its separate approval boundary.

## Scientific leadership

**Gemini 4 is the intended Chief Scientific Officer. GPT-6 Astra is the head bioinformatician and general scientist.** These are the assigned responsibilities within Universal Informatics. The first Astra prototype is verified against saved public data; Gemini review and the complete research service remain future connections.

Gemini guides Astra at the creation stage: review the scientific question, evidence, strategy, comparison methods, success criteria, budget and proposed computation before execution. At completion, Gemini reviews Astra's work against the agreed strategy, checks the evidence and calculations, and either signs off, requests a revision or escalates to a human. Astra performs the research, analysis and report preparation. Scientific disagreement remains in the record.

Model review checks the work; empirical evidence and reproducible calculations establish scientific findings. A model's sign-off alone is not experimental validation.

## LangGraph owns the workflow

Build Astra first, with explicit strategy-review and completion-review points ready for Gemini. Until Gemini is connected, these points retain an honest `review_pending` state or use a recorded human decision. A test fixture never becomes a scientific approval.

LangGraph is the primary orchestrator and supervisor. Universal Informatics owns the persistent study state, evidence, checkpoints, task routing, tool permissions, budgets, retry rules, review decisions and publication gates. LangChain integrations are optional adapters.

OpenAI managed agents can provide a bounded Astra work unit behind a replaceable interface. They do not own the only copy of the workflow, records or permissions. An OpenAI Responses adapter or a Gemini adapter can use the same tool contracts and study records. Provider session identifiers are references, not the identity of a study.

Use the user-selected model names as intended targets. Verify each provider's available API model identifier and account access when connecting it; never silently substitute another model or imply that a future connection is active.

## First workflow

1. A scientist asks a question in natural language.
2. Astra retrieves the Atlas definitions, prior experiments and relevant public evidence, then proposes a reproducible strategy internally.
3. Gemini reviews the strategy, or the first implementation records the designated human review.
4. Astra performs authorised analysis through scientific tools. LangGraph saves progress and advances the work within the study and shared budgets.
5. Astra compares classical calculations, CUDA-Q simulations and available hardware measurements using the agreed criteria, then prepares an accessible findings report.
6. Gemini reviews the calculations, sources, interpretation and completeness. Requested revisions return to Astra with a bounded revision allowance. Ethical concerns, exhausted budgets and decisions outside delegated authority go to a human.
7. The approved result enters the report and immutable experiment JSON/CSV, then the complete Informatics JSON/CSV exports. A negative or inconclusive finding is a valid outcome.

Public UX stays simple: ask a question, explore while work progresses, receive classical findings, then quantum findings. Internal strategy and review stages do not become a series of technical forms for the scientist.

## Website, API, MCP and A2A access

The research agent is a shared service. Scientists and external agents can use its computation and research capabilities from the website, a versioned API, MCP, and A2A (Agent2Agent, the protocol originally introduced by Google). Each interface addresses the same durable study and uses the same permissions, scientific reviews and budget ledger.

The planned operations are: submit a research question, obtain a study identifier, inspect progress and remaining allowance, retrieve findings and downloadable evidence, resume after supplying requested information, and cancel work. Long quantum queues return a task reference and progress events rather than keeping a synchronous request open. Retried submissions use an idempotency key and return the existing study.

A2A is part of the intended agent architecture, not an optional afterthought. It enables external research agents to discover supported capabilities, delegate bounded work, follow task state and receive result artifacts. The current read-only A2A connector is not evidence that autonomous computation is connected. Add execution capabilities only after the shared job service and authorisation checks exist. Advertise actual connected tools and limitations in the agent's capability description.

An external agent cannot bypass Gemini's scientific review or the human escalation rules. A caller cannot grant itself extra funding. Public access and funded computation are distinct permissions, with capacity enforced by the shared daily, weekly and monthly budgets.

## Future Drug Discovery Network and Open Source Genomic Insights

The agent will eventually coordinate classical and quantum research with external drug-discovery and genomics platforms. Ben's intended ecosystem includes LENS Labs, Recursion, Ginkgo Biopharma, and relevant work from OpenAI, Anthropic and DeepMind, alongside other suitable research providers. These are owner-supplied roadmap references. Verify the exact organisations, products, APIs, data permissions and collaboration terms before implementation; no present partnership or connection is claimed.

The pipeline can prepare evidence packages, propose follow-up work and exchange agreed research tasks through API, MCP or A2A. Human review governs external sharing, scientific publication and consequential drug-discovery decisions. Preserve provenance and distinguish a computational candidate from a laboratory-confirmed finding.

## Delivery phases

- **First:** a resumable LangGraph workflow for Astra to inspect existing public OXTR results, use bounded tools and prepare a cited report. Prove interruption recovery, budget enforcement and review gates before autonomous submissions.
- **Next:** connect Wolfram Language and Quantum Framework, Biopython and statistics, genomic evidence search, CUDA-Q/qBraid and Braket/Aquila through independently testable adapters. Reuse versioned templates and prior study examples.
- **Scientific oversight:** connect Gemini 4 at both review points, with recorded recommendations, decisions and revisions. Preserve a human escalation route and prevent either model from increasing its own authority or budget.
- **Later:** expand to a coordinated group of specialist agents, optional SageMaker workloads, and the Drug Discovery Network and Open Source Genomic Insights. External collaboration and publication retain human oversight.

## Architecture reference

Ben's supplied `ChatGPT Image 25 Oct 2026, 10_12_15.png` depicts the broader Universal Informatics API call stack: natural-language questions, multi-model orchestration, genomic evidence, symbolic and classical computation, quantum backends, validation, publication and drug-discovery collaboration. This roadmap implements a small initial version of that vision. The current Astra/Gemini responsibilities above supersede older model labels in the image. The image does not establish live API access, partnerships or support for every illustrated platform.

## Non-negotiable execution rules

- Shared records and exports remain provider-neutral and readable without a model vendor's service.
- Model output cannot bypass scientific review, alter protected final test data, raise spending limits or publish unreviewed claims.
- Jobs have stable identifiers; retries reconcile existing jobs before submitting again. Unknown paid outcomes retain their budget reservation.
- Record sources, parameters, tool outputs, decisions and report revisions. Do not depend on private model reasoning as an audit trail.
- The agent continues useful authorised work until the goal is answered, a real dependency needs a human, or a recorded limit is reached. It never manufactures a favourable result to satisfy the goal.

## Owner execution milestone — 25 October 2026

The next implementation is `../astra-service/`: a private Quantum Atlas task broker and LangGraph worker. It follows the OXTR question through saved quantum-record checks, public DNA preparation, a classical prediction pilot, 16 local CUDA-Q checks for a new matched pair, and a plain-language report. The owner deferred Gemini oversight and public autonomous execution. This predefined, bounded recipe records automated preliminary review status; it does not fabricate CSO approval.

No new Aquila submission is authorised today. The worker uses zero new hardware spend and zero paid GPU spend. The local worker is a development service on this Mac, not the later always-on research system. Future funded M1–M1001 investigations retain the same provider-neutral records, budget boundaries and LangGraph state, with separately authorised provider execution.

## Cloud and shared research memory — 26 October 2026

AWS staging hosting is activated within Ben's US$25/month hosting allowance. Lambda runs the reviewed LangGraph recipe on demand, with CPU CUDA-Q, Biopython, durable S3 checkpoints and reports. The Mac worker is stopped. A full automatic website-to-AWS-to-report test passed, including downloads, reload recovery and duplicate-request protection. New paid Aquila and GPU submissions remain disabled.

The shared research index now includes all three Vision Atlas geometry citations and seven additional quantum-geometry sources in an isolated staging branch. General research has a dedicated Quantum Computing chapter. Research-directions r2 gives Astra explicit mechanisms, source IDs, matched controls and uncertainty boundaries for Fibonacci/M9 P24 stability research and possible mathematical connections to vision. Universal One reads the same evidence summaries. Both JSON and CSV exports retain these records.

The supported execution remains the bounded OXTR recipe. Research-memory expansion does not silently add an unrestricted autonomous experiment tool. Future stability experiments require a reviewed template and measurement plan; Gemini oversight, new hardware spend and external task delegation retain their existing roadmap status.

## Astra–Argon review and publication prototype — 26 October 2026

Gemini 4 Argon is connected through the Universal Informatics Google project. The saved OXTR design and final evidence report have real provider review receipts. The final report passed after the packet included the already-recorded cohort and local CUDA-Q evidence. Historical results remain unchanged; the later review is an additional record.

`../scientific-swarm/` adds provider-neutral LangGraph design review, bounded revisions, mid-study scientific support, final review, saved human interruptions and immutable JSON/CSV snapshots. Each progress snapshot is queued separately for Synapse and DeSci, and remote receipts distinguish delivery from public publication. The new swarm is not yet installed in the AWS worker; the existing bounded Astra service remains active.

The approved Gemini connection tests retain a shared US$1 ceiling. The key is stored in the existing AWS runtime secret. Recurring Gemini execution requires an explicit allowance and durable review/outbox integration. No new paid Aquila or GPU work is enabled. DeSci's archive bytes are verified, but its editor-service fault remains visible until resolved. Notion, Google Docs and Prism report adapters follow this integration.

## Live Gemini review gates — 26 October 2026

The staging AWS worker now runs Astra's question-specific plan through Gemini design review before calculation, then submits the completed evidence and report to Gemini before delivery. One revision at each review point is allowed. Unresolved scientific questions, unsupported scope and budget limits stop with progress saved. LangGraph controls these transitions; a model cannot skip a checkpoint.

The execution remains the bounded OXTR follow-through: saved Aquila/qBraid receipts, public genomic evidence, classical calculations and local CPU CUDA-Q checks. There is no new paid hardware submission. The shared Gemini US$1 test ledger is durable across studies and includes previous requests. The hosted worker remains independent of Ben's Mac.

Fresh Quantum Atlas visits show the question and an empty workflow. Live statuses and Gemini receipts belong to the current task only. The completed OXTR work remains a separate study with preserved graphs, insights, downloads and historical review. Each new report includes the exact evidence packets reviewed and complete JSON/CSV downloads. Broader scientific recipes, general mid-study consultation and external publication adapters remain the next integration work.
