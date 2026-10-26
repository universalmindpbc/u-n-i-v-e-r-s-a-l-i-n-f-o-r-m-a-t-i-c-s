# Universal Informatics — delivery sequence

## Interactive experiment — 20 October 2026

Ben approved implementation of the three-source selector (published studies, ENCODE, JASPAR), original DNA line, full-position circle and M9 circle, with Alpha/Beta/Gamma/Epsilon comparisons. The feature is deployed and verified on [permanent staging](https://staging.universalmind.tv/informatics/?atlas=1&record=fibonacci-m9-p24) at source commit `2334096f`. Production remains unchanged. Open **TF binding sites → Binding-site geometry**. All 36 source/view/sequence combinations passed at desktop and mobile widths, alongside the existing catalogue regression checks. All 38 changed HTTP assets match the committed source. The source catalogue and human-site project in Notion now link to the experiment, methods and verification reports. Release evidence: `research/informatics/oxtr-human-site-project/geometry-experiment-staging.json`.

Canonical methods and scripts: `Informatics/pilot-v1/oxtr-tfbs-v1/experiment-v1/`. Python 3.12.14 and Biopython 1.88 prepare 9,421 observations. Python, JavaScript and Wolfram agree on all 39,248 possible offsets and circle coordinates to nine decimal places. Biopython rechecks both published sequence placements, parses 1,019 JASPAR templates and validates 9,163 eligible JASPAR sequences; one crosses the saved reference boundary. The browser filters and renders these checked, precomputed data, with JSON/CSV/SVG/PNG exports. No live Python/Wolfram service or biological significance result is claimed.

The fixed mapping uses the saved GRCh38 OXTR window, follows the minus strand from genomic base 8779627, and selects the central base of each interval's visible intersection (first central base in OXTR direction for even widths). Original extents and clipping remain explicit. Alpha/Gamma/Epsilon retain their 24-position grids; Beta retains its nine-position grid and original phase. Dense visualisations use labelled count bins, with individual source inspection.

Next: Ben's staging review; refine any presentation issues; then agree a comparison statistic, matched controls and origin/anchor sensitivity before testing association. NVIDIA CUDA/CUDA-Q remains a discussion item, with no paid compute authorised or launched.


## Next session — OXTR visual comparisons and Biopython, 19 October 2026

Ben approved production publication of the OXTR data browser (staging release `ef68380e`) and asked to record the next work for his return this afternoon. The immediate next-session priority is Python illustrations of the existing binding-site data in three views: **original DNA line → full-position circle → Modulus 9 circle**, compared with **Alpha, Beta, Gamma and Epsilon of M9 P24**.

The intended Atlas feature combines **Wolfram calculations and Biopython-based sequence analysis** with clear visualisations, source inspection and matching data exports. The Biopython feature remains planned; this note does not launch it. Ben also explicitly asked to discuss **NVIDIA products, services, APIs and compute, especially CUDA and CUDA-Q**, when he returns. Bring this up in the next session alongside the illustrations and Biopython feature.

[Next-session brief and task checklist](oxtr-human-site-project/next-steps.md). Resume illustration and feature decisions with Ben. Earlier milestones below remain historical context; this entry identifies the next agreed work.

## Internal research perspective — 18 October 2026

Current public focus: the elegance of M9 P24 and the subtle native variation of M18 P24. Add the approved native-sequence observations to their Geometry Summary and Beta sections, with the same content in Notion and versioned JSON.

The following perspective is internal roadmap context, not public Atlas copy or an established biological finding:

> We hypothesize that the Beta sequences across M9–M1001 are mitotic fingerprints that may be of some use in biological compute for efficiency and accuracy in the future. We don't know. We can feed them into mitotic challenges in bioinformatic simulations in the future to see if this assists.

### Founder’s initial insight — internal discussion

> The Fibonacci sequence scales, within reason, harmonically and perfectly across expanding moduli, at least 11.1% of the time.

Preserve this as Ben’s initial perspective for future investigation. The measured reference is 111 matching multiples of nine among M1–M1001 (111/1001, approximately 11.1%): their Beta sequences recover the original M9 cycle under exact reduction. “Harmonically and perfectly” is the founder’s exploratory language, not an additional property tested by this audit. The shuffle probability remains a separate calculation.

Future investigation: preserve each native Beta sequence and its alignment; define suitable inputs for mitotic simulation challenges; compare efficiency and accuracy with explicit baselines and controls. Record beneficial, neutral and negative results. The initial biological investigation remains focused on OXTR.

Interpretations of Beta sequences in M19–M1001 and higher-dimensional geometric-space hypotheses are deferred. Keep them within internal discussion; no modulus-to-spatial-dimension mapping is established or added to the public record. This roadmap does not launch simulations or paid computing jobs.

## 1. Complete M1–M24 pilot — ready for Ben’s staging proof
- Verify native Pisano periods and nested sequences with Python and Wolfram.
- Finish versioned JSON, separate diagrams, Notion records and assessment placeholders.
- Verify all 24 records on permanent staging, including navigation and media continuity.
- Ben proofs the pilot before production or expansion.

## 2. Universal One in the Atlas — approved next, gated on pilot proof
Reuse I AM's Universal One voice AI, with an unobtrusive Ask Universal One control and optional text/voice. Keep the current Atlas record as context.

Ground answers in versioned JSON and reviewed literature; cite records and sources. No model retraining is required to add dataset records. Separate computed findings, hypotheses and biological evidence. Expose verification scope and unassessed status accurately.

Initial evaluation: exact period/sequence questions, M9 versus M18 comparisons, cross-record counts, phase-preserving Beta answers, Epsilon qualification, and questions unsupported by evidence. Verify the full GPT workflow on permanent staging: request, server, model response and rendered answer. Audio QA uses a dedicated process with --mute-audio; any audible test requires immediate approval.

Deployment: staging → Ben proofs → explicit production approval. No deadline or recurring automation is set; this is the next project milestone after M1–M24 proof.

## 3. Research API — after Universal One
Expose read-only access to the same versioned JSON and supported research queries used by the Atlas and Universal One. Return record identifiers, rule definitions, dataset version, verification scope and evidence provenance. External access does not permit changes to the research database. Define authentication, limits and API documentation before release.

## 4. MCP connector — after the API
Allow scientists to query the research through their own compatible AI assistants. Wrap the research API rather than maintain a separate dataset. Support record retrieval, sequence and geometry comparisons, and counts of individual and jointly occurring patterns. Preserve the distinction between computed findings, hypotheses and biological evidence.

## 5. Scale toward M1–M1001
Use the proofed rules and pipeline. Count individual patterns and joint occurrence, distinguish exact M9 value matches from geometric matches, and catalogue additional patterns. Scientific significance remains unassessed until appropriate analysis and evidence review.

## 6. Agent-to-agent collaboration — later phase
Explore exchange of research tasks and proposed findings between Universal One and external agents after the API and MCP are established. Agree a concrete workflow and review process before implementation; external proposals must not silently modify verified records.

These are ordered project milestones, not calendar bookings or recurring automations. Sequence approved by Ben on 16 October 2026: finish and proof M1–M24 → Universal One → API → MCP. No dates have been assigned.

### Scientist connection experience — planned with MCP
Use one unobtrusive **Connect your AI** entry point. Candidate destinations: **ChatGPT, Claude, Gemini, Grok, DeepSeek, Perplexity, Muse (Meta)**, plus **Other AI / MCP**. Keep Universal One as the no-setup option within the Atlas.

Each destination gets an accurate compatibility label and a short guided flow: choose client → setup or supported installation link → authorise only if required → run a sample research question. A login button alone does not install an MCP connection. Do not imply endorsement, universal availability, or one-click installation before the exact account/client journey is tested. Use read-only public access where appropriate; account-specific features can use scoped authorisation.

Official documentation checked 16 October 2026:
- ChatGPT: custom MCP setup and reviewed plugin publishing: https://developers.openai.com/plugins/deploy/connect-chatgpt and https://developers.openai.com/plugins/deploy/submission
- Claude: remote custom connectors: https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp
- Grok: custom connectors: https://docs.x.ai/grok/connectors
- Perplexity: custom remote connectors: https://www.perplexity.ai/en-GB/changelog/what-we-shipped---march-13-2026
- Gemini: remote MCP in developer API; consumer-chat setup not verified: https://ai.google.dev/gemini-api/docs/function-calling
- DeepSeek: tool calling and third-party client integrations; consumer-chat setup not verified: https://api-docs.deepseek.com/guides/tool_calls/
- Muse (Meta): include in compatibility investigation; exact external connector installation remains unverified: https://about.fb.com/news/2026/09/introducing-muse-personal-ai-agent

Recheck current platform documentation during implementation. These are roadmap options, not live integrations.

### Scientific computing ecosystem — planned workflows
Connection page: group **Connect your AI** separately from **Research platforms**. Show Available, Planned, or Explore documentation according to tested capability. These are prospective integrations; do not present unconfirmed organisations as established partners.

- NVIDIA Nemotron / NIM: tool-enabled agents querying our read-only API/MCP, with endpoint/model compatibility verified.
- NVIDIA BioNeMo: model-specific biomolecular workflows using actual molecular/omics inputs; no assumption that Atlas numeric CSV is a valid model input.
- NVIDIA PhysicsNeMo: reproducible Python notebooks with defined physics equations, inputs, boundary conditions and validation.
- NVIDIA CUDA: GPU acceleration for sufficiently large simulation/statistical workloads. Native M1–M1001 arithmetic itself does not require GPU compute.
- NVIDIA CUDA-Q: hybrid quantum-classical algorithms and GPU-accelerated quantum simulation for explicitly defined Hamiltonians. Keep physical claims separate from model results.
- D-Wave Leap: retain the approved quantum annealing investigation as a separate workflow.
- AlphaFold: prepare model-specific molecular JSON only after a real molecular target/sequence is selected; compare resulting structures with Atlas features using controls. Not a direct Fibonacci CSV upload.
- Insilico Medicine PandaOmics: gene/protein measurement matrices plus sample metadata in CSV/TSV; Atlas data is a comparison source, not an omics matrix.
- Insilico Medicine Chemistry42: future molecular design/optimisation workflows with defined biological targets.
- Amazon SageMaker: reproducible notebooks and managed machine-learning jobs using versioned Atlas exports, with required biological data and explicit evaluation criteria.
- Amazon Braket: quantum simulation and supported quantum-hardware experiments using explicit algorithms; prepare research notebooks rather than imply a universal CSV upload or one-click account connection.

No paid cloud jobs, GPU/QPU reservations, account authentication, or partner outreach is authorised by this roadmap. Implement and proof the interface after the agreed M1–M24 → Universal One → API/MCP sequence. CSV and JSON access remain independent of AI-account connections.

## Collaborative research submissions — planned after UO and API/MCP

Add a researcher-controlled Submit a finding workflow: draft → reproducibility checks → review → publish a versioned result → optional integrity proof. Include author/ORCID, Atlas record IDs, dataset and rule versions, native versus reduced sequence context, hypothesis, methods/code/environment, results (including negative results), limitations, peer-reviewed references, review status, licensing/access choice, and links to subsequent corrections or replications. Every publication gets a stable ID and content hash; corrections create linked versions rather than silently overwriting history.

- Sage Bionetworks Synapse is the preferred collaboration destination: data/code sharing, teams, provenance, wiki discussion and DOI support. Build an export first, then authenticated publishing after API/access validation. Researchers review the package and its permissions before publishing. https://www.synapse.org (platform; confirmed by Ben). Documentation: https://sagebionetworks.org/platform/synapse
- Triall Blockchain Proof is a proposed optional integrity adapter. Its product describes hashing and timestamping documents, and a Verifiable Data Integrity API. Confirm current access, cost and integration terms before implementation. Store the returned receipt with the result version. Keep research files in the governed repository; anchor a hash rather than participant data. Timestamp integrity does not establish correctness, novelty or biological significance. https://triall-platform.webflow.io/products/blockchain-proof
- Bridge is a later participant/sensor study integration, not the primary findings repository. Current Bridge documentation says it was developed by Sage and is now operated by GRIP. https://developer.sagebridge.org/

Status: planned, no accounts connected, no external findings published, no partnership claimed. Review code and data licensing before presenting the project as open source. Preserve M1–M24 proof → UO → API/MCP ordering; this collaboration layer follows those foundations.


## M1–M1001 Atlas expansion — 18 October 2026

The complete range now includes 1,001 records and all 111 multiples of nine. The Research record selector begins with M family, followed by numerical modulus order. Each family member retains native Alpha, Beta, Gamma and Epsilon assessment alongside its phase-preserving M9 view. The shared register remains 14 studies, including four HERO references; 3,003 WSS tier slots remain unscored.

Current JSON entry points are pilot-v1/index.json and pilot-v1/dataset.json (a versioned manifest). Each record file remains independently readable; the full bundle contains all 1,001 JSON records. Formation intervals in new records use endpoints into shared Alpha to avoid duplicate arrays, with exact sums and reconstruction rules retained. The previous 48-record dataset and bundle remain archived.

Validation: Python checked all native returns and 646,188 compact formation intervals; Wolfram independently passed 8,008 core comparisons. Desktop and mobile candidate journeys covered M1, M49, M54, M99, M750, M999 and M1001, both family comparison modes, all four sequence controls, and literature/genomics lightboxes. Deployment uses the existing Panthur FTP script and host; sequential curl connection reuse and manifest checks accommodate the expanded asset count. Production awaits explicit approval.

## Complete CSV export — 18 October 2026

The CSV route includes an overview with direct native/M9 PNG and SVG URLs, 1,001 complete typed CSV records, a full image index, publication table and shared audit CSVs. The complete CSV ZIP contains no JSON files. Its CSV reader reconstructs every record exactly, including nulls, empty containers, sequences and nested calculations. The overview remains included in the JSON/source ZIP as well. Original mathematical JSON records are unchanged.

Website entry points: Download → Complete CSV bundle; per-record sidebar → Record CSV; Geometry summary → Complete CSV bundle. The existing publication register and Notion record data are unchanged. This export addition requires staging proof before production publication.
## OXTR source collection — 19 October 2026

Ben authorized broad literature/database collection and matching local, JSON, CSV, Atlas and Notion resources. The next visualization retains the original DNA line, full-position circle and separately labelled M9 circle. New motif scans, Atlas comparisons and significance calculations remain deferred.

The initial supplement contains 24 publications, 43 literature evidence records, the GRCh38 OXTR locus plus 10 kb on each side, 217,857 existing JASPAR predictions (9,164 at the default display filter), 1,019 motif templates, 255 ENCODE occupancy clusters with 751 experiment links, and 37 ORegAnno regional annotations. These are different evidence units, not one confirmed binding-site total. The original 14-study research register and all 1,001 mathematical records remain unchanged.

Notion: https://www.notion.so/3fd7b9c99c5681239265feb88ed38d68 — a dedicated database contains the same 43 evidence records, with species/evidence filters. Bulk positional data remain in the linked complete JSON/CSV downloads.

Source collection: `Informatics/pilot-v1/oxtr-tfbs-v1/`. Builders and validation live in `research/informatics/pilot/`; the same supplement is included in both complete M1–M1001 export bundles. The catalogue opens as a lightbox through the existing research context and register. TRANSFAC access, legacy individual coordinates, further database exports and full systematic screening remain documented gaps.

Staging proof: https://staging.universalmind.tv/informatics/?atlas=1&record=fibonacci-m9-p24 . Production approval is separate.
## Binding-site introduction — approved, 19 October 2026

Ben approved “Understanding Binding Sites and our approach” as an optional link inside the TF binding-site lightbox, opening the existing approved teaching graphic. Keep the literature and source data prominent, retain About’s compact scope, and preserve the active experience. The graphic explains protein recognition, DNA order/spacing/orientation and the Beta M9 P24 reference. A brief explanation distinguishes the 15-site promoter study, ENCODE occupancy clusters and computer-predicted motif matches. The next teaching illustration is the original DNA line → full-position circle → Modulus 9 circle; it is not implemented in this update.


## Human OXTR site recovery and harmonisation — 19 October 2026

Ben authorized a focused project to recover human binding-site sequences and positions, combine imports under a consistent schema and resolve overlapping evidence. The original 15-site human paper is the first recovery priority. Publisher full-text access remains unresolved; no purchase or author contact was made.

The first pass preserves every imported observation, groups identical GRCh38 intervals, records literature-to-database overlaps and known evidence reuse, and maps the published US-1 and US-2 sequences by exact local reference matching. Laboratory, binding and functional evidence are overlapping categories; previous zero values for the narrow tested-candidate label did not indicate no human testing. JSON, CSV, Notion and Atlas use the same derived fields. Human and animal records remain distinguishable. A total of unique confirmed human sites and all DNA-to-M9 comparisons remain open.

Project: https://www.notion.so/3fd7b9c99c56815a8b84e6839dc1d3ed

## OXTR data browser — approved and published, 19 October 2026

Ben approved the reviewed staging release `ef68380e` for production. The unchanged website source is live at https://universalmind.tv/informatics/?atlas=1&record=fibonacci-m9-p24 . Production release tag: `production-2026-10-19-1001`. All 77 changed assets (including complete data exports) matched the approved checksums. Desktop/mobile dataset searches, score filters, pagination, JSON/CSV exports and the preserved lightbox experience passed.

The next-session brief and checklist at the top of this plan cover Python illustrations, all three coordinate views, Alpha/Beta/Gamma/Epsilon comparisons, the planned Wolfram + Biopython Atlas feature, and the requested NVIDIA CUDA / CUDA-Q discussion. These remain planned for Ben’s return.


## Connector implementation — 24 October 2026

Ben confirmed Synapse and DeSci accounts are registered. DeSci Publish replaces the earlier Triall proposal. Account credentials, destination projects and draft IDs are not committed. The first release exposes a read-only research API, MCP, A2A direct-message research, client guides and lossless export tooling on staging. Synapse publishing and DeSci private-draft upload stay separate from the public connectors. No autonomous research publication is enabled.

See `connectors/README.md` for the current implementation, setup boundaries and verification.
