# OXTR binding-site illustrations and Atlas computing roadmap

## Next discussion after Ben proofs this version — 20 October 2026

- [x] Ben requested the infographic after reviewing the source explanations. Added a separate “How binding sites help make OXTR” link beside the existing explainer, opening the saved teaching image in a nested lightbox. Staging review remains pending.
- [x] Reviewed and reused the existing teaching draft, **How binding sites help make the OXTR receptor**. Cover what a binding site is, why a gene can have multiple regulatory sites, how proteins act together, and how binding relates to transcription. Explain that the number and use of sites depend on the gene and cell context, and that DNA position is not a fixed timeline of binding events.
- [ ] Ben proofs the new teaching entry point and compact comparison board. Discuss `preliminary-discussion-draft.md` before publishing any initial-findings call-out.

## Current source and outcome presentation — 20 October 2026

The current comparison is DNA on the left and original Atlas geometry on the right, with Blend off by default. This supersedes the earlier shared-only design described in the historical notes below. Source cards now explain published studies, ENCODE laboratory measurements and JASPAR predictions before the experiment. A visible evidence-priority guide starts with positive binding at a defined sequence, followed by regional association and predicted matches, while retaining assay quality and biological context. Publication status is not a binding outcome.

The plotted observations carry source-linked binding outcomes. US-1 and US-2 have reported positive binding results; ENCODE records have positive regional association signals; JASPAR records remain predictions. The record inspector separates assay, context, gene-activity effect and source-review depth. US-1 reporter activity is distinguished from native OXTR regulation, and the proposed US-2 expression role remains unconfirmed in the reviewed abstract. Full-text reassessment and recovery of the original 15 sites remain priorities.

`binding-outcomes.json`, its typed CSV and `mapped-binding-outcomes.json/csv` retain the interpretation for human and AI users. The latter covers all 9,421 mapped records without changing their source measurements or projections. Complete Atlas packages and filtered exports include the same information. No negative controls have been curated for this map; missing data is not a failed binding test. Outcome categories reserve explicit negative and mixed results for source-supported future additions.

Next: Ben reviews the source explanations, outcome labels and comparison design on permanent staging. Then define a test of geometric prediction with appropriate controls and sensitivity to coordinate anchors. Positive binding observations are biological anchors; two study elements alone do not establish predictive performance. Do not treat ENCODE region centres as identified motifs or apply the Fibonacci shuffle probability to DNA matches.


## Shared-graph refinement — 20 October 2026

Ben approved one shared graph for the DNA observations and selected Alpha, Beta, Gamma or Epsilon geometry. The original DNA line, full-position circle before compression, and M9 circle remain available for each of the three source collections. Both the full-position and compressed comparisons are part of the research question. Fixed origin, direction and native Atlas grids remain explicit; no rotation is fitted to the observations.

This revision replaces the previous side-by-side panels. M9 dot area is proportional to observation count, with counts and percentages shown. Dense full-circle views use outward count bars for 72 equal DNA sections; all original observations remain inspectable and downloadable. `experiment-v1/overlay-v1.json` and its typed CSV record the shared-graph rules. Source measurements and base-coordinate mappings are unchanged. The new browser checks cover all 36 combinations at two viewport sizes, exports, keyboard inspection and preservation of the lightbox context. Deployed and verified on permanent staging at source commit `82c2c688`. All 24 changed HTTP assets match the committed source. The actual research-package buttons and both complete Atlas bundle URLs deliver current files byte-for-byte. Notion links and explanations are updated and verified. Final staging evidence is recorded in `shared-geometry-staging.json`. Production remains unchanged.

Immediate next step: Ben reviews the design and interpretability. Agree the comparison statistic and controls only after that review. No biological prediction accuracy, new motif scan or enrichment result is claimed. The earlier task list below is retained as the original roadmap; preparation, coordinate checks, interactive views and exports are now implemented.

The accompanying teaching draft, **How binding sites help make the OXTR receptor**, is saved in the Informatics research workspace under `research/teaching/oxtr-transcription-v1.png`, with the full image prompt, scientific scope and sources in the adjacent Markdown file. It separates OXTR gene regulation, RNA copying, receptor production and activation by OXT. DNA order does not establish binding chronology. The image awaits Ben's review before website placement.

## Interactive experiment — 20 October 2026

Ben approved implementation of the three-source selector (published studies, ENCODE, JASPAR), original DNA line, full-position circle and M9 circle, with Alpha/Beta/Gamma/Epsilon comparisons. The feature is deployed and verified on [permanent staging](https://staging.universalmind.tv/informatics/?atlas=1&record=fibonacci-m9-p24) at source commit `2334096f`. Production remains unchanged. Open **TF binding sites → Binding-site geometry**. All 36 source/view/sequence combinations passed at desktop and mobile widths, alongside the existing catalogue regression checks. All 38 changed HTTP assets match the committed source. The source catalogue and human-site project in Notion now link to the experiment, methods and verification reports. Release evidence: `research/informatics/oxtr-human-site-project/geometry-experiment-staging.json`.

Canonical methods and scripts: `Informatics/pilot-v1/oxtr-tfbs-v1/experiment-v1/`. Python 3.12.14 and Biopython 1.88 prepare 9,421 observations. Python, JavaScript and Wolfram agree on all 39,248 possible offsets and circle coordinates to nine decimal places. Biopython rechecks both published sequence placements, parses 1,019 JASPAR templates and validates 9,163 eligible JASPAR sequences; one crosses the saved reference boundary. The browser filters and renders these checked, precomputed data, with JSON/CSV/SVG/PNG exports. No live Python/Wolfram service or biological significance result is claimed.

The fixed mapping uses the saved GRCh38 OXTR window, follows the minus strand from genomic base 8779627, and selects the central base of each interval's visible intersection (first central base in OXTR direction for even widths). Original extents and clipping remain explicit. Alpha/Gamma/Epsilon retain their 24-position grids; Beta retains its nine-position grid and original phase. Dense visualisations use labelled count bins, with individual source inspection.

Next: Ben's staging review; refine any presentation issues; then agree a comparison statistic, matched controls and origin/anchor sensitivity before testing association. NVIDIA CUDA/CUDA-Q remains a discussion item, with no paid compute authorised or launched.


Recorded at Ben’s request on 19 October 2026, Australia/Sydney. Resume together when Ben returns this afternoon. This is the next-session brief and task list; it does not start the comparisons or schedule an automatic run.

## Bring up when Ben returns

1. The three binding-site illustrations and comparison with all four Atlas sequences.
2. The planned Wolfram + Biopython feature for Atlas users.
3. **NVIDIA products, services, APIs and compute: CUDA and CUDA-Q.** Ben explicitly asked to discuss this on resuming; raise it alongside the illustration and Biopython work.

## Agreed direction

Use Python to create clear illustrations from the binding-site research data already collected, comparing the data with **Alpha, Beta, Gamma and Epsilon of Fibonacci M9 P24**. Show the same selected observations in all three agreed views:

1. **Original DNA line.** Preserve genomic positions, interval widths, order, spacing and strand. Display broad experimental regions as intervals and recovered sequence elements at their recorded extents.
2. **Full-position circle.** Wrap the same defined DNA window around a circle, preserving proportional positions and interval widths. Keep the origin and direction visible. This is a representation of the coordinates, not a claim that the DNA takes that physical shape.
3. **Modulus 9 circle.** Show the same observations under an explicitly defined M9 mapping, alongside the relevant Atlas sequence. Retain the original coordinates, show where observations coincide after reduction, and make it possible to trace a reduced mark back to its source.

Make all four Atlas sequences available for comparison. Preserve their existing values, order and alignment; use the verified Atlas records rather than redraw them from memory. Do not silently adjust rotation, origin or scaling to make a pattern fit. Agree which coordinate or interval feature is mapped before producing the comparison.

## Planned website feature: Wolfram and Biopython

Ben’s intention is to include **Biopython as a feature in the Atlas**, alongside **Wolfram-powered calculations**, for people using the website.

- **Biopython:** prepare and inspect DNA sequences, motif models and strand-aware sequence operations. Use it within a reproducible Python analysis pipeline; retain source identifiers, reference assembly, methods and parameters.
- **Wolfram / Wolfram Language:** symbolic calculations, modular arithmetic, sequence and geometry representations, and independent checks where appropriate.
- **Python plotting and the Atlas interface:** generate the scientific illustrations and let people select data, switch among the three views, compare Alpha/Beta/Gamma/Epsilon, inspect sources and download results.

These are complementary tools. A future public “powered by” statement refers to the capability actually implemented and verified. The Biopython website feature and binding-site comparison workflow are planned, not live in this release.

## NVIDIA discussion — CUDA and CUDA-Q

Explore how NVIDIA products and services fit the actual Atlas research workload. Review current official documentation when this discussion resumes and distinguish accessible services from proposed integrations.

- **CUDA:** identify candidate GPU workloads, such as large motif scans, many comparison runs or matched random controls; assess whether their size justifies acceleration beyond the existing Python/Biopython CPU workflow.
- **CUDA-Q:** discuss whether there is a well-defined quantum or hybrid computational problem worth formulating for the OXTR investigation, and what a simulation actually tests. Keep this distinct from drawing the DNA circles or applying modulo nine.
- Compare local hardware, hosted compute, APIs and SDK integration, including access, cost, reproducibility and the data passed between components.
- Choose a small benchmark and explicit accuracy/runtime comparison before proposing a public Atlas compute feature. Identify which calculations remain with Wolfram, Biopython and ordinary plotting tools.

This is a discussion and feasibility item for the next session. No NVIDIA account, paid compute, GPU/QPU job, API integration or quantum result is being created in this release.

## Existing material available for the first illustrations

- **2 published human DNA elements with recovered positions:** US-1 and US-2. Their reference placements were computed from published sequences; this does not constitute a new binding experiment.
- **255 ENCODE experimental occupancy regions**, with target, cell-type and experiment provenance.
- **9,164 JASPAR predicted matches** at the current score-400 display threshold; all **217,857** archived matches remain available.
- **37 ORegAnno annotated regions**.
- The saved **39,248-base human OXTR reference window**, GRCh38 chr3:8740380–8779628, 0-based and end-exclusive; OXTR is on the minus strand.

Keep these collections identifiable as separate layers, with their own count units and filters. The 2006 paper’s 15 reported sites remain in the literature collection; individual coordinates are still unresolved. Their recovery can continue alongside the first illustrations using the data already mapped.

## Next-session task list

- [ ] Raise the requested NVIDIA CUDA / CUDA-Q discussion when Ben returns; review products, services, APIs, compute options and a useful first benchmark.
- [ ] Confirm the first data selection and the labels for the three views with Ben; begin with a legible example before displaying dense collections.
- [ ] Specify the DNA window, reference strand, coordinate origin, interval representation and exact M9 mapping. Record these choices before comparing shapes.
- [ ] Build a reproducible Python/Biopython preparation notebook or script using the archived inputs. Record package versions, source checksums and parameters.
- [ ] Generate the original DNA line, full-position circle and M9 circle from one shared set of observations. Preserve positions and source links across all three.
- [ ] Add comparisons with each verified Alpha, Beta, Gamma and Epsilon sequence from M9 P24. Make any reduced-coordinate collisions visible.
- [ ] Use Wolfram for appropriate independent arithmetic and geometry checks, recording which results it actually verifies.
- [ ] Review the illustrations with Ben for clarity and interpretation before implementing the interactive Atlas feature.
- [ ] Define the user-facing Biopython feature: supported operations, controls, computation location, downloadable methods and truthful loading/error states.
- [ ] Keep the Atlas, JSON, CSV and Notion representations consistent. Export the figures as SVG/PNG and retain the underlying data and calculation steps for human and AI researchers.
- [ ] Define matched comparison data and statistical controls before making any claim of enrichment or biological significance. The existing Fibonacci shuffle probability is a separate model and does not become a probability for these DNA patterns.
- [ ] Stage the approved interactive implementation at https://staging.universalmind.tv, preserve lightboxes/fullscreen/music, and follow the existing proof-then-production workflow.

## Scope of this record

Today’s action is to publish the already approved OXTR data browser and save this plan. The illustration work resumes with Ben. No new motif scan, DNA-to-Atlas mapping, statistical result, paid computing job or scientific relationship is asserted by this note.

Resources already selected for the workflow: [JASPAR](https://jaspar.elixir.no/docs/), [TRANSFAC](https://genexplain.com/transfac-database/), [Biopython motif analysis](https://biopython.org/docs/latest/Tutorial/chapter_motifs.html) and [Wolfram BioSequence](https://reference.wolfram.com/language/ref/BioSequence.html). TRANSFAC data remain subject to available access and reuse terms.


## Comparison revision — 20 October 2026

Ben approved restoring DNA on the left and the original Atlas geometry on the right, with Blend off by default. Blend overlays the selected guide in the DNA panel and preserves the independent reference. Retain all three DNA views and all four Atlas sequences. The line uses mapped reference nodes, without decorative connecting arches. Use distinct orange observation diamonds, direct study labels, explicit count scales and deliberate selection. Node/bin inspection lists observations in original DNA order.

Before the change, audit source scope: identical 39,248-base windows across all sources. ENCODE pools 133 targets / 119 cell types / 444 contributing experiments; 41 of 72 centre-count bins are empty. JASPAR pools 872 motif models at score 400. The two published elements are an incomplete collection, so their blank areas are not evidence of absent sites. The audit and methods travel in both JSON and CSV bundles. No biological inference or new statistical experiment is performed.

Next: Ben reviews the comparison interface again. Agree on a specific factor/experimental context and prediction question before adding a statistical test. Do not choose a factor, rotation or window because it resembles an Atlas shape. No production approval for this revision has been given.
