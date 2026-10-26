# Universal Informatics research library

## Delivery scope

1. Preserve the complete readable research content from Sound (144 pages), Mantra (73 pages), and EMDR (19 pages), with source page references, original quotations/notes, links, methods labels, historical citation counts and section grades. Remove legacy branding from the reading experience. Keep ambiguous and non-article material as labelled source notes.
2. Build one versioned evidence collection with Sound, Vision and Language entry points. Prioritise genomics, gene expression and epigenetics while retaining physiological, neural, clinical and survey outcomes. A gene mention is a discovery tag; it is not evidence that its expression was measured.
3. Add a dated, reproducible research update covering 2022–21 October 2026. Record search strings, inclusion decisions and sources. Distinguish primary studies, syntheses, protocols, guidelines and non-peer-reviewed material. This is a curated research update, not a completed systematic review.
4. Preserve the original section evidence grades separately from article stars. Article stars follow the requested design shorthand: three for an explicitly double-blind RCT, two for an explicitly single-blind RCT, one for other confirmed peer-reviewed papers. Unconfirmed articles and non-research resources remain unrated. Preserve imported design labels and flag them as unverified until checked against the paper. The shorthand is separate from risk of bias, relevance and certainty of the results.
5. Add a compact About section and an accessible candidate-network diagram. Expansion follows independent validation of any OXTR–Fibonacci biological relationship. Label genes, receptor families, RNA, metabolites and enzyme complexes accurately. The graphic is a research roadmap, with no asserted molecular edges.
6. Integrate the library inside the active Atlas experience; retain current geometry, music, fullscreen container and existing OXTR studies. Add direct proof URLs and downloadable structured records. No AI integration or model training is represented as complete.
7. Verify source coverage, search/filter/deep-link/export behaviour, accessibility, responsive layouts and existing Atlas navigation. Use a dedicated Chromium process with --mute-audio. Commit, deploy with the existing Panthur staging script, verify live, push staging records. Production requires a separate explicit instruction.

## Record contract

Stable ID; title; source URLs/DOI/PMID; collection membership; category; target discovery tags; biological measurement; study design; species/tissue/sample context where verified; original source note; new editorial finding and limitations; source collection/page; rating origin; verification status; date checked. Preserve multiple provenance entries when duplicate citations are merged. Keep bibliography fragments and editorial notes discoverable even when they lack a resolvable article identity.

## Expansion and AI

Language begins with mantra, mindfulness and related source material, and expands into communication, psychotherapy, psychology, psychiatry and epigenomics. Separate whole-treatment associations from causal effects of particular words. Future gene-specific libraries use the same target index, beginning with OXTR. Future AI retrieval uses stable records and citations and distinguishes RNA expression, DNA methylation, protein concentrations and downstream outcomes. No claim of training or AI answer verification is made in this release.

## Import and verification record

The migration retains 236 readable source pages and 802 citation entries. Adjacent citations without a divider are separated at their historical citation-count lines. Consolidation by confirmed PMID/DOI produces 833 research records, including 41 new appraisals and 13 retained portal notes. These are record counts, not counts of independent experiments.

All body citation links are accounted for in the extraction audit. Three consolidated records retain original title/link conflicts and remain unrated. Unresolved source material stays searchable. Historical citations and section grades are preserved; new article stars never inherit a topic-wide grade. Article stars use indexed publication types and blinding descriptors, with a specific correction for the mixed EMDR/medication trial. The pilot near-infrared paper is not rated as an RCT merely because its conclusion recommends future trials.

The update is an abstract-and-metadata appraisal, with full-paper appraisal explicitly pending. The search log records nine discovery queries and their retrieval limits. Crossref verifies additional bibliographic identities without automatically assigning research stars. One retracted recent music paper was excluded. Null, mixed and adverse-exposure findings remain included. The two supplied OXTR papers are shared pathway foundations; they do not establish sensory-induced expression effects.

## Reproduction

Use Python with pdfplumber and pypdf. Set INFORMATICS_PDF_DIR to the supplied PDF directory and INFORMATICS_SOURCE_CACHE to an external working directory. The defaults point to the source collection used for this migration. The source manifest records file hashes.

1. Run tools/extract-informatics-library.py, then tools/build-informatics-library.py.
2. Export the existing I AM portal's IAMScienceResearch object to portal.json in the source cache; its original source is i  a m/science-research-data-v1.js.
3. Run tools/fetch-informatics-library.py to replay the dated Europe PMC queries and resolve source identifiers. The replay log is separate from the recorded original search log.
4. Run tools/finalize-informatics-library.py. Optional bibliographic enrichment uses tools/enrich-informatics-library-crossref.py, then finalization again.
5. Run tools/check-informatics-library.py. Browser verification uses tools/check-informatics-library-browser.mjs with INFORMATICS_CANDIDATE=1 before deployment, and no candidate flag for live staging checks. Both modes use the permanent staging domain and a dedicated muted Chromium process.

The published JSON is a versioned, downloadable evidence index. AI retrieval, model training and a completed risk-of-bias review are future work, not functions represented as operational in this release.
