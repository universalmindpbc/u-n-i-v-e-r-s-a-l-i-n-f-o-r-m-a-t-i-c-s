# Nature Portfolio additions and publisher filters — 23 October 2026

Ben authorised expansion across the therapies, candidate genes, and fundamental oxytocin, serotonin and dopamine relationships. New research is staged for proof. The approved metadata, About backdrop and multi-select filters were released separately as `309367a9515024f2f2eeb72d519f0cbc4a5c6747`, tag `production-2026-10-23-2311`.

## Scope

- 23 additions: 14 General, 4 Sound, 3 Language and 2 Vision. They include experimental studies, reviews and human trials; journal reputation does not set a trust score.
- 758 existing unique records become 781. DOI and PMID identity checks found no overlap between the new selection and the existing export.
- 27 existing records had Nature-domain or 10.1038 identifiers. One is American Journal of Hypertension, now published by Oxford University Press. The current Nature Portfolio group contains 26 existing records plus 23 additions: 49 total, including six in the journal Nature itself.
- 42 distinct existing records have verified Frontiers DOI identities. A stray Frontiers link on an American Journal of Psychiatry record is not treated as publisher identity.
- Journal filters support Nature Portfolio and Frontiers groups alongside individual journals. Multiple journal choices use OR; different filter categories use AND.
- Verified journal labels now appear consistently in the reader, search, statistics and exports. Existing quotations, summaries, topic headings, introductions and A/B/C assessments remain unchanged.

## Scientific distinctions

Each addition identifies design, population, measurement, citation snapshot and interpretation limits. Neurotransmitter activity, receptor interactions, circulating proteins, RNA expression and DNA methylation remain distinct. Animal evidence is identified explicitly. The oxytocin fear-extinction trial reports impairment; the meditation/vacation study does not establish a novice-meditation transcriptional advantage over the active control.

New General topics each meet the minimum of three peer-reviewed papers for a trust score of one. The existing Language epigenetics topic reaches three papers and receives one. The two-paper Vision mechanisms topic remains ungraded. No individual double-blind trial upgrades an entire topic without three qualifying trials. Citation counts are Crossref snapshots, not a quality score. Jadad appraisals remain unassessed.

The selection is targeted and non-exhaustive. DOC2 and DOCK2 are not silently equated. No new candidate-network causal edges or therapeutic claims are inferred from this import.

## Verification

`metadata-review.json` and `frontiers-metadata-review.json` record exact publication identity, publisher, journal, identifiers, citation snapshots and source locations. Original response caches retain abstract text locally; the Git audit stores hashes, not full abstracts. `data-verification.json` confirms preservation of all 990 existing paper placements and the complete assessment copy.

The old `check-informatics-library.py` fails its original-import equality assertion on unchanged baseline data. It predates the documented EMDR merged-record repair. This update does not modify either legacy input file. The new preservation check compares every existing chapter paper against the approved production commit and checks all new identities directly.

## EMDR link follow-up

The prior pass requested all 161 destinations: 75 HTTP 200, 53 PubMed HTTP 203 challenge pages, 32 publisher HTTP 403 restrictions and one DOI HTTP 502. Rechecking all 86 access exceptions resolved the temporary DOI error; the 53 challenges and 32 restrictions remain. These are not verified unrestricted reader access, and they are not classified as dead papers.

An old WHO PDF URL returned a generic DSpace page despite HTTP 200. The reader now points to the matching WHO publication page and NCBI Bookshelf title, confirmed by ISBN 9789241505932. The original text is retained. The Frontiers refugee RCT also has a confirmed publisher page and PMC alternative. The replacement URLs returned HTTP 200 with publication content.

Sources: [WHO publication](https://www.who.int/publications/i/item/mhgap-module-assessment-management-of-conditions-specifically-related-to-stress), [NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/sites/books/NBK190416/), [Frontiers RCT](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.00493/full), [American Journal of Hypertension](https://academic.oup.com/ajh/pages/About), [Frontiers journals](https://www.frontiersin.org/journals).
