# Reviewed OXTR study additions — 21 October 2026

This is the retained input record for adding BrainTF (PMID 38831039) and the OXTR enhancer study (PMID 39957428) to Studies. The reviewed public data does not establish a new precise TF contact sequence or a geometric association.

The BrainTF review inspected 13 author RDS peak datasets and 226 additional public portal peak files. GRanges starts were converted from one-based to zero-based; the exclusive end is the converted start plus width. The locus extract contains 286 TF/chromatin-protein observations and 104 histone-mark observations. The published browser subset contains 126 protein observations with one matching QC row, an author replication pass, both FRiP values at least 1%, author NOT-HOT annotation and no identified source-label conflict. The FRiP threshold is a transparent Atlas review filter. The four donors are shared across brain contexts.

The QC extraction is from Supplementary Table 2, ChIP-seq sheet (publisher attachment MOESM4), retaining original spreadsheet row numbers. Original file URLs, hashes and row indices remain attached to the observations. Full raw source files remain in the local research review; large RDS files and controlled individual-level reads are not published here.

The 2025 paper tests human candidate DNA in mouse reporter systems. OCE7's reported candidate interval differs from the Table 1 luciferase-primer span. Both remain in the retained review, with no single precise binding coordinate assigned. OCE1/OCE5 outside the current Atlas window do not expand its boundary. OCE2/OCE6 are negative reporter results in the paper, not negative TF-binding controls.

Run the pipeline from the repository root:

1. `python research/informatics/pilot/add_oxtr_brain_studies.py`
2. `python research/informatics/pilot/normalize_oxtr_sites.py`
3. `python research/informatics/pilot/annotate_oxtr_binding_outcomes.py`
4. `python research/informatics/pilot/package_oxtr_catalogue.py`
5. Run `check_oxtr_brain_studies.py`, `check_oxtr_catalogue.py` and `check_oxtr_normalization.py` from the same script directory.

The new study rows join the normalized observations. The original two-element Studies geometry remains intact, while BrainTF regions are searchable in the Studies data browser. Contextual ENCODE and brain cCRE overlap identifiers remain spatial annotations, not independent proof of regulation. The existing 14-paper general research register remains separate from the 26-publication OXTR catalogue.
