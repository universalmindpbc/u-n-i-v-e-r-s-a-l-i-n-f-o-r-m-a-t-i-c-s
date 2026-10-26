OXTR binding-site catalogue — 19 October 2026

This supplement collects published OXTR regulatory evidence and existing database records. It does not perform a new motif scan or compare the data with Atlas geometry.

Read catalogue.json or the lossless typed catalogue-fields.csv for scope, evidence, publications, source URLs, methods, caveats and unresolved work. literature-evidence.csv is the accessible row table; reported_count is an aggregate reported by a paper, not a new unique-site total. Null coordinates remain blank in flat CSV files. Literature positions remain verbatim; never treat a TSS-relative label or CpG coordinate as a precise motif interval.

Database region: GRCh38 chr3:8740380–8779628, 0-based half-open, Ensembl OXTR gene body plus 10 kb on each side. OXTR is on the minus strand. The reference DNA file is the genomic plus strand. Motif sequence_oriented follows the motif strand. Overlap queries retain records extending beyond the window; those sequences use the separately archived 60 kb reference extraction. No liftover or transcript realignment was performed.

JASPAR: 217,857 published predictions in the returned window; 9,164 meet UCSC’s score >=400 default display filter. The track’s underlying prediction method uses a relative PWM threshold of 0.8 and p<0.05; UCSC score is based on -100 log10(p), capped at 1000. The display filter corresponds approximately to p<=1e-4, without a locus-level multiple-testing correction. Prediction is not observed binding or proof of OXTR regulation. The complete returned set is retained, alongside the filtered CSV. 1,019 vertebrate nonredundant PFM templates are included; 1,003 different models occur among the returned matches. A PFM is a model, not an OXTR-specific site. JASPAR 2026 publication: https://doi.org/10.1093/nar/gkaf1209 .

ENCODE: 255 regional clusters for 133 target names, from the UCSC ENCODE 3 November 2018 release. Each cluster links to all its source experiments and cell types in encode-provenance.csv/json. Clusters represent broad occupancy and include coregulators. They are not 255 precisely located, functionally confirmed OXTR motifs.

ORegAnno: 37 overlapping regional annotations from the UCSC 2015 snapshot. Most factor/class metadata were not resolved. OREG1492306 is cross-referenced by the 2023 OXTR methylation paper. It is counted once in this table, not again as a new experiment.

Publication findings are factual summaries with source links. Full article text, figures, and licensed TRANSFAC data are not redistributed. The existing 14-study Atlas research register remains separate; its E03 circadian study is cross-referenced here, not a new independent study. Review depth and evidence type are recorded; numerical trust and WSS scores are unassigned for this supplement.

CSV archive: all tabular evidence, metadata, DNA bases, matrix counts, predictions, clusters and provenance are available without a JSON parser. catalogue-fields.csv uses the Atlas typed path convention (field,type,value,resolved_url); source/csv_records.py restores its nested structure. Flat tables use one row per declared record and retain explicit identifiers. Neither repeated motif names nor repeated database hits establish independent observations.

The search ledger records all returned bibliographic items; items outside the selected extraction have not all been systematically screened. TRANSFAC licensing, the unavailable legacy TFBSdb endpoint, individual old-paper coordinates and additional database exports remain open. No exhaustive completeness claim is made. The three future views preserve original DNA positions, full-circle spacing and a separately labelled M9 reduction.

Human site harmonisation — 19 October 2026

Read human-site-project.json (or human-site-project-fields.csv) for scope and open recovery tasks. normalized-observations combines every imported record under one column schema, while the original tables remain intact. Evidence categories are multi-valued: laboratory tested includes positive, negative and mixed results; functional evidence includes regions and element-based reporters. The category counts are evidence records, not unique confirmed sites.

reference-intervals groups only identical genomic intervals; every observation remains linked to its source. Shared coordinates do not establish the same factor, experiment, or biological function. literature-overlaps contains spatial links from the mapped human literature records to all imported database records. It is not an all-pairs network or independent corroboration. evidence-relationships records known reuse and distinct cooperating elements.

US-1 and US-2 each match once in the saved human OXTR window on the reverse strand. These are computed reference placements of published sequences, not new experiments or a claim of genome-wide uniqueness. mapped-literature.bed contains two sequence elements and one broad annotation; use normalized-literature to distinguish them.

The original human 15-site report remains an aggregate of 3 AP1, 9 C/EBP and 3 NF-kappaB sites. Its individual sequences and historical coordinate anchors remain unresolved. No total of unique confirmed human OXTR sites or DNA-to-Atlas comparison is asserted.

Coordinate experiment — 20 October 2026
experiment-v1/ adds a fixed DNA-line, full-position-circle and M9-circle representation, with all four M9 P24 reference sequences. Sources remain separate: 2 published elements, 255 ENCODE regions and 9,164 JASPAR score-400 matches. Read experiment-v1/methods.md for the origin, direction, centre-base rule and clipping. All mappings, verification reports and scripts are downloadable. CSV includes full observation tables and metadata field tables. Biological association remains untested.

Binding outcomes — 20 October 2026

Tested is not an outcome. binding-outcomes.json and the lossless typed binding-outcomes-fields.csv define the source-linked interpretation, reviewed study results, regional evidence and predictions. mapped-binding-outcomes.json/csv provide one outcome row for each of the 9,421 plotted records, joined by observation_id. The same interpretation is embedded as binding_interpretation in catalogue.json/catalogue-fields.csv. Browser selection exports include binding_evidence per row.

US-1 and US-2 report positive binding results under their assay conditions; their expression effects and abstract-only review depth remain explicit. ENCODE clusters are regional association evidence, not exact motif positions or proof of direct binding. JASPAR records are predictions. No negative binding controls have been curated for this map. Missing or unclassified outcomes do not mean binding failed. Biological support and statistical significance of an Atlas match remain separate.

Rebuild the interpretation layer with source/annotate_oxtr_binding_outcomes.py after preparing the experiment, then run source/package_oxtr_catalogue.py. Raw observations and coordinate mappings remain unchanged.

Study additions — 21 October 2026

BrainTF: 126 screened ChIP-seq peak observations, 43 protein targets and 13 brain contexts. GRCh38, 0-based starts and exclusive ends. Records can overlap and share donors. These are regional occupancy measurements, not 126 distinct binding sites. Read review.json or the equivalent review-fields.csv for exact inclusion criteria and limitations. braintf-locus-review retains all 390 reviewed locus observations and the selection flag. Histone marks and rejected observations do not enter the study browser.

The 2025 enhancer study contributes functional reporter evidence. Its reported candidate intervals and primer mappings remain separate in enhancer-regions. No precise TF binding position is inferred.

The DNA window, ENCODE and JASPAR imports, and the existing two-element Studies geometry remain unchanged. All study data is included in the complete JSON/CSV archives. Source URLs, row numbers and SHA-256 hashes identify the reviewed public files. Raw sequencing reads and controlled individual-level data are not included.

Rebuild with add_oxtr_brain_studies.py, normalize_oxtr_sites.py, annotate_oxtr_binding_outcomes.py and package_oxtr_catalogue.py, in that order. The archived review inputs are in research/informatics/oxtr-study-review-20261021.
