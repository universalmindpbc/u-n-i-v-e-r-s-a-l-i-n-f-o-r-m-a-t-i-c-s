Universal Informatics — complete CSV data, M1–M1001

No JSON file is needed to use this archive.

m1-m1001.csv: one row per modulus, including the native/reduced sequences, geometry counts, verification status and direct production PNG/SVG URLs. The record_csv_file column locates the complete record inside this archive. A blank Epsilon URL means no qualifying or reference native Epsilon image; it is not an omitted upload. The epsilon_status and detailed record explain the rule. Reduced M9 images exist for all four sequences in each of the 111 M9-family records.

csv-records/m1.csv through m1001.csv: every field and value in the corresponding Atlas record, including full sequences, formation catalogues, calculation intervals, Epsilon assessments, native/M9 comparisons, verification, probability where present, research assessments and publication metadata. These are complete records, not just the overview table. Redundant formation interiors retain their documented endpoint-based reconstruction rule and full Alpha values.

m1-m1001-images.csv: all 8,938 distinct diagram URLs, with record identifiers, source fields, formats and file hashes. PNG and SVG files remain online; image pixels are not embedded in this archive.

research-publications.csv: all 14 studies, including the four HERO references, citations and appraisal fields. csv-shared/ preserves the full shared registers, audits, manifests and existing probability evidence as CSV.

Typed record format: field is a slash-separated path; type distinguishes text, integer, number, boolean, null, object and array. Null is distinct from empty text. Primitive lists use semicolon-separated CSV values within a cell (list:integer, list:text, etc.). Nested arrays and objects use indexed child paths. Empty containers are retained. ~1 represents a slash in a field name and ~0 represents a tilde. resolved_url provides a complete production URL for referenced Atlas resources. Original field values are preserved.

source/csv_records.py reads any typed record using only Python's standard CSV library. Example: from csv_records import read_record; record = read_record('csv-records/m999.csv'). The remaining Python sources document the Atlas calculations and drawing methods; rendering uses Pillow and an available font. No JSON parser is needed to load the CSV record. Website-specific build scripts retain their repository layout requirements.

The JSON link in the overview is optional. All computational fields are also in these CSV records. WSS remains awaiting assessment; arithmetic/geometry evidence does not assign biological significance.

OXTR transcriptional-regulation supplement — 19 October 2026

oxtr-tfbs-v1/ contains a separate source-linked catalogue, literature records, reference DNA, JASPAR motif templates and existing predictions, ENCODE occupancy clusters and experimental provenance, and ORegAnno annotations. Read its README.txt and catalogue metadata for the bounded reference window and interpretation. CSV users have complete tables and typed metadata without needing JSON. The supplement is shared across the Atlas; no modulus-specific biological comparison has been performed. The original 14-study register is unchanged.

Coordinate experiment — 20 October 2026

oxtr-tfbs-v1/experiment-v1/ contains fixed projections for 128 study observations (2 sequence elements and 126 BrainTF occupancy regions), 255 ENCODE regions and 9,164 JASPAR predictions (score >=400), compared visually with all four M9 P24 Atlas sequences. It includes original intervals, source provenance, mappings, methods, Python/Biopython code and Wolfram checks. JSON and CSV exports carry the same experiment information. This is visual exploration with computational verification; biological association is untested.
