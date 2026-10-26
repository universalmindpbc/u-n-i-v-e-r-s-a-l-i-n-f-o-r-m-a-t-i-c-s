# Human OXTR site recovery and harmonisation

Approved 19 October 2026. The first pass combines all existing source records under one schema and exposes human laboratory and functional evidence without relying on narrow legacy labels.

Canonical reproducible builder: `../pilot/normalize_oxtr_sites.py`. Public scope, priorities and counts: `../../../Informatics/pilot-v1/oxtr-tfbs-v1/human-site-project.json`.

The original 15-site study is human (PMID 16569740; DOI 10.1210/jc.2005-2649). Its publisher abstract confirms 3 AP1, 9 C/EBP and 3 NF-kappaB sites and laboratory assays. The publisher lists the full paper for purchase; a direct PDF request returned HTTP 403 during this pass. No paid access, author contact, or claim of recovering the individual sites was made. Routes are library access/document delivery, publisher purchase, or an author-provided copy and original sequence tables. The individual coordinates recovered for that set remain zero.

Published US-1 (PMID 10221779) and US-2 (PMID 10527846) sequences each match once, in reverse-complement orientation, inside the saved 39,248-base GRCh38 window. This is local reference placement, not genome-wide uniqueness or new experimental confirmation. The original US-1 report describes a 20-base region but supplies a 17-base sequence; only the supplied sequence is mapped.

The 2006 and 2011 shared promoter regions are linked using reported numbering, without guessing a genomic transcription-start anchor. US-2 reuse and the OREG1492306 annotation citation are linked. GABP and CRE-like elements within one broad region remain separate observations.

All imported records remain intact. Exact genomic intervals have stable identifiers derived from species/build/chromosome/start/end. The 5,114 spatial overlap pairs connect three mapped human literature records (two sequence elements plus one annotation) to all imported database records; they are not independent confirmations. No cross-species merge, motif scan, genome liftover, M9 transformation or enrichment test is performed.

Additional source discovery: GSE197160 contains human myometrial HIF1A data. GEO lists bigWig signal tracks and SRA reads. No exact peak coordinates or additional sites are inferred from those descriptions. The legacy sequences, GABP/CRE-like extents, ERE-like extents and human-source expansion remain active priorities.

Notion project: https://www.notion.so/3fd7b9c99c56815a8b84e6839dc1d3ed

Primary sources: https://academic.oup.com/jcem/article/91/6/2317/2843585 ; https://pubmed.ncbi.nlm.nih.gov/21734268/ ; https://pubmed.ncbi.nlm.nih.gov/10221779/ ; https://pubmed.ncbi.nlm.nih.gov/10527846/ ; https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE197160

## Next visual phase — 19 October 2026

Ben asked to record Python illustrations of the existing binding-site data in the original DNA line, full-position circle and Modulus 9 circle, compared with all four M9 P24 Atlas sequences. The planned website feature uses Wolfram and additional Biopython calculations. See the [next-session brief and task list](next-steps.md). Resume with Ben; the comparison feature is not implemented by this note.
