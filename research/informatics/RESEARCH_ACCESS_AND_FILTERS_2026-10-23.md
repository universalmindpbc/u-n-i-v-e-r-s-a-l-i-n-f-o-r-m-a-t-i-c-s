# Universal Research: access, retrieval and filters

23 October 2026 · first access pass and staging update

## Agreed infrastructure

Use a dedicated Postgres database with pgvector and native full-text search, plus S3-compatible document storage. These serve different purposes: structured records and search versus article files and supplements. OpenAI File Search and Pinecone are alternative retrieval implementations, not additional required services. No database, bucket, vector index or paid model service was provisioned in this task.

Universal One retrieves approved records and permitted article passages at query time. Local article ingestion does not train or fine-tune a model. Preserve the API-first design supporting Universal One, external MCP clients, open exports, and later reviewed collaboration and Synapse packages.

## First article-access pass

- 758 public records screened from the live all-research export.
- 474 matched in Europe PMC using identifiers or high-confidence title/year matches.
- 270 records have free full-text locations reported by Europe PMC. These include publisher and repository destinations; publisher-page readability is not individually confirmed for every destination.
- 160 matched records are flagged as Europe PMC Open Access.
- 115 complete article XML files were retained after checking their own CC BY licence statements and archive identifiers. Each retains attribution, licence text, SHA-256 and extracted source sections/paragraphs: 7,537 paragraph/table entries in total.
- One CC BY-labelled candidate has a contradictory noncommercial clause in its article statement (PMC2855827). It remains on hold. Other NC, ND or unspecified terms remain for review.
- 202 matched records have no free full-text location in this index. This is not proof of a paywall.
- 284 records lacked a confirmed Europe PMC match; two additional identifier matches need short-title/identity review. The complete original corpus remains intact.
- Crossref was also checked for the 137 unresolved records with DOIs. Sixteen have deposited Creative Commons licence metadata; these need publisher/version confirmation before article ingestion. Crossref returned 115 with no open licence metadata, two identity-review candidates and four unresolved requests.
- No full article files were published to the website, sent to an AI provider, embedded, or used for model training.

The reports `access-audit.json`, `crossref-access.json` and `fulltext-manifest.json` record the identifiers, evidence links, checked date, exceptions and retrieval result. The scripts in `tools/` make the pass reproducible. Index-reported access is distinct from a downloaded and licence-checked article.

Local article storage: `/Users/benkorbel/Documents/m o d u l 8 i n f i n i t y/i n f o r m a t i c s/research/open-fulltext-20261023/`.

## Multi-select research search

Checkbox dropdowns support multiple topics, conditions/symptoms, journals, research categories, physiological outcomes, molecular targets, study designs, dates, trust scores and Jadad assessments. Sort order is a single selection.

Within a filter, selected options use OR. Across filters, the groups use AND. Repeated URL parameters preserve selections, and removable chips expose each choice. JSON export contains the filtered records and condition evidence.

Condition and symptom labels initially cover 211 records. The evidence is explicit wording in publication titles or transcribed study sample/measurement fields. A tag does not assert a participant diagnosis, treatment effectiveness or full-text appraisal. Papers without a tag remain searchable and are not presumed irrelevant.

Mindfulness-Based Stress Reduction as an intervention name alone is not a stress outcome. Oxidative or cellular stress alone is not a mental-health symptom label. PTSD, trauma-related symptoms, psychosis and schizophrenia retain separate selectable concepts. No guessed gene, condition, citation or trust score is added.

Next annotation pass: use the available full texts to distinguish diagnosed population, symptom measurement, risk factor, mechanistic model and incidental mention. Retain the exact supporting passage and study population. Review proposed additions before merging them into the public classification. Link multiple reports from the same trial without removing publication records.

## Approved wording and website scope

Metadata uses verbatim About sentences: “Bioinformatics platform for mental health. Investigating potential relationships between the Fibonacci sequence and OXTR gene expression.” The same description is present in HTML, Open Graph and Twitter metadata.

About and its nested initial-focus dialog now use an 18% black backdrop with no blur. The existing readable panels and video remain in place. Search, narrative texts, quotes, grades and citation counts retain their approved content.

This update is staging only. The previous production release remains unchanged pending Ben's approval.

## Sources

- https://europepmc.org/RestfulWebService
- https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
- https://www.crossref.org/documentation/retrieve-metadata/text-and-data-mining/

Use current supported APIs. The former PMC OA Web Service was retired in August 2026; this audit uses Europe PMC's documented REST API instead.
