# Jadad appraisal rollout

Requested by Ben, 23 October 2026. Status: planned; no numeric scores assigned in this release.

## Scope

Screen every indexed publication in the 1,000+ record collection for eligibility. Score eligible randomized trial reports manually from their full text and relevant supplementary material. Reviews, observational studies, animal/cell experiments, protocols and non-research references receive `not-applicable` with a reason after review. Never assign zero to an unassessed or inaccessible paper.

Codex can perform the initial paper-by-paper appraisal. Keep each decision transparent and available for Ben to review. Do not describe a single AI appraisal as independent dual review. A later second reviewer can resolve uncertain decisions and check a sample or all scores.

## Original 0–5 Jadad scale

- Randomization: one point for reporting random assignment; an additional point for an appropriate sequence-generation method; deduct the first point if the reported method is inappropriate.
- Double blinding: one point for reporting double blinding; an additional point for an appropriate method; deduct the first point if the reported method is inappropriate.
- Withdrawals/dropouts: one point for the numbers and reasons in each group, or an explicit report of none.

Record the individual components and their supporting page/section, source link, short evidence excerpt, rationale, reviewer identity, date and scale version. Preserve the difference between information absent from an accessible report and unavailable full text. The latter stays unassessed with an access reason.

Do not infer Jadad from the existing crosses, clinical A–C grade, journal prestige, abstract, citation count, or statistical significance. Do not silently substitute a modified Jadad scale. Assessor masking alone is not double blinding. Many mindfulness and psychotherapy trials cannot blind participants; explain that constraint alongside their original-scale score. Jadad does not cover every source of bias; allocation concealment, selective reporting and other domains need separate appraisal.

## Record structure

`jadad`: `status` (unassessed, assessed, not-applicable), `score` (integer 0–5 only when assessed; otherwise null), `scaleVersion`, `components` (randomization, blinding, withdrawals), `evidence` (source, locator, excerpt, rationale for each component), `assessedBy`, `assessedAt`, `reviewStatus`, `applicabilityReason`.

Keep report-level appraisals linked to a trial identifier. Multiple publications from one trial remain separate papers but never count as independent replication. Preserve earlier assessments in a version history when corrections are made.

The first UI release offers All, Not yet assessed, Not applicable and exact scores 0–5. No placeholder scores or automatic applicability decisions. Search, deep links, reset and exported records carry the same status. A zero means an assessment awarded no points.

## Plain-language guide

Jadad is a five-point check of how a randomized trial reports assignment to groups, double blinding and people leaving the study. Zero means no points awarded; one through four mean that number of points; five means the maximum on these criteria. Each detailed assessment explains the exact components behind its total. A higher score does not establish that a treatment works or that a paper has no bias. The website guide is distinct from topic trust scores and the A–C assessment.

## Delivery sequence

1. Resolve publication identities and shared-trial links; inventory full-text availability.
2. Prioritize mindfulness/gene-expression RCTs and the approved key clinical trials.
3. Appraise papers individually using the original criteria; retain locators and reasons.
4. Review uncertain decisions with Ben; add a second review when available.
5. Publish a scoped batch to staging with coverage totals and an appraisal date; Ben proofs before production.
6. Continue across Sound, Vision, Language and General without duplicating cross-listed studies.

Sources: [Original Jadad paper, 1996](https://pubmed.ncbi.nlm.nih.gov/8721797/); [AHRQ scoring form and guidance](https://www.ncbi.nlm.nih.gov/books/NBK56923/).
