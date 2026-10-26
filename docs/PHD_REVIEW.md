# Academic Context and Doctoral Foundations

## Purpose of this guide

This guide connects three related software projects maintained by Ben Korbel. Universal Mind I AM and Universal Informatics are licensed by Universal Mind TV, Public Benefit Corporation. Re-Model Intelligence is licensed by Universal Mind TV, Public Benefit Corporation and Re-Model Universe, Public Benefit Corporation. It provides a route through the existing implementation and research foundations for doctoral review. It does not assert university affiliation, doctoral admission, ethics approval, clinical efficacy, patent status or independent validation.

The existing software and associated background intellectual property are distinct from proposed doctoral investigations and any future findings. Ownership, institutional access and research permissions are matters for the relevant written agreements.

| Project | Role in the research programme | Start with |
| --- | --- | --- |
| [Universal Mind I AM](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d) | Adaptive audiovisual experiences, check-in interfaces, inference policies and participant-facing research infrastructure. | [`i  a m/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d/tree/main/i%20%20a%20m), [`research/privacy-lab/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d/tree/main/research/privacy-lab), [`database/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d/tree/main/database). |
| [Universal Informatics](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-i-n-f-o-r-m-a-t-i-c-s) | Mathematical, bioinformatic, molecular and computational exploration; evidence-linked research and provenance. | [`Informatics/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-i-n-f-o-r-m-a-t-i-c-s/tree/main/Informatics), [`research/informatics/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-i-n-f-o-r-m-a-t-i-c-s/tree/main/research/informatics), [`cloudflare/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-i-n-f-o-r-m-a-t-i-c-s/tree/main/cloudflare). |
| [Re-Model Intelligence](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e) | Social and cultural research workflows, observed evidence, research review, network analysis and strategic reporting. | [`frontend/`](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e/tree/main/frontend), [`rmu_backend/`](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e/tree/main/rmu_backend), [`migrations/`](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e/tree/main/migrations). |

## Suggested review sequence

1. Read each project README to establish its scope, current implementation and service dependencies.
2. Follow the project-specific academic code map into methods, evidence handling and verification.
3. Trace a claim or output back to its input, source record, transformation and review status.
4. Inspect the tests and provenance records alongside the code. Historical verification records describe a particular version and test; they are not proof of every current capability.
5. Separate a reproducible software operation from an empirical or scientific conclusion. Record the additional data, study design and validation needed for each proposed investigation.

## I AM: adaptive experience and research infrastructure

- [`i  a m/homepage-checkin.js`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d/blob/main/i%20%20a%20m/homepage-checkin.js) and the portal source expose the check-in and experience flow.
- [`i  a m/selector_core.py`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d/blob/main/i%20%20a%20m/selector_core.py), [`adaptive_inference_matrix.py`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d/blob/main/i%20%20a%20m/adaptive_inference_matrix.py) and the versioned inference policies provide inspectable selection logic.
- [`i  a m/adaptive-sound/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d/tree/main/i%20%20a%20m/adaptive-sound) and [`i  a m/research/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d/tree/main/i%20%20a%20m/research) contain experimental sound and research components.
- [`research/privacy-lab/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d/tree/main/research/privacy-lab) and [`database/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d/tree/main/database) document privacy and data infrastructure work.

These modules make the methods and implementation available for review. They do not, by their existence, demonstrate therapeutic effectiveness, diagnostic accuracy or a validated inference of emotion, physiology or gene expression.

## Informatics: hypotheses, computation and evidence

The proposed relationship between Atlas sequences and biological function is a research hypothesis. The initial bioinformatics and in silico investigation is specific to OXTR. Wider literature supplies context; it does not establish universal biological applicability. Public copy uses the term **Phi Ratio**.

- [`Informatics/app.js`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-i-n-f-o-r-m-a-t-i-c-s/blob/main/Informatics/app.js), the sequence records and Atlas views expose mathematical constructions and visualisations.
- [`Informatics/molecular/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-i-n-f-o-r-m-a-t-i-c-s/tree/main/Informatics/molecular) and [`Informatics/quantum/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-i-n-f-o-r-m-a-t-i-c-s/tree/main/Informatics/quantum) contain distinct molecular and computational experiences.
- [`research/informatics/oxtr-human-site-project/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-i-n-f-o-r-m-a-t-i-c-s/tree/main/research/informatics/oxtr-human-site-project) and the TFBS evidence modules trace regulation-related evidence and its limits.
- [`research/informatics/astra-service/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-i-n-f-o-r-m-a-t-i-c-s/tree/main/research/informatics/astra-service) contains research workflow, computation and review components.
- [`tools/`](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-i-n-f-o-r-m-a-t-i-c-s/tree/main/tools), the source manifest and the generated-asset manifest support technical verification and recovery of the pinned source version.

Mathematical identities, visual analogies, database annotations, docking output, quantum calculations and model-assisted reviews represent different forms of evidence. Treat them separately. Receptor structure confidence is not a direct measurement of folding state; generated reviews are not experimental confirmation.

## Re-Model: observed evidence and human review

- [`frontend/app/network/`](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e/tree/main/frontend/app/network) defines the observed network, expanded discovery and reviewed recommendation workflow.
- [`frontend/app/lab/`](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e/tree/main/frontend/app/lab), [`frontend/docs/`](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e/tree/main/frontend/docs) and [`frontend/app/report/`](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e/tree/main/frontend/app/report) connect research exploration, methodology and reporting.
- [`rmu_backend/`](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e/tree/main/rmu_backend) and [`migrations/`](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e/tree/main/migrations) implement collection, records, model jobs and evidence storage.
- [`tests/`](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e/tree/main/tests) and [`frontend/tests/`](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e/tree/main/frontend/tests) expose technical checks. Tests that access live databases, APIs or paid providers are separate from offline checks.

Observed source material, inferred labels, synthetic demonstrations, strategic scenarios and approved recommendations need explicit distinctions. Model output is not human approval. Campaign context, provenance and review state are part of the analytical record.

## Reproduction, access and citation

Academic researchers require individual written permission before running or modifying the software. Normal access to the hosted portals is separate from permission to execute their source. Provider credentials, private participant records, raw retained social data and operational secrets are not supplied with the public source.

For an approved reproduction, record the repository commit, software environment, input version, relevant service configuration, checks performed and limits of the reproduction. Cite each project through its CITATION.cff. Request research access from Ben Korbel at ben@internetofhappiness.org.
