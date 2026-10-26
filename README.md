# Universal Informatics

A bioinformatics research portal exploring mathematical sequences, molecular systems, computational models and evidence-linked research for mental health.

**Founder and maintainer:** Ben Korbel<br>
**Copyright holder:** Universal Mind TV, Public Benefit Corporation<br>
**Website:** [universalmind.tv/informatics/](https://universalmind.tv/informatics/)<br>
**Proofing environment:** [staging.universalmind.tv/informatics/](https://staging.universalmind.tv/informatics/)

Universal Informatics combines sequence and geometry exploration, molecular and quantum views, research libraries, machine-readable exports and assistant-supported research workflows. This dedicated repository makes the project independently reviewable on GitHub. Its hosted address and established deployment remain unchanged.

## Scientific scope

Potential relationships between Atlas sequences and biological function are research hypotheses. The initial bioinformatics and in silico investigation is specific to OXTR. Wider studies provide context, not proof of universal biological applicability. Mathematical results, model calculations, literature evidence and biological validation remain distinct. Public terminology uses **Phi Ratio**.

## Repository map

| Location | Responsibility |
| --- | --- |
| [`Informatics/`](Informatics/) | Portal entry point, Atlas interfaces, APIs, evidence views and browser-side code. |
| [`Informatics/molecular/`](Informatics/molecular/) | Molecular visualisation and supporting components. |
| [`Informatics/quantum/`](Informatics/quantum/) | Computational research interface and quantum-related workflows. |
| [`research/informatics/`](research/informatics/) | Methods, service implementations, source records and historical verification evidence. |
| [`cloudflare/`](cloudflare/) | Informatics service modules and existing environment configuration. |
| [`tools/`](tools/) | Research preparation, export and verification utilities. |
| [`docs/SOURCE_MANIFEST.json`](docs/SOURCE_MANIFEST.json) | Exact upstream commit, imported file hashes and provenance. |
| [`docs/GENERATED_ASSETS.json`](docs/GENERATED_ASSETS.json) | Pinned locations and hashes for large or generated assets retained upstream. |

## Source provenance and generated assets

The imported files come from Universal Mind commit `a18a06fa8793d50e51caf2b24bc58cc4babe9de4`. Their existing paths are preserved so the code, research tools and historical links retain their context. The manifest records original Git blob identifiers and current SHA-256 hashes. Documented import adaptations cover private-data exclusion rules and removal of expired third-party signed URL parameters from six publication metadata files. Earlier history remains available in the [upstream repository](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d/tree/a18a06fa8793d50e51caf2b24bc58cc4babe9de4).

Several gigabytes of generated sequence views, CSV records, archive payloads and large assets remain at that pinned upstream version. They are indexed rather than duplicated in the initial source checkout. After obtaining source-use permission, an authorised researcher can inspect the asset plan with `python3 tools/restore-source-assets.py --list`, restore a selected `--prefix`, or explicitly request `--all`. Downloads are verified against the recorded Git blob hash. The standard clone therefore contains the research and application source; complete data-heavy reproduction also requires the listed assets and documented shared services.

## Shared services and deployment

The current portal uses shared I AM assets and session/service infrastructure, including resources under `/vogue/iam/`. See [ARCHITECTURE.md](docs/ARCHITECTURE.md). This repository does not provision credentials or replace those services.

The existing integrated Universal Mind workspace remains the deployment source until an explicitly reviewed migration changes that responsibility. Informatics changes need a recorded synchronisation with that source before the normal Panthur staging and production process. No automatic deployment is attached to this repository. The new GitHub project is not a new website host.

## Academic Context and Doctoral Foundations

The sequence code and Atlas views support inspection of mathematical constructions. The [`oxtr-human-site-project`](research/informatics/oxtr-human-site-project/), TFBS evidence modules, molecular components and [`astra-service`](research/informatics/astra-service/) expose different research methods and evidence types. Model-assisted reviews and computational outputs are not experimental confirmation.

Use the [cross-project doctoral review guide](docs/PHD_REVIEW.md) for the research map, evidence distinctions, reproduction requirements and relationship to I AM and Re-Model. Use [SOURCE_MANIFEST.json](docs/SOURCE_MANIFEST.json) when identifying the exact imported version.

## Verification and access

Start with [VERIFICATION.md](docs/VERIFICATION.md). Historical reports are preserved as records of their stated versions and checks. Paid services, live model requests and cloud computation require separate credentials and explicit operational authorisation; they are not necessary merely to inspect this repository.

The source is available under a [permission-required notice](LICENSE). Academic researchers require individual written permission before running or modifying the software. Normal use of the hosted website is separate. Approved organisations below USD 1 million annual revenue receive a fee-free licence within the agreed scope; organisations at or above the threshold negotiate a nominal fee in good faith.

See [LICENSING.md](LICENSING.md), [CITATION.cff](CITATION.cff), [MAINTAINERS.md](MAINTAINERS.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Related projects: [Universal Mind I AM](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d) · [Re-Model Intelligence](https://github.com/universalmindpbc/r-e-m-o-d-e-l-i-n-t-e-l-i-g-e-n-c-e).
