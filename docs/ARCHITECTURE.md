# Architecture and repository boundary

The dedicated Informatics repository is an auditable source snapshot of the integrated Universal Mind implementation at commit `a18a06fa8793d50e51caf2b24bc58cc4babe9de4`. Import paths are retained. No files were removed from I AM and no hosted URL was changed.

## Components

- `Informatics/`: application, browser code, evidence views and PHP adapters.
- `research/informatics/`: computational and research implementations, provenance and historical verification records.
- `cloudflare/`: Informatics service modules and environment configuration.
- `tools/`: relevant research generation, export and technical verification utilities.

## Shared runtime dependencies

The current application references I AM assets under `/vogue/iam/`, including the favicon, brand image and realtime assistant stylesheet. `Informatics/api/astra-session.php` uses the existing I AM runtime-private session location. Other service bindings, provider integrations and environment configuration remain deployment-specific. The shared implementation remains inspectable at the pinned [I AM source](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d/tree/a18a06fa8793d50e51caf2b24bc58cc4babe9de4/i%20%20a%20m).

These are explicit system dependencies. A clone alone does not provision them, provider credentials or research inputs. The portal entry point is `Informatics/index.html`; the public hosted path remains `/informatics/` with the existing compatibility routing.

## Source of deployment

The existing integrated Universal Mind workspace remains the deployment source. Its `deploy-panthur-staging.sh` and `deploy-panthurlive.sh` workflows retain responsibility for publishing. Do not deploy a second copy from this presentation repository or attach an automatic hosting pipeline.

For a future code change, record the source commit, synchronise the affected paths with the integrated deployment source, run relevant checks, publish through the established staging workflow, obtain proofing approval, and record any separately authorised production release. Keep both repository records linked to the exact version. A change of deployment ownership needs a separate explicit decision.

## Data and provenance

`SOURCE_MANIFEST.json` records the imported bytes. `GENERATED_ASSETS.json` records generated or large upstream assets by path, size and Git blob hash. The retrieval tool restores only files from the pinned source revision and verifies each completed download. It never reads deployment credentials or activates research services.
