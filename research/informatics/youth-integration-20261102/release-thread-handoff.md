# Read-only release handoff

Request: youth-informatics-destination-20261102. Delivered once to “Add Music Atlas measurement architecture”, thread `01a19041-e26c-7b51-ab88-39253a76a2cf`, local host. Reply received 2 November 2026. This records operational facts and a proposed synchronisation sequence; it does not authorise publication or deployment.

The authoritative deployment source remains the **integrated Universal Mind repository**. The dedicated Informatics repository is the public source/research record; its [AGENTS.md](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-i-n-f-o-r-m-a-t-i-c-s/blob/main/AGENTS.md) and [architecture document](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-i-n-f-o-r-m-a-t-i-c-s/blob/main/docs/ARCHITECTURE.md) explicitly preserve that boundary.

Remote references verified during this read-only check:

| Repository / reference | Current commit |
|---|---|
| Integrated `main` — production | `57747030461016e4fc2c29bd4534df483118857d` |
| Integrated `staging` | `fcb07c7cd1f7cdf284ac4ebbe29871fc08daf214` |
| Annotated production tag | `production-2026-11-02-1509`, resolving to `577470304…` |
| Dedicated Informatics `main` | `7d89224d2caca17debcd8170a1873a488d5af82a` |
| Dedicated Informatics `staging` | No remote branch exists |

The canonical integrated repository is [u-n-i-v-e-r-s-a-l-m-i-n-d](https://github.com/universalmindpbc/u-n-i-v-e-r-s-a-l-m-i-n-d). Its configured `origin` still uses `universal-mind-website.git`; GitHub redirects it successfully. Preserve that configuration.

The local mapping is:

- Integrated shared checkout: [universal light](</Users/benkorbel/Documents/m o d u l 8 i n f i n i t y/universal light>). Its checked-out local `staging` remains at `5e3e07d21…`, with six tracked deletions and 27 untracked entries. **Do not reset or deploy this checkout.**
- Youth workspace: [youth-research-20261102](</Users/benkorbel/Documents/Codex/2026-11-02/task/youth-research-20261102>). Verified local `staging` at `fcb07c7cd…`, with its own `.git` directory and 140 changed entries.
- The registered [Informatics project folder](</Users/benkorbel/Documents/m o d u l 8 i n f i n i t y/i n f o r m a t i c s>) is **not a Git repository**. I have not confirmed an active local checkout of the dedicated public repository. Do not treat that project folder as its checkout.

The staging destination is [Universal Research](https://staging.universalmind.tv/informatics/research/), within [Informatics](https://staging.universalmind.tv/informatics/). The existing immutable review resides at `/informatics/review/youth-20261102/`. Local `Informatics/` publishes to lowercase `/informatics/`.

Use the integrated scripts:

- `./deploy-panthur-staging.sh informatics` for the committed Informatics candidate.
- `./deploy-panthur-staging.sh ai-wiki` only when its scoped navigation/wiki outputs change.
- Retrieval/service changes also require the existing staging Worker configuration: `cloudflare/wrangler.informatics-staging.toml`, Worker `informatics-universal-one-staging`.
- Production remains a separate approval and uses `deploy-panthurlive.sh`.

Credential mechanisms, identifiers only:

| Environment | Existing mechanism |
|---|---|
| Staging FTP | Keychain service `universalmind-staging-ftp`; optional credential file `/private/tmp/um-staging-ftp-pass.txt` |
| Production FTP | Keychain service `universalmind-production-ftp`; optional credential file `/private/tmp/um-public-html-ftp-pass.txt` |
| Workers | Existing Wrangler authentication and environment-specific Worker secrets |

This thread’s release is complete and its temporary checkout was removed. No deployment processes were visible during this check. The shared integrated repository has a zero-byte `.git/index.lock`, dated **30 October, 12:40 Sydney time**; ownership is unconfirmed. Leave it untouched. The youth workspace had no index lock.

For synchronisation, I recommend this sequence:

1. Finalize the scoped integrated candidate from `fcb07c7cd…`, preserving the completed release and immutable review.
2. In a separate branch based on dedicated Informatics `main`, synchronise an explicit list of affected paths and required dependencies from that candidate. Import paths already match. Preserve dedicated repository governance, licensing and historical provenance.
3. Add a versioned synchronisation receipt containing both repository commit IDs, the integrated baseline, file hashes and generated-asset references. Keep the original `a18a06fa…` import record as history.
4. Publish the approved integrated candidate through Panthur staging, then record the verified staging commit. Wait until every upload finishes before requesting new asset URLs; premature probes caused the cache repair in this release.
5. Resolve dedicated-publication approval and production approval separately in the parent task.

No files, remotes, branches, deployments or credentials were changed for this handoff.
