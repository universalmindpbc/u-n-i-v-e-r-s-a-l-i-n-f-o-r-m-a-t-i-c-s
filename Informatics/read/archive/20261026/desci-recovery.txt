# DeSci archive manifest recovery — 26 October 2026

The archive draft is `n-_wkF_5LYvm_uoZ_hdFE7wEybmRoXFi2J1Qa4F0USA`. Its original document ID is `43oxboqgNaSWgHDZVsM3vQ3ECuqi`. Draft creation returned a populated document with one root component. Subsequent owner-authorised file uploads saved their bytes but returned HTTP 500 during metadata processing.

All 33 saved files have independently computed CIDs that match the provider tree. The owner-authorised node response retains the same document ID as the creation receipt. The document endpoint returns a null document. A documented action that reapplies the existing title returns HTTP 400 with `Document not found`. Preparation returns HTTP 400 with `Cannot read properties of null (reading 'components')`.

This establishes a missing document lookup in DeSci's editor service. It does not establish why that document is missing. Provider logs and internal storage access are required to distinguish persistence failure, mapping failure and service configuration issues. No internal provider credentials or infrastructure are available in this task.

Recovery completed through the existing owner-authorised `POST /v1/nodes/updateDraft` endpoint, which accepts a supplied manifest when its live document lookup returns null. The recovery preserves the draft, its title, rights metadata and every stored file. No deletion, public publication, signing, account permission change or duplicate data upload occurred.

The directory was reconstructed using DeSci's published UnixFS directory algorithm: each link binds the existing file name, size and CID. DeSci's CID metadata endpoint confirmed the resulting directory already exists with 33 links and 2,535 block bytes. The saved replacement archive manifest was retrieved through the node API and its CID independently recomputed from its exact canonical JSON bytes.

- Recovered manifest CID: `bafkreicldcdkxjvsdqwbntljqhfacjsmhbsutaz7qtfcbvvgusuw5j4s7m`
- Directory CID: `bafybeiegycaritcwqqx6rfeoazvv3cdlh7klfaw5gqftdtik6pdxz5egbe`
- Original timestamped package manifest SHA-256: `df177d308cf36853b5c23e5d47356bc0d7976a1bb12792a0771020477adcbd5a`
- All original file CIDs and sizes remain unchanged; the draft remains private.

The recovered archive is complete. DeSci's editable document service still needs provider-side restoration. No public API for resetting an existing document ID was found in the published routes. A repair request was submitted through DeSci’s official contact form on 26 October 2026 with Ben’s explicit authorisation. The form confirmed receipt. A fresh empty diagnostic draft reproduced the same missing-document error before being removed; this extends the observed fault beyond the existing archive. Editor-service restoration remains pending DeSci’s response.

Provider repair request: restore or recreate the editable document for this existing draft, reattach its document ID, and verify document retrieval and manifest preparation while preserving the recorded archive CID and all 33 file CIDs.

Evidence and primary implementation references:

- [Draft creation](https://github.com/desci-labs/nodes/blob/develop/desci-server/src/controllers/nodes/draftCreate.ts)
- [Document retrieval and actions](https://github.com/desci-labs/nodes/blob/develop/desci-server/src/controllers/nodes/documents.ts)
- [Document service](https://github.com/desci-labs/nodes/blob/develop/desci-server/src/services/repoService.ts)
- [Supported draft-update fallback](https://github.com/desci-labs/nodes/blob/develop/desci-server/src/controllers/nodes/draftUpdate.ts)
- [Directory construction](https://github.com/desci-labs/nodes/blob/develop/desci-server/src/utils/draftTreeUtils.ts)

Repository source explains the observed API behaviour; it does not establish the exact revision deployed by DeSci. No changes were made to DeSci's source repository or service infrastructure.
