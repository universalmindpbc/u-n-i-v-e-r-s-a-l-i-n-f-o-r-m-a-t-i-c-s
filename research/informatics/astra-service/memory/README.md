# Reviewed research memory

The experiment registry remains the public source of truth. JSON and CSV revisions are immutable. The research database holds exact copies and hashes; retrieval uses short, embedded evidence records alongside the existing literature library.

For every approved experiment publication:

1. Register the immutable JSON/CSV revision and rebuild the complete Informatics exports.
2. Run `import.mjs` with an owner-only file containing the existing importer database connection. Test schema changes on a Neon branch before the existing database.
3. Verify archive hashes and the retrieved experiment revision. A conflicting hash requires a new revision.
4. Publish the normal staged website release. Private task drafts never enter shared memory automatically.

`intent.json` records Ben's mission. `research-directions.json` records hypotheses separately from findings. Changes need a new revision. Retrieval supplies provenance and bounded excerpts; it does not grant tool permissions or count repeated observations as independent evidence.

The internal `/agent/memory` route accepts only the existing worker bearer. The runtime database role has SELECT access only. Report JSON and full-field CSV retain the memory identities used for each calculation.
