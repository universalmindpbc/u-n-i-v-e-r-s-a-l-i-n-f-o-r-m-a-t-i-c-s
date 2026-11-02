# Production release scope

Ben explicitly approved: “Okay, that's all approved. So we can, um, we can push all of these informatics builds to production now. Thanks and namaste.”

Promote only the approved youth research, black-background correction, related exports and Universal One retrieval, with exact bytes from staging fc218a2f82982f7ddf66e32fba3acf2ed7178ef7. Preserve production baseline 57747030461016e4fc2c29bd4534df483118857d outside the explicit allowlist. Lab work and deployment-script additions remain staging-only. The prior release owner confirmed the same scoped promotion/reconciliation pattern; no active deployment or owned release lock was reported. Do not touch the stale shared-checkout lock.

The review interface and screening exports remain staging-only. The immutable source-batch JSON remains a necessary audit/test dependency. Unqualified candidates are excluded from the 105-paper published library, approved retrieval and complete public exports.

Production main advances to this scoped release; its release audit and manifest then merge into latest staging while preserving all unrelated staging trees. The annotated production tag uses Australia/Sydney time. Existing Panthur and environment-specific Wrangler deployments remain the publication mechanisms.
