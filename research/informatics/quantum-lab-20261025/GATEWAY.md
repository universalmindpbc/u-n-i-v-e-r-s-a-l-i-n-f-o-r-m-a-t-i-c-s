# Quantum Atlas gateway — current product contract

Updated 25 October 2026. This supersedes the initial per-researcher allocation design. Earlier release evidence remains a historical record.

## Public service

Quantum Atlas — Biological questions via Quantum Computing follows Molecular Atlas. Scientists ask biological questions in natural language. The service clarifies the question and relevant evidence; technical models and numerical details are optional. DNA assembly, haplotypes, DNA-binding prediction and compatible molecular contacts are the initial research directions. No broad quantum advantage is claimed.

Universal Informatics aims to fund public use through shared daily, weekly and monthly budgets. There is no individual researcher approval or personal hardware allocation. Fair-use controls protect shared access; they do not become grant review.

## This staging implementation

- A bounded GPT-6 Astra endpoint returns structured biological research guidance from five curated primary-research summaries. It is not live literature, ENCODE or JASPAR retrieval.
- Natural-language assembly requests containing 3–6 distinct synthetic/public DNA fragments can execute exact classical read ordering and deterministic QUBO/Ising compilation. The assumptions are explicit, ties are reported, and gaps are flagged. This is a teaching exercise, not a genome assembler.
- Questions have a shared planning pool of 30/day, 120/week and 300/month, enforced transactionally in the existing backend. These are question counts, not monetary limits. UTC calendar days/months and Monday-starting weeks apply. The authenticated QA pool is separate and equally bounded. Failed provider calls retain their reservation because a charge may already exist; there are no automatic retries.
- No GPU/QPU dispatch endpoint is active. No hardware funding amount is inferred from the earlier one-job approval. No credentials enter the browser.
- Wolfram Language model export is available; the web endpoint does not execute Wolfram Quantum Framework. The earlier two-atom experiment separately verified that framework and the qBraid GPU route.
- Conversations stay in tab memory. Download preserves the plan and model; no server project history or authenticated private storage is claimed. API requests use store:false; this is not a claim that the external provider retains no operational data.
- The earlier two-atom numerical workspace remains an optional verification view. Its recorded GPU output and timestamped QPU status do not represent new executions.

## Before public hardware activation

Configure funded currency amounts for the shared day/week/month pools and allowlisted provider adapters. Reserve a conservative total cost, including applicable task, shot, compute and storage fees, in all three pools atomically before dispatch. Requests that exceed any pool wait or decline with an honest reset/availability message; do not silently switch to paid overflow. Add fair-use rate and concurrency controls, without individual approvals.

Persist versioned experiments, input hashes, classical baselines and provider receipts. Use idempotency keys and a durable queue. Retries reuse the original submission; deliberate reruns create new records and reservations. Unknown submission outcomes retain reservations until reconciled. Reconcile actual charges before releasing unused funds; failure and cancellation are not assumed free.

Use restricted server identities and separate validated adapters for Aquila analog programmes, annealing objectives and gate circuits. A valid Ising objective alone does not establish an Aquila-compatible embedding. Results distinguish physical measurements from simulations and preserve uncertainty, compiler/device versions, units, bit ordering and missing shots.

An explicitly authorised bounded end-to-end hardware test remains necessary. The earlier Aquila approval covers the already submitted task only. D-Wave access and future Google integration remain unverified/pending.
