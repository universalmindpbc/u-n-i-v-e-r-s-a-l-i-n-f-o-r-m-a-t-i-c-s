# Astra cloud worker — activation plan

Prepared 26 October 2026. Hosting allowance: **US$25/month**, separate from model API calls and quantum hardware. Website hosting stays on Panthur; the existing staging broker stays in place.

## Runtime

One on-demand AWS Lambda CPU worker (4 GB, 15-minute maximum, concurrency one) runs LangGraph, Biopython, statistics and local CUDA-Q. A small dispatcher checks the existing queue once per minute. It reserves at most 120 worker invocations per calendar month before invoking. The broker retains six studies/day, forty/month and the per-study model allowance. Encrypted S3 retains checkpoints, model reservations and JSON/CSV reports. Secrets Manager holds the existing worker bearer, OpenAI key and Gemini key. ECR holds the image; a bounded CodeBuild job builds it on Linux.

Wolfram Language and Quantum Framework are available through the free remote Wolfram Cloud MCP service. A fresh QuantumOperator evaluation passed on 26 October. The existing OXTR recipe still uses its recorded Wolfram receipt; new scientific model evaluations remain a separate, recorded tool action. qBraid and Aquila remain independent remote services, with new paid runs disabled.

The worker has no Braket submission, IAM mutation, billing mutation or GPU provisioning permission. It has no public function URL. Cloud roles cover only its own state, logs, secret and invocation. The schedule defaults to disabled during deployment and is enabled only after cloud execution passes verification.

## Monthly estimate without free-tier credits

| Item | Budget assumption | Estimated USD |
|---|---|---:|
| Main CPU worker | 120 invocations × 900 seconds × 4 GB | 7.20 |
| Dispatcher | 44,640 invocations × maximum 15 seconds × 128 MB | 1.40 |
| Secrets Manager | one secret and bounded reads | 0.65 |
| ECR, S3, requests and logs | two images, small study records, seven-day logs | 1.50 |
| Container build reserve | two bounded builds | 0.25 |
| Total planning estimate | before taxes; workloads normally finish earlier | **11.00** |

Use a US$15 operational target inside the US$25 allowance. Storage, network and tax variations remain outside the compute-attempt counter; this is an estimate, not an AWS billing guarantee. No paid QPU/GPU cost is included. No always-on VM, NAT gateway or provisioned concurrency is planned.

Pricing checked: [Lambda](https://aws.amazon.com/lambda/pricing/), [Secrets Manager](https://aws.amazon.com/secrets-manager/pricing/), [ECR](https://aws.amazon.com/ecr/pricing/), [S3](https://aws.amazon.com/s3/pricing/), [CodeBuild](https://aws.amazon.com/codebuild/pricing/).

## Activation sequence

1. Deploy `bootstrap.template.json` in us-east-1: private bucket, image repository, empty runtime secret and a scoped build role.
2. Upload only the allowlisted build context; build and scan the image, record its digest.
3. Store the existing worker bearer and OpenAI API key in the new runtime secret; never include them in source, build logs or browser output.
4. Deploy `runtime.template.json` with that image digest and schedule disabled. No new broad administrator credential is created.
5. Stop the local worker, run a cloud study and verify progress, model response, archived JSON/CSV, checkpoint recovery and browser rendering.
6. Enable the scheduler and verify a second request is claimed without the Mac worker. Record the runtime, source revision and hosting controls.

Current state, 26 October 2026: both staging stacks are active in us-east-1. The scheduler is enabled. The first AWS study completed in 534.26 seconds with the Mac worker stopped; task `32aedf33-31af-4632-a169-81736771bbca`. The subsequent automatic-dispatch browser check is recorded in `research/informatics/quantum-unified-20261026/`.

The current runtime image is pinned to `sha256:1247bf2b972331af749e16140feb3040fa70c633213c4327a80a7576abdc9823`, built from source `33c384d3d`. The live Gemini review gates passed staging task `95d33a87-ecbf-4700-a178-112daa83ac2a`; its design and findings both received approval. A report-context reservation issue was corrected, then the same task resumed from its checkpoint in 58.71 seconds without repeating its calculations. JSON/CSV delivery and browser rendering passed. See `research/informatics/live-agent-20261026/`. The Linux build executed a CUDA-Q CPU smoke test. OS packages were upgraded and unused compiler headers omitted. ECR reports no critical findings and 19 high, 23 medium, 9 low and one undefined finding in remaining base OS packages. This is not a clean vulnerability scan. The runtime has no inbound function URL, installs no packages at execution time, runs fixed reviewed code, and does not execute model-written commands. Track vendor fixes for libcap, PCRE2, curl, OpenSSL and RPM; rebuild the pinned image when patches become available.

The staging research service now reads the isolated Neon branch `astra-memory-20261026` (`br-jolly-union-azkvnqqq`). The new literature and research-direction revision are tested there. The existing parent database and production Worker configuration remain unchanged by the literature update. Preserve this separation until Ben approves publication.

Gemini uses a shared encrypted S3 ledger with the existing US$1 test allowance across all studies. At this verification, accounted usage and retained reservations total US$0.628036, leaving US$0.371964. Neither model can raise this ceiling. The new live design/final reviews accounted for US$0.189400. New paid Aquila/GPU jobs remain disabled.
