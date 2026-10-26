# Universal Informatics connectors — 24 October 2026

## Scope

One read-only service exposes the same approved publication records and complete M1–M1001 Atlas data used by Universal One. The source dataset contains 811 publications, 149 permitted full texts and 1,001 Atlas records. Public passage access is restricted further to recorded CC BY or CC0 licences; the live inventory gives the exact count. Metadata, quotations, commentary, probability models, native and reduced-to-M9 views, nested sequences and evidence status remain intact.

The website entry is **Connect your AI**. Provider buttons open setup guides; they never claim an account is connected merely because a provider page opened. API/MCP are public and read-only. A2A invokes the existing Universal One research service with mandatory retrieval and citations. The first A2A release accepts independent messages, without persistent tasks, subscriptions or autonomous publication.

## Interfaces

- REST: `/api/v1/inventory`, `/api/v1/publications`, `/api/v1/publications/{id}`, `/api/v1/publications/{id}/passages`, `/api/v1/atlas`, `/api/v1/atlas/{id}`, `/api/v1/atlas/{id}.csv`.
- OpenAPI 3.1: `/api/v1/openapi.json`.
- MCP Streamable HTTP: `/mcp`; eight read-only tools plus inventory resource. Protocol versions 2025-11-25, 2025-06-18 and 2025-03-26.
- A2A 1.0 JSON-RPC: `/a2a/v1`; discovery at `/.well-known/agent-card.json`. Send `A2A-Version: 1.0`; `SendMessage` returns a direct message with source metadata. Unsupported tasks/streaming/push return protocol errors.
- Data reads: 120 requests/minute/IP. Public A2A: 30 questions/day, separately budgeted from website visitors. The API/MCP do not invoke a paid model.
- Origin checks apply to browser MCP/A2A calls. Server-to-server clients need no Origin. No API credential is accepted at public connectors.
- JSON and CSV publication exports are paginated at the API; browser/CLI exporters assemble every page. CSV record_json preserves nested fields. The Atlas archives include the complete original JSON and lossless field-path CSV records. Full-text articles are not bundled into these exports.

## Research collaboration

Ben confirmed Synapse and DeSci accounts are registered. DeSci replaces the earlier Triall proposal. Synapse and DeSci authentication succeeded on 24 October. A private Synapse project and a private DeSci draft are prepared. Private credentials are outside the repository and website.

- Synapse: `publish-synapse.py` verifies hashes, then uploads only with explicit `--upload --project synID` and a personal access token in the local environment. Existing project permissions are retained. The private project is `syn78366147`, with an owner-only ACL. Ben completed account certification on 24 October. The three-file connection check then uploaded successfully into `syn78366366`; downloaded bytes, SHA-256 hashes, provenance and owner-only permissions all passed verification. The adapter checks certification before creating a folder. No research content or participant data was uploaded. OAuth requires registration and verification by Synapse before use.
- DeSci: `draft-desci.mjs` verifies the same export and optionally uploads to an existing private draft. `--list` and a three-file connection-check upload were tested with the registered account. The draft is `pC6RBXus7yEUVvcvbjLHdzp77rN8UAUwEw__O6K9aC0`. DeSci returned HTTP 500 after saving files; the adapter reconciles saved raw-CID files against their local SHA-256 instead of duplicating uploads. Larger files with an unresolved error stop for inspection. The saved check contains no research text or participant data. The adapter never publishes, assigns a licence, signs blockchain transactions or claims an identifier was registered.
- The package manifest contains SHA-256 file hashes; integrity.json hashes the manifest. A local hash is an integrity check, not a third-party timestamp or scientific assessment.
- New research results, author attribution, licences and external publication need a specific reviewed package. No third-party full-text articles are silently republished.

## Client compatibility and verification boundaries

ChatGPT, Claude, Grok, Perplexity and Mistral guides follow documented custom-connector workflows. Gemini CLI and NVIDIA NeMo have downloadable configuration fragments. DeepSeek has a tool-calling adapter. Meta Muse has a custom API guide. Personal-agent guides inherit platform connector permissions rather than claiming a universal A2A personal-agent endpoint.

No third-party consumer account installation is represented as tested until its actual account journey succeeds. SDK/protocol and OpenAI API tests are recorded separately from consumer-account setup. Developer examples need their provider dependencies and account access. The public endpoint test checks this service, not the visitor’s external account.

## Primary documentation consulted

- OpenAI: https://developers.openai.com/plugins/deploy/connect-chatgpt and https://developers.openai.com/api/docs/guides/tools-connectors-mcp
- Dots: https://learn.chatgpt.com/docs/dots/getting-started
- MCP: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
- A2A: https://a2a-protocol.org/v1.0.0/specification/
- Claude: https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp
- Duet: https://claude.com/resources/articles/claude-duet
- Grok: https://docs.x.ai/grok/connectors and https://docs.x.ai/grok-bot/overview
- Perplexity: https://www.perplexity.ai/en-GB/hub/workshops/perplexity-computer-for-buyside-professionals
- Gemini: https://geminicli.com/docs/reference/configuration/ and https://ai.google.dev/gemini-api/docs/function-calling
- Mistral: https://docs.mistral.ai/studio/connectors
- DeepSeek: https://api-docs.deepseek.com/guides/tool_calls/
- NVIDIA: https://docs.nvidia.com/nemo/agent-toolkit/latest/build-workflows/mcp-client.html
- Muse: https://platform.muse.ai/docs and https://about.fb.com/news/2026/09/introducing-muse-small-business/amp/
- Synapse: https://docs.synapse.org/synapse-docs/client-configuration and https://docs.synapse.org/synapse-docs/using-synapse-as-an-oauth-server
- DeSci: https://codex.desci.com/readme/quick-start and https://desci-labs.github.io/nodes/
- DeSci adapter paths and key header checked against https://github.com/desci-labs/nodes/tree/develop/nodes-lib/src (shared/api.ts, shared/config/index.ts, shared/util/headers.ts, node-only/file-uploads.ts).

## Later work

Reviewed research packages for Synapse and DeSci; OAuth consumer onboarding where supported; account-specific A2A quotas and persistent tasks; and the scoped CORAL research coordination prototype. No 24/7 paid research jobs were enabled by this release.

## RCT workflow

DeSci can preserve versioned study protocols, analysis plans, code, shareable datasets and results. It is not the clinical trial operations system or formal trial registry. Synapse supports shared research files and provenance. An Australian RCT can be registered through ANZCTR; randomisation, consent, clinical data collection and safety monitoring need their own governed workflows.

- https://docs.desci.com/find-help/faq/using-nodes
- https://www.anzctr.org.au/Faq.aspx
- https://docs.synapse.org/synapse-docs/synapse-user-account-types
