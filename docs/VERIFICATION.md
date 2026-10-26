# Verification

The repository preparation checks verify source hashes, the pinned generated-asset manifest, documentation links, excluded-word policy for newly authored copy, and selected syntax. They do not repeat clinical studies, execute paid models or re-run every historical browser test.

After written source-use permission:

```sh
python3 tools/restore-source-assets.py --list
python3 tools/verify-source-manifest.py
```

The first command lists the pinned external assets without downloading them. The second verifies the imported source bytes. To reproduce a data-heavy check, restore its required asset prefix first. Node.js syntax checks and Python compilation can inspect source without contacting providers. Read an individual research tool before execution: some tools intentionally perform network research, model calls, downloads or cloud jobs.

Historical verification directories under `research/informatics/` describe their original source versions and environments. Treat them as provenance records. Current live deployment verification remains part of the established I AM/Panthur workflow.
