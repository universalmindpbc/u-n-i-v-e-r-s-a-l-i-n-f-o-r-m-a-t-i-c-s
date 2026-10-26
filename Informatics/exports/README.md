# Complete Universal Informatics dataset

Download one JSON file or one CSV file from Connect your AI → Work with the data.
Each file uses gzip compression for a smaller download. Decompression produces
one `.json` or `.csv`; this is not a ZIP of separate datasets. The full CSV is
large; Python, R and databases are suitable for reading it in chunks.

The export includes every tracked public JSON dataset, scientific text files,
and additional CSV tables. Exact alternate CSV copies of existing Atlas JSON
records are omitted to avoid redundant copies. Their complete field values are
retained. Binary molecular structures, images and PDFs retain URLs and hashes.
Website software, playback media, secrets and private research are outside the
data export. Source-specific licences and attribution remain attached.

JSON has `metadata` and `records`. Each record contains its source path, format,
source SHA-256 and complete data. A version identifies the exact source snapshot.
The CSV contains the same records in a lossless typed table; no JSON reader is
needed. Columns:

- `record_id`: source path; `__metadata__` contains dataset metadata.
- `field`: slash-separated field path; `~1` means `/`, `~0` means `~`.
- `type`: object, array, text, integer, number, boolean, null, or `list:TYPE`.
- `value`: scalar value; object/array rows give the number of members. Homogeneous
  scalar lists use a semicolon-delimited inner CSV row.
- `escaped`: 1 means remove one leading apostrophe from `value` before reading.
  This preserves the original data while preventing spreadsheet formula execution.

Null, empty text, empty arrays and empty objects remain distinct. For example,
filter `record_id` for `quantum/data/oxtr-qrc-qualification.json` to read the
current measurements. Filter for `quantum/experiments/` for permanent experiment
revisions with methods, physical settings and field explanations.

Every published experiment gets its own immutable JSON/CSV revision and joins
the full dataset. Revised results get a new revision. Older revisions remain.
The deployment check rejects missing experiment exports and stale full exports.

The transfer uses checksum-verified parts internally. The website joins them
into the one named gzip file. `manifest.json` records the source inventory,
transfer hashes and the checksum of the decompressed file for independent checks.
