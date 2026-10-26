# M9-family formation summary

The Other Geometry catalogue is now summarized directly in the Atlas and Notion.
These are 2D representations using pairs of alternating equilateral triangles.

| Record | Full Alpha circle | Primary Gamma subset | Native Beta doubling |
| --- | ---: | ---: | --- |
| M9 P24 | 4 | 1 | Present |
| M18 P24 | 4 | 1 | Present |
| M27 P72 | 12 | 3 | Absent |
| M36 P24 | 4 | 1 | Absent |
| M45 P120 | 20 | 5 | Absent |

Each regular six-node set yields one paired-triangle formation. Node sets are
deduplicated while retaining their orientations on the fixed circle. For periods
divisible by six, the full-circle count is P / 6. Gamma counts only sets whose
vertices all belong to primary Gamma. The count is a descriptive property;
native node values and Beta patterns provide the accompanying research context.

The data is additive under `other_geometry_summary`, including counts, native
Beta status, counting rules, and IDs pointing to the complete `formation_catalogue`.
The UI displays it in Geometry Summary, the M9-family cards and Other Geometry.
JSON, CSV and the download ZIP carry matching fields. Notion has two number
properties, `Stellated count (Alpha)` and `Stellated count (Gamma)`, visible in
its existing Geometry Summary view. The previous Star count property is retained.

All five records have Beta sequences. Native doubling currently motivates the
mitosis hypothesis; the other Beta sequences remain available for future research.
Probability selection remains M9 and M18 only. M42 was checked as a comparison:
eight full-circle formations, two in primary Gamma, a defined Beta sequence and
no native doubling. No new fields were assigned outside the five M9-family records.

Regenerate after geometry/literature/probability generation:

```sh
python3 research/informatics/pilot/sync_stellated.py --data-dir Informatics/pilot-v1
INFORMATICS_DATA_DIR=Informatics/pilot-v1 python3 research/informatics/pilot/export_csv.py
python3 tools/check-informatics-stellated.py
```
