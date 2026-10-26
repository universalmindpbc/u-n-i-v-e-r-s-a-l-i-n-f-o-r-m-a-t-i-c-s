# Molecular Docking visual repair

Ben reported a static molecule while the approach counter advanced, a displaced ligand at the bound endpoint, and missing geometric overlays. The original source was reproduced from commit `840ae0e6`: ligand pixels were identical at progress 0, 0.5 and 1. This corrects the earlier release record, whose playback checks inspected progress state without establishing visible atom motion.

Custom shape state now updates the Mol* Canvas3D scene and visibility before commit. Complex uses a side view of the approach. Rings has a closer camera and connects the nearest ring carbons with a dashed gold line. The deposited coordinates remain unchanged: the Tyr2/Phe291 aromatic contact is tilted, with a 3.541 Å nearest carbon distance, 5.857 Å centroid distance and 44.134-degree plane angle. This is an illustrative approach to an experimental final pose, not a docking search or predicted trajectory.

All existing geometry constructions return to the Docking sidebar with scale, rotation, offset and opacity. The construction follows the active focus and is retained in the saved view. Quantum functionality and data remain unchanged in this release.

## Verification

The dedicated Chromium process has audio disabled before navigation. A new regression suite compares screenshot pixels at three progress values, during Play/Pause, at automatic completion, and after restoring Bound pose. The candidate passes desktop and mobile, including visible geometry, all four geometry controls, exact view export, no horizontal overflow and no browser errors. Baseline and candidate evidence are recorded separately. Embedded shared-header entry and view changes are also verified. No native fullscreen is invoked.

## Coordination and release

Changes are isolated from the shared checkout. Universal One staging release `78972a30` was integrated before upload, preserving its source and deployment manifest. Deployment uses the existing Panthur Informatics-only workflow. Final live results, source hashes and manifest are recorded with this release. Production is excluded.

Deployed source: `648df4f2bd34ffb2487d737b31011e7fbdee363b`. Live desktop and mobile screenshot comparisons pass: the ligand moves approximately 204 and 151 pixels respectively across the approach; Play moves, Pause stops, completion reaches the deposited pose and Bound pose restores identical rendered coordinates. All five updated assets and ten preserved assets match committed bytes.
