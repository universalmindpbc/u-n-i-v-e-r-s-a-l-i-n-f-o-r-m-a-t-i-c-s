# Quantum Teaching and Calculated modes

Ben approved the ImageGen concept before implementation. Quantum now opens a rotatable teaching model with three steps: the carbon framework, six schematic p orbitals and the shared pi system. Labels, symbolic motion guides, play/pause and teaching speed are independently controlled. A persistent note identifies the motion as symbolic. Teaching uses idealised phenol geometry with a peptide attachment cue, not a new electronic-structure calculation. The approved reference image is retained beside this record.

Calculated retains the original capped Tyr2-Phe291 fragment, four independently calculated fields, controls, source records, checksum verification and exact field downloads. Each mode retains its own settings. Teaching exports carry teaching provenance. The shared Atlas pauses the guide animation on view changes and exit.

The colours show opposite wavefunction signs in an illustrative bonding combination. They are not charges. The complete aromatic pi system occupies several molecular orbitals; the teaching bands are a qualitative summary, not a complete set of calculated orbitals. Ring hydrogens are implicit, and the number and speed of guide markers have no physical interpretation.

## Docking clarification

The supplied infographic's Tyr-Trp sandwich does not represent the selected 7QVM geometry. The focused Rings view shows Tyr2 and Phe291; tryptophan has a fused indole group, while phenylalanine has one aromatic ring. The paper describes Trp288 in the activation rearrangement. Direct coordinate comparisons in docking-clarification.json establish the distinction. Docking now explicitly labels the gold dashes as a distance guide, with a separate explanation for the Tyr2-Leu316 hydrogen-bond contact. Its rendering and deposited coordinates remain unchanged.

## Verification and release

Dedicated muted Chromium checks compare actual rendered pixels for the three lesson steps and moving/paused markers, exercise modes and controls, save teaching provenance, verify pause-on-exit, and inspect desktop and mobile layouts. The existing calculated-field and embedded-entry checks remain part of release verification. The existing Panthur staging workflow is used after integration of the latest completed staging release. Production remains excluded.

## Completed staging proof

Source `a5310660c3886b21f458d83d8f85491ec075b428` integrates the completed Universal One release `fed46344f1572381dfd14a24983a8ec59bad728c`. The existing Panthur workflow uploaded five Molecular Atlas assets. Eighteen live asset downloads match the committed source and deployment manifest exactly, including the preserved Universal One source and calculated fields.

Live Teaching checks pass at 1470, 1024, 390 and 320 pixels. The desktop animation check compares moving and paused model pixels; all four layouts exercise the lesson steps and independent modes. Shared Atlas entry passes at desktop and phone widths, with visible molecular pixels checked after the camera settles. The original calculated-field checks passed on the candidate at four widths, and the live invalid-field check retains a visible checksum failure. Screenshots retain the reviewed desktop and phone layouts.

All browser checks used a separate Chromium process with the mute-audio flag verified before navigation. Native fullscreen was disabled. Automated pages were paused and closed. Production was not deployed; its remote main reference remains `d82c7ae080a5236cd1029bd9605d7b4e3988feb4`.
