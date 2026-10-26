# Molecular Docking and Quantum staging release

Four peer views: Gene, Protein, Molecular Docking and Quantum. The new views have independent scenes and controls. The first study uses oxytocin Tyr2 and receptor Phe291 in deposited 7QVM coordinates. The approach animation is illustrative, with contact measurements shown only at the final bound pose. The separate Quantum scene displays stationary, calculated fragment orbitals and density, with a visible model-method panel and exact CUBE downloads.

The supplied illustration is an interaction reference. Its Tyr/Trp parallel-stack claim is not reproduced as experimental fact. Source geometry identifies the Tyr2/Phe291 contact with a 44.134-degree ring-plane angle; the Tyr2 hydroxyl also contacts Leu316 backbone oxygen. All source measurements, preparation assumptions, field units and limits appear in the model evidence.

## Calculation

PySCF 2.14.0, PBE0/def2-SVP, density fitting, grid level 3, neutral singlet, 108 electrons, 31 atoms. SCF converged to -617.172473610136 Hartree. The 7QVM heavy-atom geometry is fixed, and idealised hydrogens cap the two side chains as p-cresol and toluene. Gas phase, no embedding, no optimisation. No docking score, affinity, activation prediction or time-dependent electronic response is claimed.

All CUBE checksums and fragment coordinates are verified. Orbital sample norms are within 0.2% of one. The coarse visualization density grid integrates to 104.209 electrons because it undersamples core peaks; the physical electron count from composition and charge is 108. The grid is explicitly labelled as unsuitable for electron-count integration.

## Coordination

Created and worked in an isolated checkout. Integrated latest verified staging 1f4beba506faa4201c3553648a08cabcfa30dc8a, containing the Universal One changes and the truncated icosahedron. Preserved geometry.js, the geometry menu order, root Informatics/index.html, Universal One and research-library/API files. Coordinated the staging slot with both chats before upload. Universal One began a new isolated research-trigger task during this work and was notified to wait for this release.

## Verification

Scientific data checks, existing Molecular Atlas coordinate/geometry checks, gene checks and atomic-DNA checks pass. Browser proof covers four fixed viewports, separate direct URLs, all four quantum fields, motion controls, surface controls, exact downloads, invalid-field rejection, keyboard rotation, Gene/Protein return, truncated geometry and embedded entry. The browser has process-level audio disabled; native fullscreen is never invoked. Final live verification and deployment hashes are recorded in verification.json.

## Release boundary

Existing Panthur Informatics-only staging workflow. Source is committed before upload. The completed manifest and live proof are committed and pushed to origin/staging. No production deployment, main update or production tag is authorized.

Final deployed source: `02b9540aa2707eda382bf4941d24388207f67ca3`. All four viewport widths and both embedded journeys pass against live HTTP after the final metadata corrections. Thirteen changed website files and nine preserved shared assets match the committed bytes. The final staging verification commit contains this report, screenshots and the deployment manifest.
