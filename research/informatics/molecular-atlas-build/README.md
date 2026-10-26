# Molecular Atlas

An experimental-structure viewer for the eight research subjects agreed on 21 October 2026. The source catalogue names DOCK2 and uses a human GluN1/GluN2B assembly for the NMDA learning-and-memory context. All public φ terminology is Phi Ratio.

## Build

Use Node.js 22 or newer. Run `npm ci` and `npm run build` in this directory. This compiles the pinned Mol* 5.13.1 engine into `Informatics/molecular/vendor/engine.js`. The HTML, CSS and application controller are buildless. Production uses the committed bundle; there is no runtime dependency on a package CDN.

`viewer.js` owns scientific rendering, selection and camera state. `geometry.js` defines unit-circumradius constructions independently of molecule data. `tools/check-molecular-atlas-browser.mjs` checks the candidate by intercepting requests to the permanent staging hostname. Set `ATLAS_LIVE=1` for unmodified live HTTP verification. Browser tests use a separate headless Chromium process with `--mute-audio`; they never invoke native fullscreen.

## Shared navigation and video

Research Atlas contains Geometry Atlas and Molecular Atlas, followed by the existing About, Sound, Vision, Language, Universal Research and Partners entries. Molecular Atlas loads in a transparent iframe beneath the shared header. The existing background video, playback position, fullscreen state and audio controls stay in the parent experience. The selected molecule survives area changes.

The standalone route uses the same background video with muted playback and explicit pause and sound controls. Reduced-motion preference leaves that video paused until Play. The canvas uses an alpha background; the black shade and translucent panels match the Geometry Atlas treatment. The brand uses the existing flower logo with bold UNIVERSAL and light INFORMATICS.

## Scientific scope

The catalogue is a research collection, not a single gene family or a measured interaction network. Coordinates are downloaded from the public RCSB Model Server as BinaryCIF and stored locally, with SHA-256 checksums and source accession links. A thumbnail is rendered from the same coordinates as its viewer. Protein-load failures stay explicit; no fabricated structure replaces a failed download.

| Target | PDB | Displayed label chains | Scope |
| --- | --- | --- | --- |
| OXTR | 7QVM | E; ligand D | Engineered receptor complex; G-protein/antibody context optional |
| SLC6A4 | 5I6X | A | Thermostabilised paroxetine-bound transporter |
| DRD2 | 6CM4 | A | Risperidone-bound receptor–endolysin fusion |
| NMDA | 9IYQ | A, B, C, D | Human GRIN1/GRIN2B; Mg-free state; glycine and antagonist 7RC |
| IL6 | 1ALU | A | Human cytokine |
| NR3C1 | 1M2Z | A, C | Engineered ligand-binding domains; dexamethasone |
| DOCK2 | 6TGB | A, C | DOCK2 from the DOCK2–ELMO1 deposited model |
| CD38 | 1YH3 | A | Engineered extracellular domain |

Each source entry is a specific deposited model. Complex partners expose additional deposited chains; they do not generate missing subunits or symmetry mates. Protein sequence positions use the deposited construct, while picked residue labels preserve author numbering. Missing coordinates stay unavailable. DNA regulatory coordinates remain separate from protein residue coordinates.

## Gene / Protein switch

The opening Gene view shows a 48-base window of reference DNA as a clearly labelled, constructed B-DNA atomic model. Helix, Atoms and Surface render the same coordinates. The Protein view retains the deposited experimental models. The viewer never describes the DNA model as a measured gene or cellular fold. Molecular models remain separate from the exploratory geometry layer.

`fetch-genes.py` retrieves human GRCh38 gene and canonical transcript annotations from Ensembl, genomic DNA from UCSC, and an independent genomic sequence from Ensembl. The build requires exact sequence agreement. It stores nine genes for the eight targets: NMDA has separate GRIN1 and GRIN2B choices. Source snapshots and sequence checksums are retained. The complete gene includes introns; exon markers and transcription starts refer to the named canonical transcript. All displayed genomic coordinates are 1-based and inclusive. Minus-strand DNA is reverse-complemented into the gene's 5-prime to 3-prime orientation.

`dna.js` assembles each 48-base model from the selected reference sequence and its antiparallel Watson-Crick complement. `build-dna-templates.py` extracts and checks eight heavy-atom templates from the pinned BioExcel NAB Langridge B-DNA fixture in `dna-model-source/`. Rise is 3.38 Å and twist is 36° per base pair. Both 5-prime termini omit the terminal phosphate; hydrogens are implicit. Helix uses a nucleic-acid cartoon, Atoms uses ball-and-stick with element colours, and Surface computes a molecular surface from the same atoms. No sequence-dependent relaxation or chromatin fold is inferred. The interface supports exon/TSS jumps, sequence windows, full FASTA and annotation downloads, modelled-segment PDB downloads, and view exports containing the construction method and source checksum.

Collection conceals the structure workspace and its controls while retaining the background film. Its stronger panel and card backgrounds keep labels readable. Returning to Structure preserves the selected geometric construction.

Run `node tools/check-molecular-genes.mjs` for DNA source, strand, exon and boundary checks. Run `node tools/check-molecular-dna-model.mjs` for template-source round trips, antiparallel pairing, phosphodiester continuity, glycosidic distances and terminal chemistry. The protein browser suite uses `?level=protein`; the entry suite exercises the default Gene view and shared media continuity.

## Geometry

Five Platonic solids use analytic unit-radius vertices, with edges selected by their common minimum distance. The stellated octahedron is the compound of two opposite tetrahedra. The truncated icosahedron follows it in the menu. It is an Archimedean solid formed by cutting each icosahedron edge at one third from both endpoints, then normalising the 60 resulting vertices to unit circumradius. Its 90 equal edges bound 12 regular pentagons and 20 regular hexagons; this is the ideal geometry associated with C₆₀, not an atomistic carbon model. Odom's original planar midpoint/circumcircle construction is repeated on four faces of a tetrahedron. This tetrahedral arrangement is an explicitly labelled exploratory extension. Its midpoint-segment/adjacent-extension ratio equals `(1 + sqrt(5)) / 2`. The Phi sphere contains 89 points with angular increment `2π(1 − 1/φ)`.

Geometry is centred on the target bounding sphere, with independently controlled scale, Y-axis rotation, X-axis offset and opacity. No spatial fit or biological mechanism is inferred. View exports contain the structure checksum, camera and all geometric parameters.

## References

- Truncated icosahedron: https://mathworld.wolfram.com/TruncatedIcosahedron.html
- Buckminsterfullerene terminology: https://www.acs.org/molecule-of-the-week/archive/b/buckminsterfullerene.html

- RCSB entry and publication links are preserved per target in `data/catalogue.json`.
- Mol*: https://molstar.org/ (MIT license preserved with the bundled engine).
- Hung & van Lamoen, *Odom's Triangle*, IJCDM 6 (2021), 68–77: https://www.journal-1.eu/2021/Tran%20Quang%20Hung%2C%20Floor%20van%20Lamoen.%20Odom%27s%20Triangle%2C%20pp%2068-77.pdf

## Release

Use the existing `./deploy-panthur-staging.sh informatics` workflow. Commit the source before uploading; verify the live staging journey; record the deployment manifest and browser results; push the staging branch. Production requires Ben's explicit publication approval.

## Molecular Docking / Quantum

These are independent peer views alongside Gene and Protein. Their first study is OXT–OXTR. `interaction-ui.js` owns their controls; `interactions.js` owns their Mol* shapes and volume representations. Geometry constructions are available in Gene, Protein and Molecular Docking. Docking anchors the overlay to the active Complex, Pocket or Rings focus, and includes its parameters in the view export.

Molecular Docking uses deposited 7QVM heavy atoms. Its final pose shows oxytocin Tyr2, receptor Phe291 and the Leu316 backbone oxygen. Six-carbon least-squares ring planes give a 44.134° angle, 5.857 Å centroid distance and 3.541 Å closest ring-carbon distance. Tyr2 OH to Leu316 O is 2.727 Å. This is an aromatic contact, with no assignment as a parallel pi stack. The approach slider applies a rigid illustrative translation; it does not run a docking search or a molecular-dynamics calculation. Bound-pose measurements disappear during that motion.

Docking shape transforms call `canvas3d.update` and `syncVisibility` before committing the scene. Changing representation state alone leaves GPU geometry stale. The Complex camera looks across the approach, and Rings frames the local contact with a dashed connector between the closest ring carbons. `tools/check-molecular-docking-motion.mjs` compares rendered screenshot pixels at three progress values, during Play/Pause, at automatic completion and after restoring Bound pose. It also checks visible geometry and exported parameters on desktop and mobile. `ATLAS_BASELINE=1` reproduces the original stationary-pixel fault from commit `840ae0e6`; `ATLAS_LIVE=1` checks the deployed release.

Quantum renders a stationary PBE0/def2-SVP calculation for a neutral, singlet, 108-electron p-cresol/toluene pair. The fragment keeps the deposited Tyr2/Phe291 side-chain heavy atoms; replacing the omitted backbone with methyl hydrogens creates the caps. All added hydrogens are idealised. The gas-phase calculation has no protein embedding, membrane, solvent or ions and no geometry optimisation. It does not predict binding affinity, receptor activation or time-dependent electron motion. Gold and cyan orbital lobes represent wavefunction signs, not charges or electron paths. HOMO, HOMO−1, LUMO and total electron density each have independent CUBE data with SHA-256 verification before rendering.

The reproducible calculation source is `build-interactions.py`, with the original RCSB PDB pinned under `interaction-source/`. Run with Python 3.12, PySCF 2.14.0, NumPy 2.5.4 and SciPy 1.18.1. It uses density fitting, a level-3 integration grid and 1e-9 Hartree convergence. Four threads are requested; this macOS PySCF build reports no OpenMP support, so that setting has no effect. `--geometry-only` rebuilds contact data and capped coordinates without rerunning DFT. CUBE files use Bohr coordinates and a requested 0.38 Bohr sample spacing. Actual axis steps are stored in each CUBE header. This visualization grid undersamples sharp core-density peaks; it is not an electron-count integration grid. The molecular electron count is determined by composition and charge. The orbital grid norms remain within 0.2% of one.

Run `python tools/check-molecular-interactions.py` with NumPy for source-coordinate round trips, measured geometry, fragment rigid-transform preservation, chemistry, CUBE alignment and hashes. Run `node tools/check-molecular-interactions-browser.mjs` for all four viewport sizes and embedded entry. Set `ATLAS_LIVE=1` after the staging deployment. The browser process verifies `--mute-audio`; native fullscreen is disabled during embedded entry. It exercises independent navigation, playback and pause, field switching, surface controls, error states, exact downloads and preservation of Gene/Protein and truncated-icosahedron geometry. After bundling, strip trailing whitespace from the generated engine before committing.

## Quantum Teaching mode

`quantum-teaching.js` builds a schematic six-carbon phenol ring, six paired p-orbital lobes and shared pi bands using independent Mol* shapes. The guide markers move by a scene-updated rigid rotation, and screen-projected labels follow camera movement. The three teaching steps and play/pause, labels, guides and speed controls follow the approved ImageGen concept. These shapes are idealised teaching geometry, not derived electronic fields. Same-sign upper lobes illustrate one bonding combination; the aromatic pi system occupies several molecular orbitals. Motion guides do not represent electron trajectories or physical timescales.

Quantum defaults to Teaching. `?level=quantum&quantum=calculated` opens the retained calculation directly. Mode switches preserve each mode's settings; leaving the view pauses teaching motion. Teaching JSON exports contain schematic provenance, while Calculated exports retain the computational source. The permanent symbolic-motion note stays beside the scene.

Run `node tools/check-quantum-teaching-browser.mjs` for rendered lesson changes, actual marker motion and pause, independent mode settings, annotation controls, teaching exports and four viewport widths. Use `ATLAS_LIVE=1` for live staging. The existing interaction suite explicitly selects Calculated for the four fields and invalid-CUBE check.
