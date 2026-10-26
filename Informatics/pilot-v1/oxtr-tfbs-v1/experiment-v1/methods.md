# OXTR binding-site geometry: first interactive experiment

Version: oxtr-coordinate-experiment-2026-10-20-v1. Exploratory coordinate comparison.

## Question and sources

Explore whether the Alpha, Beta, Gamma and Epsilon geometries of Fibonacci M9 P24 help predict OXTR binding-site locations. DNA observations and the original Atlas reference appear side by side. Blend is off by default. Enabling Blend adds the selected Atlas guide to the DNA panel while retaining the independent reference. The DNA panel offers an original line, a full-position circle before compression, and an M9 circle. Three separate selections are available: 128 published-study observations, ENCODE occupancy regions, and JASPAR motif predictions with UCSC score at least 400. The complete underlying catalogue retains all lower-score predictions. Source observations and overlapping predictions are not deduplicated into a count of confirmed binding sites.

The Studies collection contains two sequence elements (US-1 and US-2) and 126 screened BrainTF ChIP-seq occupancy regions from Loupe et al. (2024), with separate binding outcomes. All 128 are interactive in every view; the within-Studies filter isolates either evidence type. The 126 are repeated observations across targets and brain contexts, not 126 distinct binding sites. Selection criteria and provenance are retained in study-additions-v1. Animal findings and human reports without resolved reference coordinates remain in the literature catalogue; they are not projected onto this human reference. The 2006 study's 15 reported sites have no recovered individual coordinates in this release and do not appear as plotted sites. ENCODE regions include multiple experimental and cell-type contexts. JASPAR matches are predictions, not experimentally established binding events.

## Fixed mapping

Human GRCh38 chr3:[8740380,8779628), length 39248 bp. All raw intervals use 0-based starts and exclusive ends. OXTR runs on the minus strand. The line runs left to right from high to low genomic coordinates; the circles run clockwise from north. Origin is genomic base 8779627, the first base at the high-coordinate edge of the saved window. This is a plotting origin, not a transcription start site.

For a raw interval [s,e), first intersect it with the saved window: a=max(s,8740380), b=min(e,8779628). Original intervals remain unchanged in the exports. Clipped intervals are labelled. Local interval boundaries are d0=8779628-b and d1=8779628-a. The representative base is d=floor((d0+d1-1)/2). For an even width this picks the first of the two central bases in the OXTR direction. The genomic representative is 8779627-d. This representative is a plotting convention; a BrainTF or ENCODE region centre is not a resolved binding nucleotide.

The DNA line and full circle retain interval extents. A full-circle turn is d/39248; the M9 residue is d mod 9, with zero labelled 9. Both circles use x=sin(theta), y=-cos(theta), with theta=2*pi*turns or 2*pi*residue/9. M9 collisions are shown as counts and remain traceable to every source observation. Modulo reduction discards absolute distance and maps many unrelated coordinates to the same node.

Selections exceeding 50 observations use 72 equal-coordinate anchor-count bins; sparse selections show interval marks. On the full circle, outward bar length represents the bin count. Bins are for display and selection only, and do not change the stored coordinates. M9 dot area is proportional to count: radius = 19 * sqrt(count / maximum_count), in SVG units. Zero counts have no filled observation disc. Each view scales within its current selection. Percentages use the total filtered observation count as denominator; they are shares of recorded observations, not biological binding probabilities. Zoom changes the visible line range, never the mapping origin or the circle scale. Individual observations are orange diamonds, with named callouts for US-1 and US-2. Selection is deliberate, never automatic. Selecting a count node or bar opens the underlying records in DNA order. Original intervals remain in the inspector and JSON/CSV downloads.

## Atlas comparison

Use the versioned m9.json reference with its original phase and clockwise orientation. Alpha displays the 24 residues by period index. Gamma uses its six primary indices and two triangle paths. Epsilon uses its canonical seven-visit traversal through those period positions. Beta follows 4,8,7,5,1,2 on the nine-position residue circle.

The shared graph uses a common angular origin at north and clockwise direction. Alpha/Gamma/Epsilon turns equal period_index / 24; Beta turns equal residue / 9. The full-position circle represents the whole 39,248-base DNA window in one turn. The DNA line maps these same Atlas turns using guide_offset = turn * window_length. It shows reference node positions only; the earlier connecting arches have been removed. The original geometry remains visible in the right panel. In M9, the DNA uses nine residue positions while the Atlas guide retains its native nine- or 24-position grid. Equal angles do not equate a period index with a residue value. These fixed registration choices are recorded in overlay-v1.json and its typed CSV equivalent, overlay-v1-fields.csv, and accompany selection exports.

No fitted rotation, arbitrary connections between DNA observations, or inferred physical circular DNA structure is introduced. Atlas path arrows describe the mathematical traversal, not measured movement or a temporal order of transcription-factor binding. The full-circle and compressed views remain separate exploratory comparisons; neither evaluates predictive accuracy in this release.

## Verification and reproduction

Python computes all projections and independently reconstructs the Fibonacci cycle, primary Gamma nodes and Beta interval sums. Biopython parses the reference FASTA and JASPAR templates, rechecks both published sequence placements, and checks the reference sequence and strand of every eligible displayed JASPAR prediction. Exact checks are local to the saved window. The archived motif predictions are not rescanned or experimentally validated here.

Wolfram independently checks the projection rules across every possible base offset in this window, raw-interval test cases, the M9 cycle and canonical paths. Its saved code and actual returned results accompany the Python validation. The browser displays precomputed data and performs filtering and rendering locally; it does not call a Python or Wolfram server for each interaction.

Create a Python environment, install requirements.txt, then run build_experiment.py with --source pointing to the extracted original catalogue folder and --atlas pointing to the original m9.json. The manifest records source and output SHA-256 hashes. Run verify.wl in Wolfram Language for the independent checks. Exported selections retain parameters, source records and the Atlas reference. SVG exports retain vector geometry.

## Interpretation

This first experiment is visual exploration with checked arithmetic. It reports no association score or biological conclusion. A subsequent test requires a defined comparison statistic, appropriate matched controls, sensitivity to the coordinate origin and interval representative, and correction for the comparisons explored. The earlier Fibonacci shuffle probability is a different model and is not applied to DNA coordinates.

## Display audit — 20 October 2026

All three sources use the same fixed window. No bounds are fitted to occupied positions. The reproducible `display-audit.json` and its CSV equivalent compare all 9,547 plotted source coordinates against the archived originals.

- The study selection contains 128 observations: two sequence-mapped elements and 126 BrainTF regions. Selected regions retain their full interval extents on the DNA line and full-position circle, including in dense views. Blank areas are unknown in this collection.
- ENCODE pools 255 clusters from 133 protein targets, 119 cell types and 444 contributing experiments. Of 72 equal sections, 31 contain region-centre anchors and 41 do not. The union of the clipped intervals spans 14,131 of 39,248 bases (36.004%); centres and interval coverage answer different questions.
- JASPAR pools 9,164 matches from 872 motif models at score 400 or above. All 72 sections contain at least one match centre. The source intervals overlap; their union spans 27,912 bases (71.117%). A populated section is not continuously occupied DNA, and a predicted match is not measured binding activity.

These descriptive counts do not establish regulatory activity or a relationship with the Atlas. The line and circle now draw counts at each exact plotted centre, without combining neighbouring bases. The 72-section counts above remain descriptive audit summaries; they are not the displayed position counts. The fixed window and original positions remain unchanged when Blend, view or Atlas sequence changes.

Source definitions: [JASPAR documentation](https://jaspar.elixir.no/docs/) and [UCSC ENCODE 3 cluster documentation](https://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=encRegTfbsClustered). The latter describes clusters combined across factor-specific ChIP-seq experiments and cell types. The audit script is included with the complete Atlas source bundle.

## Position counts — 21 October 2026

The DNA line and full-position circle count observations only when their plotted centre base is identical. The 128 Studies observations occupy 81 distinct centres, with at most six records at one centre. Previously, the 72-section histogram combined 113 observations at 66 centres into one tall bar. That display has been replaced because it obscured the positional differences relevant to this comparison.

The line uses DNA position horizontally and observation count vertically. The circle retains the same angular DNA positions and uses radial mark length for count. Close marks can overlap at the full-window scale. Line zoom exposes the busiest region or 50 bases on either side of a deliberately selected centre. Zoom changes only the visible range; the origin and M9 calculations remain fixed. Selecting a mark retrieves all records at its exact plotted centre. For selections of one or two records, individual diamonds and source intervals remain visible.

These are counts of evidence records at representative centres, not unique confirmed binding sites, binding strength or molecule counts. Original interval extents and study context remain available for every observation. M9 groups these same original centre coordinates by remainder; it does not retain their broad DNA location and has no established predictive value for binding.
