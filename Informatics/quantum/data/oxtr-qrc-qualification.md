# OXTR Study 01 — first measurement phase

Activated 25 October 2026. Model version: `1.1.0-qualification`. Parent study: `oxtr-qrc-01`, protocol `1.0.0`.

The study asks: **Do quantum responses generated from Fibonacci and M9 P24 representations improve our ability to predict transcription factor occupancy around OXTR?** Phi Ratio is included as a separate representation.

This first batch establishes that the mathematical model, GPU simulation and physical measurements run consistently. It does not train an occupancy predictor or establish a biological relationship. The larger prediction benchmark retains the protocol's independent test and classical comparisons.

## Frozen inputs

Two 256-base GRCh38 windows: chr3:8,752,896–8,753,152 and chr3:8,745,472–8,745,728, using 0-based, end-exclusive coordinates. These windows were selected by motif count without reading occupancy labels; selection is suitable for execution qualification, not an unbiased prediction test.

CTCF motif MA0139.2 from JASPAR 2026 was scored on both strands, with pseudocount 0.5, uniform background, relative-score threshold 0.7 and minimum separation of 15 bases. Of 152 complete windows in the locus, 138 had at least three hits. The selected windows contain ten and nine hits.

The 24 descriptors contain twelve positional motif-score bins, six spacing bins and six sequence/context summaries. Native Fibonacci uses prespecified gaps 1, 2, 3, 5, 8, 13, 21, 34 and 55; the Phi Ratio arm uses ratios between consecutive native gaps, with a fixed ten-percent similarity width. Modular residues are not treated as native physical ratios.

Alpha uses the canonical 24-residue cycle. Beta repeats the canonical 4,8,7,5,1,2 cycle. Gamma retains both closed triangle traversals; Epsilon retains its seven-visit path, including the closing point. Paths are resampled onto 24 pulse slots. Sequence weights are mean-centred, bounded and fixed before occupancy labels. Ordinary and seeded shuffled representations form the two controls. All eight arms are run for both windows.

These fixed bounds qualify the instrument route; training-only normalization for the larger predictive benchmark is a separate recorded step.

## Matched physical model

Six atoms occupy a fixed regular hexagon with nominal radius 9 micrometres. Coordinates are rounded to Aquila's 10-nanometre resolution; the model retains all fifteen pair interactions using those rounded coordinates. The frozen C6 coefficient is 5.42e-24 rad m^6/s.

Global drive reaches 2e6 rad/s, with 0.2-microsecond edge ramps. The total evolution lasts 4 microseconds. Twenty-four input values control the global detuning within ±4e6 rad/s. There is no local-detuning assumption and no molecular-carbon interpretation of this Rydberg Hamiltonian.

H(t)/hbar = Omega(t)/2 sum X_i − Delta(t) sum n_i + sum C6/r_ij^6 n_i n_j.

Outputs are six excitation probabilities and fifteen pair probabilities. Site zero is the least significant computational bit. Aquila pre-measurement occupancy is retained; initially empty shots are excluded from the conditional response and counted separately. Post-measurement loss can mimic excitation, so hardware uncertainty includes more than ideal sampling noise.

## Completed numerical checks

All sixteen local CUDA-Q calculations agree with the independent matrix calculation to floating-point precision. Time-step refinement gives a maximum final observable error of 0.00005823 against the continuous SciPy reference. Exported OpenQASM is independently interpreted and checked. Wolfram Quantum Framework independently constructs and verifies the reference 64-state Hamiltonian; its probability difference from the reference is 7.26e-9.

All sixteen CUDA-Q compiled programs completed on the qBraid NVIDIA GPU route, with 100 shots per program. A separate eight-shot bit-order probe passed. All receipts report zero credit cost. The prespecified execution tolerance is six binomial standard errors plus two counts; this is an instrument check, not an accuracy or predictive-performance claim. Shared simulator seeds mean sampling errors across representations are not independent.

Gamma and Epsilon give distinct ideal responses in both windows (largest observable differences approximately 0.041 and 0.034). One hundred shots may not resolve those differences on hardware.

## Aquila execution

Sixteen matched programs, each with 100 shots, were accepted by Amazon Braket. Quantum fees are US$20.80, within the approved cumulative US$22.10 cap including the earlier US$1.30 connection experiment. Storage and taxes are additional. Submission records prevent duplicate paid runs; recorded status is shown in the live diagram. A new question links to this shared study and does not silently purchase another hardware run.

## Biological evidence for the next phase

The candidate occupancy assay is [ENCODE ENCSR932ZMX](https://www.encodeproject.org/experiments/ENCSR932ZMX/), CTCF in GM23338. It has replicated measurements and controls. Five conservative peaks overlap the OXTR locus. Public warnings include read-depth/library and antibody-exemption annotations. This is not a neuron-specific assay. Peak intervals describe regional occupancy, not exact contact bases or affinity.

No occupancy labels entered the qualification features. Chromosome/block-separated training and validation, with OXTR withheld, remain required for the prediction question. A finding needs independent biological evaluation; hardware execution and simulation agreement alone do not prove the hypothesis.

## Records

[Measured and simulated response record](oxtr-qrc-qualification.json) · [Study status](oxtr-qrc-state.json) · [Full prediction protocol](oxtr-qrc-protocol.md).

## Returned measurements and interpretation

All sixteen Aquila tasks completed on 25 October 2026. They returned 1,600 measurements; 1,541 began with all six atoms present, 59 had at least one missing atom, and none were reported as failed. Conditional responses use the 1,541 fully occupied measurements. Mean absolute difference from the ideal reference across all 336 observable values is 0.02675079. This descriptive aggregate is not a predictive-performance measure or a pass/fail threshold.

The results interface compares average excitation across six atoms with the ordinary representation for the same DNA window. Positive values mean a larger atom response, not stronger biological binding or better prediction. Alpha and Beta have negative observed differences in both windows, matching the ideal direction. Gamma and Epsilon have positive ideal differences in both windows but negative measured differences in the second. The shuffled control produces the largest measured response in the second window. These are exploratory observations from one task per case, in a fixed execution order; they do not isolate drift or establish reproducibility or statistical significance.

The current batch holds the hexagonal atom layout fixed. It neither compares alternative geometries nor measures coherence lifetime. The later geometric-stability hypothesis is documented as a proposed mechanism, not a finding of this batch.
