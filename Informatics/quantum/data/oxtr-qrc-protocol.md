# OXTR quantum responses — study protocol

Version 1.0.0 · 25 October 2026 · Design for review

**Do quantum responses generated from Fibonacci and M9 P24 representations improve our ability to predict transcription factor occupancy around OXTR?**

## Purpose and present status

This is one biological prediction study. Experimental responses from Aquila enter the predictor itself. NVIDIA CUDA-Q supplies a matched simulation of the same physical model. Strong classical models receive the same input information. No biological study run or result is represented by this protocol. Ben reviews the design before the team executes CUDA-Q and Aquila together.

The first study tests computational predictive utility. It does not assume that DNA, aromatic carbon rings or transcription factors obey the Rydberg-atom Hamiltonian. A successful prediction does not establish a molecular mechanism, quantum advantage, or a music-intervention effect.

## 1. Biological question and data admission

Target: reproducible experimental occupancy of one transcription factor in one cell context. Affinity, exact contact nucleotides and causal changes in OXTR expression are distinct targets requiring different measurements.

The source catalogue supplies traceable OXTR observations and JASPAR motif candidates. Its pooled ENCODE intervals, repeated BrainTF observations and two sequence-mapped elements are sufficient for method development, not an independent benchmark on their own.

Before fitting, select a public human GRCh38 assay with factor identity, cell context, biological replicates, controls, quality metadata and appropriate OXTR coverage. Selection depends on assay quality and coverage, never on a favourable Fibonacci fit. Record accession IDs, source hashes, licences, replicate handling and exclusion reasons in an immutable dataset manifest. If no adequate assay exists, report that data limitation before spending on a predictive hardware study.

Use fixed, non-overlapping 256-base windows on a frozen genomic tiling. Label reproducible occupancy using the assay's prespecified replicate/peak rule. Assay non-detection is an operational comparison label, not proof that binding never occurs. Exclude ambiguous, low-mappability and unsupported regions. Match background windows for sequence composition and assay accessibility where data permit. Report the sampling scheme, class balance and consequent limits on population calibration.

Reserve chromosome 3, including the entire OXTR evaluation window, from feature selection, normalisation, training and tuning. Select models on other chromosomes using grouped genomic-block validation. Keep overlapping windows, shared sequences and all replicates of a region in the same group. Freeze the genomic buffer between groups according to the largest input context. The exact OXTR window is GRCh38 chr3:[8740380,8779628), minus strand. Retain physical base-pair positions alongside modular representations.

Do not label JASPAR matches as experimental positives. Do not treat broad peak centres as measured single-nucleotide contacts. Do not reuse the laboratory target as an input feature.

## 2. Reusable definitions and proposed encoding

The JSON study specification embeds the canonical Atlas definitions, copied from the current versioned coordinate manifest. Arithmetic uses residues 0–8; the diagram displays 0 as 9. Alpha is the 24-step period, Beta is 4,8,7,5,1,2, Gamma contains two triangle traversals, and Epsilon contains its canonical seven-visit path. Gamma and Epsilon share six positions. Their distinction therefore requires a verified temporal or interaction encoding.

The initial input specification is a 24-component vector per DNA window:

- 12 ordered bins of the strongest normalized motif score in each bin, with absent matches encoded as zero;
- 6 gap-frequency bins with fixed physical boundaries [1,2], [3,5], [6,8], [9,13], [14,21], [22,55] bases, normalized by the number of eligible motif pairs;
- 6 descriptors: GC fraction, CpG fraction, normalized motif count, orientation fraction, normalized mean nearest-motif gap, and maximum motif score.

This is a proposed compact feature design, not an empirically selected winning representation. The exact motif library, score transformation, bin assignments, handling of zero/one motifs and gap-pair rule are frozen in the data-admission record before evaluation. Fit any scale or clipping bounds using training data only. Keep this full vector available to every classical comparison, including its Fibonacci-spaced bins.

The ordinary encoding uses the input vector in its declared order. Candidate Atlas encodings apply documented bounded temporal weights/order to the same vector. The initial weighting proposal is w_j = 1 + 0.25 (s_j - mean(s))/max_abs(s - mean(s)), followed by an identical training-fitted normalization and clipping rule for every arm. A constant template receives weights of one. Resample canonical template paths onto 24 equal-time slots using a frozen, recorded interpolation rule; retain closing visits. Beta repeats its six-step cycle four times. Gamma concatenates both closed triangle paths in canonical order. Epsilon uses the saved closed traversal. Alpha uses its 24 raw residues. Record the resulting weight arrays explicitly and test that claimed distinct encodings produce different pulse programs and distinguishable simulated observables.

Native Fibonacci gap features and the Phi Ratio are separate registered feature families, evaluated with the same-data classical controls. Modular residues are not substituted for native ratios. Compare independent families separately and correct for their number. Do not add representations or choose rotations after viewing the final test.

A phase sensitivity analysis repeats the fixed coordinate calculation across the nine possible residue origins. It is reported as sensitivity, never selected as the most favourable main result. A circular plotting convention does not imply circular DNA.

## 3. Physical template and feasibility checks

Initial reservoir: six fixed atoms on a regular hexagon, radius 9 micrometres, nearest-neighbour separation 9 micrometres, with one vertex at north. Use a fixed global driving amplitude of 2 million radians per second, ramped on and off over 0.2 microseconds, within a total 4-microsecond evolution. Encode the input in a global detuning pulse bounded by ±4 million radians per second. No locally addressed detuning is assumed. The north-oriented geometry has distinct rows separated by at least 4.5 micrometres before coordinate discretization. Revalidate the frozen, discretized coordinates against current device limits.

Global driving and array symmetry can make single-site and pair features redundant. Evaluate effective feature rank on training data; this template does not provide 21 independent features by construction. Use identical symmetry handling in all comparisons.

H(t)/hbar = Omega(t)/2 sum_i X_i - Delta_x(t) sum_i n_i + sum_i<j [C6/r_ij^6] n_i n_j.

Read C6, coordinates, amplitude/detuning ranges, timing resolution, slew limits and calibration metadata from the actual device properties at compilation. Freeze the discretized AHS program, then compile the simulation from that program. The displayed hexagon is a mathematical/physical template, not a model of benzene or a claim about carbon bonding.

Wolfram Quantum Framework and an independent numerical implementation check dimensions, Hermiticity, basis conventions and representative propagators. Check full long-range interactions, not a hard-cutoff neighbour model. Verify that the output is neither constant nor saturated across actual unlabeled input examples. Verify order-sensitive distinctions before presenting Gamma/Epsilon as distinct experimental arms. If two arms are observationally identical at the chosen readout, merge them or issue a new registered protocol version before testing.

Six atoms make this pilot classically tractable. It is a test of the predictive workflow, not a quantum speedup demonstration.

## 4. Matched NVIDIA CUDA-Q simulation

Construct a CUDA-Q time-evolution circuit from the frozen AHS coordinates and pulse schedule. Refine the product-formula time step until readout errors converge against an independent exact six-qubit reference. Proposed qualification threshold: maximum absolute excitation/pair-probability error below 0.001, with convergence evidence retained. This is a numerical threshold, not a claim of biological accuracy.

For the established qBraid path, export a supported compiled program, record its hash, and confirm that backend gate limits and basis ordering preserve the model. The NVIDIA GPU performs classical simulation. Native CUDA-Q dynamics is an optional future runtime only after its actual capabilities are checked; it is not assumed to be available through qBraid.

Run ideal and calibrated-noise simulations separately. Use the same observations, time points and shot budgets as hardware for matched sampling comparisons. Never describe a simulation as a QPU measurement. A simulated run has its own backend receipt and provenance.

## 5. Aquila execution and quantum features

Measure six single-site excitation probabilities and fifteen labelled pair probabilities at the final evolution time. Derive each from repeated shots; one bitstring is not a probability estimate. The initial planning value is 100 shots per input/encoding. Final shots and repeat count are determined by precision and budget before launch, with the decision logged. Additional observation times are separate tasks and require a new frozen specification.

Record pre/post measurement arrays, atom filling, losses, exclusions, accepted-shot counts, shot uncertainty, task ARN, timestamps, calibration and S3 receipt. Report postselection and its dependence on inputs; do not conceal discarded shots. Preserve raw bitstrings so alternative readouts remain auditable. Repeat a fixed subset across batches to assess drift. Interleave encodings and control examples, rather than running all examples of one biological class together.

## 6. Biological prediction and evaluation

Train a regularized classical readout on quantum responses. Freeze its settings using training/validation groups only. The corresponding classical methods receive the same biological information and comparable tuning resources: a motif-only baseline, a regularized linear model, a nonlinear kernel model and a tree-based model. A strong validation-selected classical method is the primary comparator; a deliberately weak linear baseline is insufficient.

Primary endpoint: difference in average precision on the frozen held-out assay-labelled benchmark, quantum response features versus the strongest classical comparator using the same inputs. Report the OXTR-specific ranking and performance only where its independent label count permits estimation. If the locus contains too few positives or comparison windows, report rankings and uncertainty without an OXTR performance claim.

Separately assess the contribution of Fibonacci/M9 P24 against ordinary and matched shuffled encodings. A quantum-versus-classical gain and a Fibonacci-versus-control gain are distinct findings. Neither is inferred from CUDA-Q/Aquila agreement alone.

Use grouped/block bootstrap intervals. Keep measurement shots nested within their biological example; do not inflate sample size with repeated shots, overlapping windows or donors. Correct the prespecified representation comparisons for multiplicity using Holm's method. Preserve negative and inconclusive outcomes. Report calibration and precision at fixed recall as secondary outcomes. Label permutations test the full training/evaluation pipeline, not just the final readout.

Sample size is fixed after the assay audit and a training-only pilot variance/power assessment. No claim of adequate power is made now. Hardware qualification on a few windows is reported as qualification only, not as the biological test. A limited budget narrows the registered question or the confidence of its answer; it never licenses selective reporting.

## 7. Shared compute, persistence and status

Number of tasks = input windows × encoding variants × readout times × repeated hardware batches. Record the provider quote and maximum total spend before approval. Reserve shared daily, weekly and monthly budget atomically; reconcile actual receipts and release unused reservations. Trial credits and funded hardware have separate ledgers. No per-researcher approval is planned. The earlier one-task Aquila approval is not reused for this study.

Execution states: design review → data admission → model validation → simulation → hardware queued/running/completed or failed → evaluation → complete. Simulation and hardware can have separate concurrent states. Display a result only after validation and evaluation succeed. A failed or stale provider status remains visible. Use idempotency keys so retries never silently duplicate paid submissions.

Cache keys include template version, data/split/feature hashes, normalization, encoding, physical-program hash, backend version, shots, seeds and calibration date. Saved answers retain their original run date and receipt. A new analysis version points to its parent rather than overwriting its results. Successful examples save the request-to-specification mapping, concise design rationale and checks; they never replace deterministic compilation.

## 8. Result presentation

Lead with the biological finding and uncertainty. The interactive result combines the original DNA positions, chosen representation, simulated response, hardware response and experimental occupancy layer. Every layer has its own provenance and visibility control. Show differences and uncertainty rather than a decorative confidence score. No fabricated result points are displayed before execution.

Methods and verification expand to source accessions, frozen JSON, Wolfram checks, CUDA-Q convergence, provider task receipts, readout rules and statistical evaluation. The natural-language interface explains these records without requiring visitors to operate the quantum platform themselves.

## 9. Research basis

- QuEra: QRC tutorials — https://github.com/QuEraComputing/QRC-tutorials
- Kornjača et al., Large-scale quantum reservoir learning with an analog quantum computer — https://arxiv.org/abs/2407.02553 (preprint; hardware demonstrations, no OXTR result).
- Giusto et al., Harnessing a 256-qubit Neutral Atom Simulator for Graph Classification — https://arxiv.org/abs/2605.04737 (preprint; protein graph classification, comparable selected benchmark).
- Antoncich et al., Quantum Reservoir Computing with Neutral Atoms on a Small, Complex, Medical Dataset — https://arxiv.org/abs/2602.14641 (preprint; overfitting and feature-selection leakage limitations inform this protocol).
- Amazon Braket Aquila specification — https://docs.aws.amazon.com/braket/latest/developerguide/braket-quera-submitting-analog-program-aquila.html

## Decisions required before execution

1. Admit the factor/cell assay and freeze its accessions, labels, context and split.
2. Freeze exact feature transformations, canonical weights, shuffled controls and numerical qualification results.
3. Register sample size, shots, outcome metrics and total compute quote.
4. Execute the matched CUDA-Q and Aquila study in the agreed joint session.

## Scientist journey and report delivery

A study runs independently of the browser after submission. Its durable record owns the specification, inputs, job IDs, queue observations, stage states, report versions and delivery preference. Revisiting or refreshing never resubmits a task. A private access link authenticates access to individual study data; public demonstration records use the shared study page.

The scientist asks a question, explores geometry in Study, receives Classical results, then receives Quantum results. Routing, model preparation and evaluation remain internal. No research-plan conversation or preparation checklist is presented as the response. Publish a clearly labelled simulation report after classical comparison and CUDA-Q processing; publish the combined report after Aquila results and evaluation. Never present the simulation report as hardware-validated. A hardware failure leaves the earlier report accessible and offers an explicit retry with a new cost check.

Display the requested planning estimates directly: Classical results approximately 5–10 minutes; Quantum results approximately 24 hours or longer, measured from processing start. Mark them as estimates. Actual workload, device windows and queues can extend the wait. Record elapsed timings from the first study run to calibrate later estimates. Queue position is not a clock estimate.

Optional email is confirmed at execution launch using a verified address and consent specific to study updates. Send a link, with no genomic data in the email. Record accepted, delivered and failed notification states, support opt-out, and prevent duplicate sends. No email is sent during design review. The review interface saves an optional delivery preference and an explicitly retained study record on this device only.

Timing source: [Amazon Braket task queue and availability guidance](https://docs.aws.amazon.com/braket/latest/developerguide/braket-task-when.html).

## Stage 5 roadmap

Drug Discovery Network & Open Source Genomic Insights is a future feature. Recorded results and preliminary LLM insights pass through human review of evidence, permissions and sharing scope before collaboration with external platforms. The long-term aim is automated collaboration with human oversight and reviewed, shareable research outputs. No external submission, partnership or automated publication is represented as active.
