// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax implements a reproducible research gate for time-series retrieval.
// It compares the accepted exact scalar DTW oracle with an explicitly heuristic
// classical SAX shortlist; it does not implement a mathematically undefined
// TFSAX encoder, a TFSAX distance, or certified symbolic DTW pruning.
//
// Domain and public result contract:
//   - A Candidate is one non-nil finite scalar window with a unique uint64 ID.
//     Candidates within a corpus have equal positive length; Query.Values has
//     that same length and Query.TopK is positive.
//   - ExactDTW delegates its fixed-endpoint recurrence to dtw.Align using
//     dtw.AbsoluteCost (|a-b|) and an explicit Sakoe-Chiba window. An
//     overflow-aware local callback preserves finite-input arithmetic errors
//     without changing the cost formula or evaluating off-band pairs. A
//     window of -1 is unrestricted; 0 permits only diagonal alignment.
//   - ScanExact evaluates every corpus candidate. Reachable hits are ranked
//     by (exact DTW distance, numeric CandidateID), both ascending. TopK may
//     exceed the number of reachable candidates without fabricating hits.
//   - The prepared SAXIndex uses the existing fractional PAA / Gaussian SAX
//     encoder, with per-window population Z normalization and explicit
//     constant-window ZeroWord policy. Its MINDIST scores rank candidates;
//     only the selected TopM receive exact-DTW refinement.
//   - RetrievalResult.Complete reports procedural completion of the requested
//     search. Diagnostics.Exact separately reports a mathematical guarantee
//     relative to the complete exact-DTW TopK target. SAX and custom Ranker
//     shortlists always publish Exact=false, including observed 100% recall.
//   - (*RetrievalResult).IsNil checks pointer nilness, including typed nil
//     interfaces. It does not inspect completion, candidate count or reachability.
//
// Research execution boundaries:
//   - Run is a strict four-arm admission gate. Until a separately specified
//     TFSAX encoder and certified DTW pruning branch exist, it returns
//     ErrUndefinedRepresentation and ErrUncertifiedBound, not fake findings.
//   - RunExactReference runs only the exhaustive oracle; RunAvailableComparison
//     measures exactly two real arms (exhaustive DTW and SAX ranking with DTW
//     refinement). These diagnostics never imply that the research hypothesis
//     passed its empirical or mathematical acceptance tests.
//   - ScanLBConservative refuses to prune. LBKeoghAbsolute is a diagnostic
//     computed in float64; the implementation does not provide a rounding-safe
//     rejection certificate for the accepted DTW kernel.
//   - All branches compare identical original-valued scalar windows using the
//     same exact oracle, cost function, band and deterministic candidate ties.
//     Neither SAX MINDIST nor any user Ranker score is an arbitrary-DTW bound.
//
// Research protocol and data provenance:
//   - Protocol, Definition, Thresholds, SplitPlan and Experiment are ordinary
//     Go values; no JSON input file is required to build or execute research.
//   - DefaultProtocol and UnresolvedDefinition build these records directly.
//     ValidateProtocol and ValidateDefinition reject invalid policy before
//     indexing. The schema labels identify audit records, not implementations.
//   - The optional ParseProtocol, DigestJSON and WriteJSON helpers use only
//     Go's encoding/json standard library. JSON struct tags pin exported
//     field names for reproducible files and digests; they do not affect
//     in-memory Go algorithms or introduce third-party dependencies.
//   - When results must be independently reproduced across processes, retain
//     the frozen plan, scientific definition status, candidate/query digests,
//     generator/RNG versions and evaluation code. Merely supplying the same
//     seed is not sufficient evidence of an identical experiment.
//   - SplitPlan must be fixed on the raw time axis BEFORE sliding windows are
//     materialized. ValidateSplit enforces Train -> Calibration -> Test order,
//     non-overlap and a conservative gap of purge + horizon + embargo with
//     purge >= support width. FitIDs must belong to TrainIDs only.
//   - ValidateWindowRecords additionally checks every declared window's full
//     support and label endpoint against its claimed raw partition. These
//     checks do not cryptographically authenticate a dishonest data importer;
//     raw timestamps and fit provenance remain application responsibilities.
//
// Numeric, error and ownership laws:
//   - NaN and both infinities are invalid input samples (ErrNonFinite). An
//     unrepresentable absolute difference between finite samples, or a
//     terminal accumulated optimum outside finite float64, returns ErrOverflow
//     with dtw.ErrOverflow preserved via errors.Is. The upstream numerical
//     dtw.ErrNaNInf may accompany overflow without implying invalid input.
//     Structural no-path is distinct from either arithmetic error.
//   - Empty, duplicate, inconsistent-shape and nonfinite inputs are rejected
//     by sentinel-classified errors, matched through errors.Is. All public
//     context APIs reject nil contexts and honor ctx.Err when canceled.
//   - A failed atomic search/experiment publishes no usable partial result.
//     Every returned Candidates slice is detached; prepared SAX indexes own
//     private copies of corpus data and words. Input mutation during execution
//     is unsupported. An immutable index permits concurrent read-only Search
//     calls with independent stable queries and per-call scratch storage.
//   - rng.RNG streams and other caller-maintained mutable state belong to one
//     goroutine. The package uses no reflect, panic/recover, global PRNG or
//     hidden fitted test-data preprocessing.
//
// Empirical measurement law:
//   - RecallAtK compares unique exact-target and returned CandidateIDs. Empty
//     exact truth is not applicable (NA), not a synthetic perfect match.
//     MissedBest tests the exact first ID; SummarizeByFamily retains worst
//     applicable recall and family-specific missed-best rate.
//   - IndexBytes is an ID-plus-symbol payload estimate, excluding private raw
//     corpus retention, Go object overhead and process RSS. It is never a
//     sufficient measurement of the memory-ratio research acceptance gate.
//   - Timed comparisons exclude SAX index build from per-query latency and
//     record build time separately. Latency, allocations, OS/toolchain/hardware,
//     index memory and per-family stress must be measured and reported without
//     cherry-picking. Repeated runs do not promise bit-identical nanoseconds.
//   - Exact full scan over N equal L-sample candidates takes O(N*L^2 + N log N)
//     time and O(N+L) search workspace. Prepared SAX construction takes
//     O(N*(L+w*log(a))) time with O(N*(L+w)+a) retained storage; query search
//     takes O(N*w + N log N + M*L^2 + M log M) time, where w is segments,
//     a is alphabet size and M is the shortlist refinement count.
//
// Non-goals:
//   - No scientific definition is inferred from the letters "TFSAX", a JSON
//     "approved" value, a passing small-space counterexample search, or a
//     favorable synthetic benchmark. The gate requires independent definition,
//     proof and empirical evidence before any separate production design.
//   - No trading signal, telemetry alarm threshold, fitted model, private
//     corpus, calendar resampling, missing-value repair or adaptive tuning is
//     provided by this package. Applications own those domain policies.
//
// AI-Hints:
//   - Never replace DTW with Euclidean distance to make a benchmark faster.
//   - Never promote a heuristic MINDIST shortlist to Exact=true based on recall.
//   - Never fit breakpoints/scales on a heldout query or after reading Test.
//   - Do not include wall-clock measurements in semantic hashes or compare
//     cold index-building latency with a warmed exact-search query.
//   - For Go-only in-memory research, construct Protocol and Definition values
//     directly; export JSON/CSV only when another process must audit the run.
package tfsax
