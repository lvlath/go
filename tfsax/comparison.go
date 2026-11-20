// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax compares only implemented exact and SAX ranking retrieval branches.
// This file records genuinely measured per-query work without fabricating the TFSAX arm.
package tfsax

import (
	"context"
	"time"
)

const maxWarmupRuns = 4 // Explicit cap prevents unbounded unmeasured benchmark work.

// ComparisonOptions fixes the implemented SAX baseline and warmup policy.
// Every field must be precommitted before timing; no post-hoc tuning occurs here.
//
// Implementation:
//   - Validate SAX dimensions against experiment length and warmup count before allocating the index.
//
// Behavior highlights:
//   - SAX word dimensions and warmup repetitions are explicit.
//   - The DTW window/TopK/shortlist remain frozen by Experiment.Protocol.
//
// Inputs:
//   - SAXSegments: number of fractional PAA cells, 1..Protocol.Length.
//   - SAXAlphabet: Gaussian symbol bins, 2..256.
//   - WarmupRuns: 0..4 unrecorded exact+SAX query repetitions.
//
// Returns:
//   - A value selecting one pinned Gaussian SAX baseline for the two-arm comparison.
//
// Errors:
//   - ErrInvalidOption on invalid policy in RunAvailableComparison.
//
// Determinism:
//   - Policies are fixed; warmup execution changes timings, not semantic results.
//
// Complexity:
//   - O(1) value storage; warmups multiply search work by 1+WarmupRuns.
//
// Notes:
//   - A different SAX alphabet or segment count changes retrieval ranking and must be selected before holdout scoring.
//
// AI-Hints:
//   - Precommit this struct; benchmarking multiple configs on the test set is tuning leakage.
type ComparisonOptions struct {
	SAXSegments int // Fixed PAA segment count of the ordinary SAX comparator.
	SAXAlphabet int // Pinned Gaussian alphabet size.
	WarmupRuns  int // Number of untimed warmups per query and arm, <=4.
}

// DefaultComparisonOptions returns a fixed, readable baseline configuration.
// It deliberately exposes the existing standard eight-segment/eight-bin SAX choice.
//
// Implementation:
//   - Construct the eight-segment, eight-bin Gaussian baseline with no warmup passes.
//
// Behavior highlights:
//   - The result is a value that callers can alter explicitly before validation.
//
// Inputs:
//   - No arguments: the source length remains unknown and must be admitted by the comparison facade.
//
// Returns:
//   - Named options value without hidden corpus-dependent tuning.
//
// Errors:
//   - None; calling comparison with length<8 rejects the incompatible option.
//
// Determinism:
//   - Pure constant construction.
//
// Complexity:
//   - O(1) time and memory.
//
// Notes:
//   - An experiment with length below eight cannot use this default unchanged.
//
// AI-Hints:
//   - Use an explicitly smaller segment count for shorter windows; do not silently clamp.
func DefaultComparisonOptions() ComparisonOptions {
	return ComparisonOptions{SAXSegments: defaultSAXSegments, SAXAlphabet: defaultSAXAlphabet}
}

// ComparisonResult is an auditable, detached two-arm diagnostic result artifact.
// It does not claim the four-arm research gate or certify any production promotion.
//
// Implementation:
//   - The runner prepares one index, measures two real arms for each query, then publishes an owned observation collection.
//
// Behavior highlights:
//   - Observations contain two entries per validated query: exact, then SAX ranking.
//   - CorpusDigest hashes ordered candidate slots; QueriesDigest hashes stable query IDs.
//   - Neither digest includes nondeterministic timing, and neither authenticates the importer.
//   - SAXIndexBytes counts ID+symbol payload only and is not measured Go resident memory.
//
// Inputs:
//   - Observations are produced by RunAvailableComparison with stable query identities.
//
// Returns:
//   - Caller-owned slice and nested result-ID slices, detached from input corpus.
//
// Errors:
//   - The runner returns nil rather than this artifact when validation, alignment or reporting preparation fails.
//
// Determinism:
//   - Branches/order/results are stable; elapsed durations are not bitwise reproducible.
//
// Complexity:
//   - Space O(number of queries * (TopK+1)) excluding index and scratch buffers.
//
// Notes:
//   - Complete is procedural success, not certification of any TFSAX representation.
//
// AI-Hints:
//   - Do not use these two arms to calculate TFSAX speedup or memory acceptance ratios.
type ComparisonResult struct {
	Observations  []Observation // Detached exact and SAX observations in stable order.
	CorpusDigest  string        // Input ordered-corpus identity, validated before execution.
	QueriesDigest string        // Ascending-query-ID identity, excludes wall-clock measurements.
	SAXIndexBytes uint64        // Serialized ID+symbol payload only, excluding backing raw corpus.
	SAXBuildNS    int64         // Measured index construction time, excluded from query latency.
	Complete      bool          // Both implemented branches completed for every requested query.
}

// IsNil checks the ComparisonResult pointer without dereferencing it.
// A typed nil interface result is recognized, while a zero value is not nil.
//
// Implementation:
//   - Compare the receiver pointer with nil without examining its fields.
//
// Behavior highlights:
//   - A completed or zero-value non-nil comparison is never treated as absent.
//
// Inputs:
//   - Receiver result: a possibly typed-nil *ComparisonResult.
//
// Returns:
//   - bool: true only if the receiver is nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure pointer-identity check.
//
// Complexity:
//   - Time O(1), space O(1).
//
// Notes:
//   - Use Complete to inspect completed computation; IsNil does not certify its observations.
//
// AI-Hints:
//   - Complete=false is not equivalent to an absent ComparisonResult pointer.
func (result *ComparisonResult) IsNil() bool { return result == nil }

// RunAvailableComparison measures exact full scan against cached standard SAX ranking.
// It is a diagnostic two-arm comparison; TFSAX and certified LB pruning remain absent.
//
// Implementation:
//   - Stage 1: validate all input queries, freeze order and explicit SAX policy.
//   - Stage 2: budget both search arms and all warmups; compute the query universe digest.
//   - Stage 3: build a private SAXIndex before timing; optionally warm both kernels.
//   - Stage 4: time exact and SAX search independently, then calculate strict-ID metrics.
//   - Stage 5: publish complete detached observations only if every query succeeds.
//
// Behavior highlights:
//   - Exact result is the oracle for both branches; TFSAX is not simulated.
//   - The prepared SAX index excludes build time from per-query SAX latency.
//   - Elapsed times are noisy and cannot establish speedup from one measurement each.
//
// Inputs:
//   - ctx: non-nil active context, checked before work and between DTW refinements.
//   - experiment: frozen protocol, corpus digest, queries and definition metadata.
//   - options: explicitly preselected SAX dimensions and warmup count.
//
// Returns:
//   - *ComparisonResult with detached nested result slices, or nil on any failure.
//
// Errors:
//   - ErrInvalidInput for unfrozen corpus or invalid queries; ErrInvalidOption for options.
//   - ErrBudget when Q*(N+TopM)*L^2*(1+warmups) exceeds the frozen cell limit.
//   - DTW/SAX error identities and ctx.Err propagate.
//
// Determinism:
//   - Queries sorted by numeric ID; each query reports exact first, SAX second.
//   - Candidate order uses (DTW distance, ID); only timing and OS allocations vary.
//
// Complexity:
//   - Build O(N*L), then O((warmup+1)*Q*(N+TopM)*L^2 + Q*N log N),
//   - Space O(N*(L+segments) + Q*TopK) plus per-search scratch and DTW rolling rows.
//
// Notes:
//   - Temporal provenance must be separately validated via ValidateWindowRecords.
//   - Raw observations intentionally omit production admission and resident-memory claims.
//
// AI-Hints:
//   - Never compare a cold SAX build against a warmed exact scan as a query speedup.
//   - A two-arm diagnostic cannot satisfy the four-arm GateReport definition.
func RunAvailableComparison(ctx context.Context, experiment Experiment, options ComparisonOptions) (*ComparisonResult, error) {
	queries, err := validateExperiment(ctx, experiment, true)
	if err != nil {
		return nil, err
	}
	if options.SAXSegments < 1 || options.SAXSegments > experiment.Protocol.Length ||
		options.SAXAlphabet < 2 || options.SAXAlphabet > 256 ||
		options.WarmupRuns < 0 || options.WarmupRuns > maxWarmupRuns {
		return nil, ErrInvalidOption
	}
	// A two-arm comparison uses a full exact scan AND an exact-refined shortlist.
	// Warmups repeat both. The protocol's minimal one-arm budget is not enough
	// whenever these additional kernels are enabled; refuse before building an index.
	if !withinCellBudget(experiment.Protocol.MaxCells,
		options.WarmupRuns+1, experiment.Protocol.QueryCount,
		experiment.Protocol.CorpusSize+experiment.Protocol.Shortlist,
		experiment.Protocol.Length, experiment.Protocol.Length) {
		return nil, ErrBudget
	}

	// Commit the exact query universe before any measurement. Sorting by stable ID
	// makes this digest invariant to caller slice order, unlike the ordered corpus hash.
	queryDigest, err := QueriesHash(queries)
	if err != nil {
		return nil, err
	}

	// Keep expensive model fitting/encoding outside per-query timings.
	buildStart := time.Now()
	index, err := PrepareSAXIndex(experiment.Corpus, options.SAXSegments, options.SAXAlphabet)
	if err != nil {
		return nil, err
	}
	buildDuration := time.Since(buildStart).Nanoseconds()
	observations := make([]Observation, 0, len(queries)*2)
	for _, named := range queries {
		for warmup := 0; warmup < options.WarmupRuns; warmup++ {
			if _, _, err := scanExactValidated(ctx, experiment.Corpus, named.Query, experiment.Protocol.Window); err != nil {
				return nil, err
			}
			if _, err := index.searchValidated(ctx, named.Query, experiment.Protocol.Window, experiment.Protocol.Shortlist); err != nil {
				return nil, err
			}
		}

		// Timed methods consume already validated same-length inputs and identical DTW policy.
		exactStart := time.Now()
		exact, available, err := scanExactValidated(ctx, experiment.Corpus, named.Query, experiment.Protocol.Window)
		if err != nil {
			return nil, err
		}
		exactDuration := time.Since(exactStart).Nanoseconds()
		saxStart := time.Now()
		approximate, err := index.searchValidated(ctx, named.Query, experiment.Protocol.Window, experiment.Protocol.Shortlist)
		if err != nil {
			return nil, err
		}
		saxDuration := time.Since(saxStart).Nanoseconds()

		// Score all experimental selections against the same unmodified exact TopK.
		observations = append(observations,
			Observation{QueryID: named.ID, Family: named.Family, Branch: "exact", Result: *exact,
				Recall:     RecallAtK(exact.Candidates, exact.Candidates),
				MissedBest: MissedBest(exact.Candidates, exact.Candidates),
				Available:  available, Shortlist: len(experiment.Corpus), DurationNS: exactDuration},
			Observation{QueryID: named.ID, Family: named.Family, Branch: "sax-ranking", Result: *approximate,
				Recall:     RecallAtK(approximate.Candidates, exact.Candidates),
				MissedBest: MissedBest(approximate.Candidates, exact.Candidates),
				Available:  available, Shortlist: experiment.Protocol.Shortlist,
				DurationNS: saxDuration, IndexBytes: index.IndexBytes()},
		)
	}
	return &ComparisonResult{
		Observations: observations, CorpusDigest: index.CorpusDigest(), QueriesDigest: queryDigest,
		SAXIndexBytes: index.IndexBytes(), SAXBuildNS: buildDuration, Complete: true,
	}, nil
}

// RunAvailableDefault is an explicit convenience wrapper for a fixed Gaussian SAX baseline.
// It delegates to RunAvailableComparison and never substitutes a missing TFSAX arm.
//
// Implementation:
//   - Obtain DefaultComparisonOptions, then delegate without reimplementing the comparison kernel.
//
// Behavior highlights:
//   - No warmup is requested; SAX build remains separate from query timing.
//
// Inputs:
//   - ctx: non-nil active context.
//   - experiment: frozen input whose length must be at least eight samples.
//
// Returns:
//   - Detached two-arm *ComparisonResult or nil with delegated error.
//
// Errors:
//   - Exactly the same classes as RunAvailableComparison.
//
// Determinism:
//   - Same stable ordering with no hidden tuning or additional randomness.
//
// Complexity:
//   - Same as RunAvailableComparison with eight segments/bins and zero warmup.
//
// Notes:
//   - For non-default length, segments or alphabet use RunAvailableComparison with an explicit policy.
//
// AI-Hints:
//   - Change options explicitly to benchmark other SAX configurations predeclared in a plan.
func RunAvailableDefault(ctx context.Context, experiment Experiment) (*ComparisonResult, error) {
	return RunAvailableComparison(ctx, experiment, DefaultComparisonOptions())
}
