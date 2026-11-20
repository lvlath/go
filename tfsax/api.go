// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax defines the canonical research-gate facade and honest comparison wrappers.
// This file validates experiments and delegates only to implemented exact and SAX kernels.
package tfsax

import (
	"context"
	"errors"
	"sort"
)

// Experiment freezes the scalar universe and query collection for one research run.
// It is an experiment carrier, not a trainable symbolic encoder or production model.
//
// Implementation:
//   - Stage 1: validate Protocol, Definition, corpus digest and all query identities.
//   - Stage 2: order queries by numeric stable ID before any branch is evaluated.
//   - Stage 3: admit only existing benchmark branches; undefined TFSAX remains blocked.
//
// Behavior highlights:
//   - CorpusDigest is a trusted external SHA-256 commitment to ordered IDs and values.
//
// Inputs:
//   - Protocol: fixed exact-DTW and resource policy.
//   - Definition: unresolved or externally proposed mathematical metadata.
//   - Corpus: caller-owned read-only uniform-length candidate universe.
//   - Queries: read-only samples and stable IDs, not derived automatically from corpus.
//   - CorpusDigest: nonempty for a frozen benchmark; empty accepted for diagnostic exact scan.
//
// Returns:
//   - No input buffers are mutated or retained by returned observation artifacts.
//
// Errors:
//   - ErrInvalidOption, ErrBudget, ErrInvalidInput, ErrDuplicateID, ErrLeakage or ErrNonFinite.
//
// Determinism:
//   - Queries are evaluated in ascending ID; candidates use exact-distance then ID ties.
//
// Complexity:
//   - Validation O(N*L+Q*L+Q log Q) time and O(N+Q) working memory.
//
// Notes:
//   - This value alone cannot prove physical train/test provenance; use ValidateWindowRecords.
//
// AI-Hints:
//   - Freeze corpus and query metadata before comparison; do not reconstruct splits from windows.
type Experiment struct {
	Protocol     Protocol     // Required versioned exact/search/budget policy.
	Definition   Definition   // Explicit research status; never auto-enables a missing codec.
	Corpus       []Candidate  // Ordered read-only candidate universe.
	Queries      []NamedQuery // Stable-ID read-only query collection.
	CorpusDigest string       // Trusted corpus SHA-256 identity; required for comparison.
}

// validateExperiment admits the entire input before any timed or partial computation.
// It produces a shallow copy of query records sorted by ID; their input samples remain borrowed.
func validateExperiment(ctx context.Context, experiment Experiment, requireDigest bool) ([]NamedQuery, error) {
	if ctx == nil {
		return nil, ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := ValidateProtocol(experiment.Protocol); err != nil {
		return nil, err
	}
	if requireDigest || experiment.Definition.Schema != "" {
		if err := ValidateDefinition(experiment.Definition); err != nil {
			return nil, err
		}
	}
	if requireDigest && experiment.CorpusDigest == "" {
		return nil, ErrInvalidInput
	}
	if err := ValidateCorpus(experiment.Corpus, experiment.CorpusDigest); err != nil {
		return nil, err
	}
	if len(experiment.Corpus) != experiment.Protocol.CorpusSize ||
		len(experiment.Queries) != experiment.Protocol.QueryCount ||
		len(experiment.Corpus[0].Values) != experiment.Protocol.Length {
		return nil, ErrInvalidInput
	}
	queries := append([]NamedQuery(nil), experiment.Queries...)
	sort.Slice(queries, func(i, j int) bool { return queries[i].ID < queries[j].ID })
	for index, named := range queries {
		if index > 0 && named.ID == queries[index-1].ID {
			return nil, ErrDuplicateID
		}
		if !knownFamily(named.Family) {
			return nil, ErrInvalidInput
		}
		if requireDigest && named.Query.TopK != experiment.Protocol.TopK {
			return nil, ErrInvalidInput
		}
		if err := ValidateQuery(named.Query, experiment.Protocol.Length); err != nil {
			return nil, err
		}
	}
	return queries, nil
}

// Run is the strict four-branch research admission entry point.
// Without a reviewed TFSAX kernel and certified exact-pruning branch it refuses execution.
//
// Implementation:
//   - Stage 1: validate the entire frozen experiment before any branch could run.
//   - Stage 2: reject absent approved mathematical representation and missing executable codec.
//   - Stage 3: publish no partial or invented observations.
//
// Behavior highlights:
//   - Approved=true in Definition metadata cannot authorize an unimplemented encoder.
//
// Inputs:
//   - ctx: non-nil active context, checked before scientific dispatch.
//   - experiment: frozen value with a valid protocol, corpus and queries.
//
// Returns:
//   - nil observations and a gate error; there is intentionally no production four-way mode.
//
// Errors:
//   - Input, budget or context errors from validation; ErrUndefinedRepresentation from the gate.
//
// Determinism:
//   - Always fails with the same gate identity for an otherwise valid experiment.
//
// Complexity:
//   - O(N*L + Q*L + Q log Q) admission time; O(N+Q) validation memory.
//
// Notes:
//   - Use RunExactReference or RunAvailableComparison for *clearly labeled* partial research.
//
// AI-Hints:
//   - Do not replace the unavailable arms with fake scores or label a two-arm run four-way.
func Run(ctx context.Context, experiment Experiment) ([]Observation, error) {
	if _, err := validateExperiment(ctx, experiment, true); err != nil {
		return nil, err
	}
	return nil, errors.Join(ErrUndefinedRepresentation, ErrUncertifiedBound)
}

// RunExactReference runs only the exact full-scan branch for every frozen query.
// It provides a deterministic truth baseline without advertising missing research arms.
//
// Implementation:
//   - Stage 1: validate the entire experiment, including all queries, before scanning.
//   - Stage 2: evaluate queries by ascending ID using the shared exact-DTW kernel.
//   - Stage 3: attach strict-ID recall, missed-best and actual reachable counts.
//
// Behavior highlights:
//   - Every observation's Exact and Complete flags are true after a full successful scan.
//   - DurationNS remains zero because this facade is a semantic oracle, not timed benchmark.
//
// Inputs:
//   - ctx: non-nil active context; cancellation checked between each exact candidate.
//   - experiment: complete diagnostic experiment; an empty digest disables digest checking.
//
// Returns:
//   - Detached []Observation in ascending query ID order, or nil on failure.
//
// Errors:
//   - Validation and dtw error identities preserved; no usable partial observations on error.
//
// Determinism:
//   - Query numeric ID order; candidate score then numeric ID ordering.
//
// Complexity:
//   - Time O(Q*(N*L^2+N log N)), space O(N+Q*TopK+Q log Q).
//
// Notes:
//   - This one-arm diagnostic does not evaluate empirical TFSAX success criteria.
//
// AI-Hints:
//   - Available means all reachable candidates, not the length of the returned TopK prefix.
func RunExactReference(ctx context.Context, experiment Experiment) ([]Observation, error) {
	queries, err := validateExperiment(ctx, experiment, false)
	if err != nil {
		return nil, err
	}
	observations := make([]Observation, 0, len(queries))
	for _, named := range queries {
		result, available, err := scanExactValidated(ctx, experiment.Corpus, named.Query, experiment.Protocol.Window)
		if err != nil {
			return nil, err
		}
		observations = append(observations, Observation{
			QueryID: named.ID, Family: named.Family, Branch: "exact",
			Result: *result, Recall: RecallAtK(result.Candidates, result.Candidates),
			MissedBest: MissedBest(result.Candidates, result.Candidates),
			Available:  available, Shortlist: len(experiment.Corpus),
		})
	}
	return observations, nil
}

// ScanExactUnbounded is a convenience wrapper for unrestricted scalar DTW search.
// It delegates to ScanExact with window=-1, without introducing a second kernel.
//
// Implementation:
//   - Validate via ScanExact, set the explicit unrestricted sentinel (-1), then return its detached result.
//
// Behavior highlights:
//   - This wrapper does not introduce per-window normalization or a second DTW recurrence.
//
// Inputs:
//   - ctx, corpus, query: exact same read-only contract as ScanExact.
//
// Returns:
//   - Detached exact *RetrievalResult or nil with delegated error.
//
// Errors:
//   - Same sentinel categories as ScanExact.
//
// Determinism:
//   - Same (distance, CandidateID) tie rule as ScanExact.
//
// Complexity:
//   - O(N*L^2+N log N) time, O(N+L) working memory.
//
// Notes:
//   - Use only when unrestricted stretching is scientifically justified; wide paths can make distinct timing regimes align.
//
// AI-Hints:
//   - This is *unbanded*, not the strict diagonal (window=0) mode.
func ScanExactUnbounded(ctx context.Context, corpus []Candidate, query Query) (*RetrievalResult, error) {
	return ScanExact(ctx, corpus, query, -1)
}

// ScanExactDiagonal evaluates exact DTW restricted to one-to-one sample alignment.
// It delegates to ScanExact with window=0 and preserves the absolute local cost.
//
// Implementation:
//   - Delegate the same checked corpus and query to ScanExact with a zero-radius band; publish its result unchanged.
//
// Behavior highlights:
//   - Diagonal-only DTW uses the same absolute local cost but permits no repeated alignment steps.
//
// Inputs:
//   - ctx, corpus, query: exact same read-only contract as ScanExact.
//
// Returns:
//   - Detached exact *RetrievalResult or nil with delegated error.
//
// Errors:
//   - Same sentinel categories as ScanExact.
//
// Determinism:
//   - Same (distance, CandidateID) tie rule as ScanExact.
//
// Complexity:
//   - Upstream DTW still traverses O(L^2) row loops; O(N*L^2+N log N) bound.
//
// Notes:
//   - The upstream implementation still performs the full quadratic row traversal, so this is a mathematical policy, not a linear-time fast path.
//
// AI-Hints:
//   - Never silently turn window=0 into no constraint; that changes the alignment.
func ScanExactDiagonal(ctx context.Context, corpus []Candidate, query Query) (*RetrievalResult, error) {
	return ScanExact(ctx, corpus, query, 0)
}
