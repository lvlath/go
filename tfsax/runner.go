// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax implements exact DTW retrieval, deterministic ranking and diagnostic bounds.
// This file is the sole canonical full-DTW refinement kernel for the research harness.
package tfsax

import (
	"context"
	"math"
	"sort"
)

const (
	exactRepresentationVersion = "exact-dtw-absolute-v1" // Absolute local cost, fixed endpoints and declared window.
)

// scored carries a candidate's ranking or exact distance and its corpus slot.
// All output sorting is through lessScore, never through map or input-order ties.
type scored struct {
	id       CandidateID
	distance float64
	index    int
}

// lessScore implements the canonical strict total order on finite nonnegative scores.
// Exact equal distances always prefer the numerically smaller CandidateID.
func lessScore(a, b scored) bool {
	if a.distance == b.distance {
		return a.id < b.id
	}
	return a.distance < b.distance
}

// sortScores sorts newly owned candidate scores without changing caller samples.
// No stable-sort dependence remains because duplicate CandidateIDs are rejected.
func sortScores(values []scored) {
	sort.Slice(values, func(i, j int) bool { return lessScore(values[i], values[j]) })
}

// chosen creates an independent prefix of sorted candidate IDs.
// TopK larger than the reachable list yields a naturally shorter detached result.
func chosen(values []scored, topK int) []CandidateID {
	if topK > len(values) {
		topK = len(values)
	}
	out := make([]CandidateID, topK)
	for index := range out {
		out[index] = values[index].id
	}
	return out
}

// validateSearchInput validates context, frozen candidate shapes, query and DTW band.
// It is shared by public scan facades before any candidate is evaluated.
func validateSearchInput(ctx context.Context, corpus []Candidate, query Query, window int) error {
	if ctx == nil {
		return ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if window < -1 {
		return ErrInvalidOption
	}
	if err := ValidateCorpus(corpus, ""); err != nil {
		return err
	}
	return ValidateQuery(query, len(corpus[0].Values))
}

// scanExactValidated traverses a prevalidated candidate universe exactly once.
// It returns total reachable members as a private reporting value, not only the TopK prefix.
func scanExactValidated(ctx context.Context, corpus []Candidate, query Query, window int) (*RetrievalResult, int, error) {
	values := make([]scored, 0, len(corpus))
	for index, candidate := range corpus {
		// Cancellation stops with no usable partial result; all entries are validated already.
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		distance, reachable, err := ExactDTW(query.Values, candidate.Values, window)
		if err != nil {
			return nil, 0, err
		}
		if !reachable {
			continue
		}
		if !finite(distance) || distance < 0 {
			return nil, 0, ErrOverflow
		}
		values = append(values, scored{id: candidate.ID, distance: distance, index: index})
	}
	reachableCount := len(values)
	sortScores(values)
	result := &RetrievalResult{
		Candidates: chosen(values, query.TopK),
		Complete:   true,
		Diagnostics: SearchDiagnostics{
			Examined: len(corpus), Exact: true,
			RepresentationVersion: exactRepresentationVersion,
		},
	}
	return result, reachableCount, nil
}

// ScanExact evaluates all corpus members with the accepted scalar DTW oracle.
// It is the mathematically exact TopK reference for the declared raw local cost and band.
//
// Implementation:
//   - Stage 1: validate context, uniform finite corpus, query and Sakoe-Chiba policy.
//   - Stage 2: evaluate full DTW once for each candidate in original corpus order.
//   - Stage 3: sort reachable distances by (distance, CandidateID) and detach TopK IDs.
//
// Behavior highlights:
//   - Complete=true even when fewer than TopK candidates exist or are reachable.
//   - Exact=true because no candidate is excluded before the exact distance evaluation.
//
// Inputs:
//   - ctx: non-nil active context; cancellation checked between candidate refinements.
//   - corpus: nonempty unique-ID uniform-length finite windows, read-only during call.
//   - query: finite same-length samples and positive TopK, read-only during call.
//   - window: -1 for full dynamic warping, >=0 for inclusive Sakoe-Chiba radius.
//
// Returns:
//   - *RetrievalResult: detached ordered IDs, or nil with error; no partial result.
//
// Errors:
//   - ErrInvalidOption for nil context/invalid band; ErrInvalidInput for invalid shape.
//   - ErrDuplicateID, ErrNonFinite and upstream DTW errors preserve their categories.
//   - ErrOverflow for an unrepresentable selected distance; ctx.Err for cancellation.
//
// Determinism:
//   - Corpus input order defines evaluation order; exact distance then numeric ID defines output.
//
// Complexity:
//   - Time O(N*L^2 + N log N), memory O(N + L) plus per-call DTW rolling rows.
//
// Notes:
//   - Calls borrow inputs but never retain them; concurrent input mutation is unsupported.
//
// AI-Hints:
//   - Never substitute Euclidean distance or an unproven symbolic score for the DTW oracle.
//   - Exact=true requires checking every universe member or having a separately certified bound.
func ScanExact(ctx context.Context, corpus []Candidate, query Query, window int) (*RetrievalResult, error) {
	if err := validateSearchInput(ctx, corpus, query, window); err != nil {
		return nil, err
	}
	result, _, err := scanExactValidated(ctx, corpus, query, window)
	return result, err
}

// Ranker evaluates a user-supplied nonnegative finite heuristic score.
// Its output orders DTW refinements but never establishes a safe DTW exclusion bound.
//
// Implementation:
//   - Called once per corpus candidate by RefineShortlist, before any DTW refinement.
//
// Behavior highlights:
//   - Returned score is used for ordering only; arbitrary positive values are not DTW lower bounds.
//
// Inputs:
//   - query, candidate: read-only finite scalar windows in original units.
//
// Returns:
//   - score >=0 and finite; error aborts the whole search without a partial result.
//
// Errors:
//   - An explicit callback error aborts search; a nonfinite score is rejected by the facade.
//
// Determinism:
//   - Caller must supply a pure deterministic callback for reproducible rankings.
//
// Complexity:
//   - Callback time and memory are caller-defined.
//
// Notes:
//   - The callback must not mutate borrowed arrays or rely on global RNG/clock state.
//
// AI-Hints:
//   - Returning a distance-like number does not turn a heuristic into a certified lower bound.
type Ranker func(query, candidate []float64) (float64, error)

// RefineShortlist ranks the entire universe heuristically and refines exactly TopM members.
// It never claims mathematical completeness from an uncertified ranking heuristic.
//
// Implementation:
//   - Stage 1: validate context, universe, TopK, budget, band and ranker identity.
//   - Stage 2: score each candidate in corpus order and sort by (heuristic, CandidateID).
//   - Stage 3: call one shared DTW-refinement kernel for the top-M ranked candidates.
//
// Behavior highlights:
//   - A candidate excluded from TopM is never evaluated with DTW.
//   - Exact=false by conservative contract, even when TopM equals corpus size.
//
// Inputs:
//   - ctx: non-nil context; cancellation checked before ranking and each DTW call.
//   - corpus: read-only finite same-length unique-ID universe.
//   - query: read-only finite same-length query, positive TopK.
//   - window: -1 unbanded, otherwise inclusive nonnegative Sakoe-Chiba radius.
//   - shortlist: TopM, integer in [1,len(corpus)].
//   - version: nonempty heuristic policy identifier supplied by the caller.
//   - rank: deterministic finite nonnegative scoring callback; not retained.
//
// Returns:
//   - Detached *RetrievalResult, or nil error without partial IDs.
//
// Errors:
//   - ErrInvalidOption for nil ranker, invalid TopM/version/band/context.
//   - ErrNonFinite for invalid callback score; callback/DTW errors are preserved.
//   - ErrOverflow for unrepresentable exact refinement; ctx.Err on cancellation.
//
// Determinism:
//   - Ranking tie order is numeric ID; refinement in rank order; output exact distance then ID.
//
// Complexity:
//   - Time O(N*rank_cost + N log N + M*L^2 + M log M).
//   - Memory O(N+M+L) excluding callback or DTW workspace.
//
// Notes:
//   - Neither the callback nor the kernel may mutate input observations.
//
// AI-Hints:
//   - Do not turn Pruned=N-M into proof of safe pruning or set Diagnostics.Exact=true.
func RefineShortlist(ctx context.Context, corpus []Candidate, query Query, window, shortlist int, version string, rank Ranker) (*RetrievalResult, error) {
	if err := validateSearchInput(ctx, corpus, query, window); err != nil {
		return nil, err
	}
	if rank == nil || shortlist < 1 || shortlist > len(corpus) || version == "" {
		return nil, ErrInvalidOption
	}
	scores := make([]scored, 0, len(corpus))
	for index, candidate := range corpus {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		score, err := rank(query.Values, candidate.Values)
		if err != nil {
			return nil, err
		}
		if !finite(score) || score < 0 {
			return nil, ErrNonFinite
		}
		scores = append(scores, scored{id: candidate.ID, distance: score, index: index})
	}
	sortScores(scores)
	return refineRanked(ctx, corpus, query, window, shortlist, version, scores)
}

// refineRanked is the sole DTW-refinement kernel for ranked heuristic candidate lists.
// Scores must be sorted by lessScore and refer to validated non-repeated corpus slots.
func refineRanked(ctx context.Context, corpus []Candidate, query Query, window, shortlist int, version string, ranked []scored) (*RetrievalResult, error) {
	refined := make([]scored, 0, shortlist)
	for _, candidate := range ranked[:shortlist] {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		distance, reachable, err := ExactDTW(query.Values, corpus[candidate.index].Values, window)
		if err != nil {
			return nil, err
		}
		if !reachable {
			continue
		}
		if !finite(distance) || distance < 0 {
			return nil, ErrOverflow
		}
		refined = append(refined, scored{id: candidate.id, distance: distance, index: candidate.index})
	}
	sortScores(refined)
	return &RetrievalResult{
		Candidates: chosen(refined, query.TopK), Complete: true,
		Diagnostics: SearchDiagnostics{Examined: shortlist, Pruned: len(corpus) - shortlist,
			Exact: false, RepresentationVersion: version},
	}, nil
}

// LBKeoghAbsolute computes a diagnostic envelope lower-bound candidate for absolute DTW.
// Its mathematical envelope law is distinct from a machine-certified floating-point pruning bound.
//
// Implementation:
//   - Stage 1: validate finite equal-length windows and a feasible radius.
//   - Stage 2: for each candidate index find query minima/maxima over its Sakoe-Chiba band.
//   - Stage 3: accumulate out-of-envelope absolute deviations and step down one ULP.
//
// Behavior highlights:
//   - For exact real arithmetic, every DTW path visits each candidate index at least once.
//   - Single-ULP correction is NOT a universally certified floating-point error bound.
//
// Inputs:
//   - query, candidate: nonempty finite read-only equal-length sequences.
//   - radius: integer in [0,len(query)-1], where n-1 represents no band restriction.
//
// Returns:
//   - Nonnegative finite diagnostic value; never a pruning certificate.
//
// Errors:
//   - ErrInvalidInput for nil, empty or nonmatching windows.
//   - ErrInvalidOption for infeasible radius; ErrNonFinite for NaN/Inf samples.
//   - ErrOverflow for subtraction or sum beyond finite float64 range.
//
// Determinism:
//   - Candidate indices ascend; each min/max scan proceeds left to right.
//
// Complexity:
//   - Time O(n*min(n,2*radius+1)), additional space O(1).
//
// Notes:
//   - Use ScanLBConservative to observe the explicit refusal of uncertified pruning.
//
// AI-Hints:
//   - One Nextafter cannot compensate arbitrary cumulative IEEE-754 rounding error.
//   - Never use this score to exclude an exact-DTW candidate.
func LBKeoghAbsolute(query, candidate []float64, radius int) (float64, error) {
	n := len(query)
	if query == nil || candidate == nil || n == 0 || len(candidate) != n {
		return 0, ErrInvalidInput
	}
	if radius < 0 || radius >= n {
		return 0, ErrInvalidOption
	}
	for index := range query {
		if !finite(query[index]) || !finite(candidate[index]) {
			return 0, ErrNonFinite
		}
	}
	var total float64
	for index, value := range candidate {
		minValue, maxValue := query[index], query[index]
		left := index - radius
		if left < 0 {
			left = 0
		}
		right := index + radius
		if right >= n {
			right = n - 1
		}
		for sampleIndex := left; sampleIndex <= right; sampleIndex++ {
			if query[sampleIndex] < minValue {
				minValue = query[sampleIndex]
			}
			if query[sampleIndex] > maxValue {
				maxValue = query[sampleIndex]
			}
		}
		deviation := 0.0
		if value < minValue {
			deviation = minValue - value
		} else if value > maxValue {
			deviation = value - maxValue
		}
		if !finite(deviation) {
			return 0, ErrOverflow
		}
		total += deviation
		if !finite(total) {
			return 0, ErrOverflow
		}
	}
	if total == 0 {
		return 0, nil
	}
	return math.Nextafter(total, math.Inf(-1)), nil
}

// ScanLBConservative refuses exact-DTW pruning without an independently audited bound.
// A diagnostic real-number inequality is insufficient to certify float64 exclusion.
//
// Implementation:
//   - Stage 1: classify nil or cancelled context before any work.
//   - Stage 2: reject the absent numeric proof and do not perform partial DTW retrieval.
//
// Behavior highlights:
//   - Refuses even when a diagnostic LB_Keogh value is available; no candidate is dropped.
//
// Inputs:
//   - ctx: non-nil active context; other arguments are unused while the mode is blocked.
//
// Returns:
//   - nil result and classified error, never a fake successful search.
//
// Errors:
//   - ErrInvalidOption for nil context; ctx.Err for cancellation.
//   - ErrUncertifiedBound while the numeric pruning proof is unavailable.
//
// Determinism:
//   - The mode is unconditionally blocked for valid contexts.
//
// Complexity:
//   - O(1) time and memory.
//
// Notes:
//   - A future version may enable this only after proof, adversarial tests and roundoff audit.
//
// AI-Hints:
//   - Never change this to return Exact=true merely because a small exhaustive test passes.
func ScanLBConservative(ctx context.Context, _ []Candidate, _ Query, _ int) (*RetrievalResult, error) {
	if ctx == nil {
		return nil, ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return nil, ErrUncertifiedBound
}
