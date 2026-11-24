// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies exact window search admission, resource and cancellation boundaries.
// Tests use only exported DTW calls, deterministic small fixtures and architecture-neutral budgets.
// AI-Hints:
//   - Do not allocate impossible slices to test internal integer-overflow guards.
//   - ExcludedStarts removes start indices, not samples or intersecting time intervals.
package dtw_test

import (
	"context"
	"math"
	"strconv"
	"testing"

	"github.com/lvlath/go/dtw"
)

// TestCoverageSearch_ValidationBoundaries verifies shape, numeric and normalization admission.
// Pre-compute failures must not publish a partial ranking, including on empty candidate domains.
// Implementation:
//   - Stage 1: construct independent invalid exclusion and query fixtures.
//   - Stage 2: invoke Search under explicit options and verify all expected sentinels.
//   - Stage 3: verify singleton population normalization rejects zero variance.
//
// Complexity: Time O(N+L+R), Space O(1) for these pre-compute failures.
// AI-Hints: A shorter data sequence is valid only with no exclusion ranges.
func TestCoverageSearch_ValidationBoundaries(t *testing.T) {
	ctx := context.Background()
	query, data := []float64{1, 2}, []float64{1, 2, 3}
	for _, tc := range []struct {
		name       string
		data       []float64
		rangeValue dtw.IndexRange
	}{
		{"short-data-with-exclusion", []float64{1}, dtw.IndexRange{Start: 0, End: 0}},
		{"negative-start", data, dtw.IndexRange{Start: -1, End: 1}},
		{"reversed-range", data, dtw.IndexRange{Start: 1, End: 0}},
		{"end-outside-domain", data, dtw.IndexRange{Start: 0, End: 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := dtw.DefaultSearchOptions()
			opts.ExcludedStarts = []dtw.IndexRange{tc.rangeValue}
			got, err := dtw.Search(ctx, query, tc.data, opts)
			mustCoverageSearchFailure(t, got, err, dtw.ErrShape)
		})
	}
	// Every query sample must be finite even when the candidate domain is empty.
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		got, err := dtw.Search(ctx, []float64{1, value}, []float64{1}, dtw.DefaultSearchOptions())
		mustCoverageSearchFailure(t, got, err, dtw.ErrNonFinite, dtw.ErrNaNInf)
	}
	// Population variance of a singleton is exactly zero, not an invalid shape.
	opts := dtw.DefaultSearchOptions()
	opts.Normalization = dtw.SearchZPopulation
	got, err := dtw.Search(ctx, []float64{1}, []float64{1, 2}, opts)
	mustCoverageSearchFailure(t, got, err, dtw.ErrZeroVariance)
}

// TestCoverageSearch_ExclusionCoalescingAndCapacityClipping verifies admission and exact ranking.
// Unsorted overlapping ranges with equal starts must be coalesced without mutating caller options.
// Implementation:
//   - Stage 1: admit two known starts from a five-start domain.
//   - Stage 2: request more workers and hits than the admitted candidate count.
//   - Stage 3: verify exact hit order, accounting and original exclusion order.
//
// Complexity: Time O(R log R+C*L) for this strict-diagonal fixture; Space O(R+C+L).
// AI-Hints: Do not compare only distances; equal-ranked hits must retain correct window coordinates.
func TestCoverageSearch_ExclusionCoalescingAndCapacityClipping(t *testing.T) {
	query := []float64{1, 2}
	data := []float64{1, 2, 9, 9, 1, 3}
	exclusions := []dtw.IndexRange{{Start: 3, End: 4}, {Start: 1, End: 3}, {Start: 1, End: 2}}
	original := append([]dtw.IndexRange(nil), exclusions...)
	opts := dtw.DefaultSearchOptions()
	opts.Radius, opts.TopK, opts.Workers = 0, math.MaxInt, math.MaxInt
	opts.ExcludedStarts = exclusions
	got, err := dtw.Search(context.Background(), query, data, opts)
	mustNoError(t, err, "coalesced exclusions")
	if got == nil || !got.Complete || got.CandidatesTotal != 2 || got.CandidatesCompleted != 2 ||
		got.CandidatesPruned != 0 || got.CellsVisited != 4 || len(got.Hits) != 2 {
		t.Fatalf("unexpected admission/accounting: %+v", got)
	}
	// Radius zero gives the ordinary sum of squared coordinate differences.
	want := []dtw.SearchHit{
		{Window: dtw.IndexRange{Start: 0, End: 2}, SquaredDistance: 0},
		{Window: dtw.IndexRange{Start: 4, End: 6}, SquaredDistance: 1},
	}
	for i := range want {
		if got.Hits[i] != want[i] {
			t.Fatalf("hit[%d]=%+v; want %+v", i, got.Hits[i], want[i])
		}
	}
	for i := range original {
		if exclusions[i] != original[i] {
			t.Fatalf("Search mutated caller exclusion[%d]: got %+v want %+v", i, exclusions[i], original[i])
		}
	}
}

// TestCoverageSearch_PreDispatchCancellation protects cancellation during preparation.
// Cancellation before validation, after workspace accounting or after query/envelope preparation
// must return no SearchResult; partial results belong only to a started search.
// Implementation:
//   - Stage 1: inject cancellation at three observations using a real cancellable context.
//   - Stage 2: call Search with a nonconstant z-normalized query and a valid candidate.
//   - Stage 3: require context.Canceled and nil result.
//
// Complexity: Time O(L+N), Space O(L) before worker dispatch.
// Notes: Observation triggers target the current checkpoints without asserting a public call count.
// AI-Hints: Avoid wall-clock cancellation races and never invoke testing.Fatal inside a worker.
func TestCoverageSearch_PreDispatchCancellation(t *testing.T) {
	for _, checkpoint := range []int32{1, 2, 3} {
		parent, cancel := context.WithCancel(context.Background())
		ctx := &coverageCheckpointContext{Context: parent, cancel: cancel, cancelAt: checkpoint}
		opts := dtw.DefaultSearchOptions()
		opts.Normalization = dtw.SearchZPopulation
		got, err := dtw.Search(ctx, []float64{1, 2}, []float64{3, 4}, opts)
		cancel()
		mustCoverageSearchFailure(t, got, err, context.Canceled)
	}
}

// TestCoverageSearch_WorkspaceBudgetBoundary checks the exact admission boundary without ABI assumptions.
// A raw search and its normalized counterpart have different required workspace; optional pruning
// storage may be omitted but required buffers cannot be silently dropped.
// Implementation:
//   - Stage 1: check that one byte cannot pay for a copied nonempty exclusion slice.
//   - Stage 2: find the minimum accepted raw workspace using monotone budget bisection.
//   - Stage 3: check the adjacent failing budget, exact result and normalized-buffer requirement.
//
// Determinism: a single candidate and one worker avoid scheduling-dependent counters.
// Complexity: O(log B * L) time for strict-diagonal runs, O(L) per run; B is the test budget ceiling.
// AI-Hints: Do not hardcode unsafe.Sizeof(privateType) or assume a 64-bit ABI.
func TestCoverageSearch_WorkspaceBudgetBoundary(t *testing.T) {
	ctx := context.Background()
	query, data := []float64{1, 2, 3, 4}, []float64{2, 3, 4, 5}
	opts := dtw.DefaultSearchOptions()
	opts.Radius, opts.MaxWorkingBytes = 0, 1
	// Even an empty half-open exclusion range occupies input-copy storage.
	opts.ExcludedStarts = []dtw.IndexRange{{Start: 0, End: 0}}
	got, err := dtw.Search(ctx, query, data, opts)
	mustCoverageSearchFailure(t, got, err, dtw.ErrBudget)
	opts.ExcludedStarts = nil

	// Establish a valid upper bound and a known failing lower bound before bisection.
	const budgetCeiling uint64 = 1 << 20
	opts.MaxWorkingBytes = budgetCeiling
	got, err = dtw.Search(ctx, query, data, opts)
	mustNoError(t, err, "workspace ceiling")
	if got == nil || !got.Complete || len(got.Hits) != 1 || got.Hits[0].SquaredDistance != 4 {
		t.Fatalf("unexpected unrestricted reference: %+v", got)
	}
	lo, hi := uint64(1), budgetCeiling
	opts.MaxWorkingBytes = lo
	got, err = dtw.Search(ctx, query, data, opts)
	mustCoverageSearchFailure(t, got, err, dtw.ErrBudget)
	for hi-lo > 1 {
		mid := lo + (hi-lo)/2
		opts.MaxWorkingBytes = mid
		got, err = dtw.Search(ctx, query, data, opts)
		if err != nil {
			mustCoverageSearchFailure(t, got, err, dtw.ErrBudget)
			lo = mid
			continue
		}
		if got == nil || !got.Complete || len(got.Hits) != 1 || got.Hits[0].SquaredDistance != 4 {
			t.Fatalf("budget=%d changed the exact result: %+v", mid, got)
		}
		hi = mid
	}
	// The adjacent budgets must differ only in admission, never in mathematical semantics.
	opts.MaxWorkingBytes = hi - 1
	got, err = dtw.Search(ctx, query, data, opts)
	mustCoverageSearchFailure(t, got, err, dtw.ErrBudget)
	opts.MaxWorkingBytes = hi
	got, err = dtw.Search(ctx, query, data, opts)
	mustNoError(t, err, "minimum raw workspace")
	if got == nil || !got.Complete || len(got.Hits) != 1 || got.Hits[0].SquaredDistance != 4 {
		t.Fatalf("minimum raw budget changed exact result: %+v", got)
	}
	// Normalization needs a detached query and a candidate buffer in addition to raw workspace.
	opts.Normalization = dtw.SearchZPopulation
	got, err = dtw.Search(ctx, query, data, opts)
	mustCoverageSearchFailure(t, got, err, dtw.ErrBudget)
	opts.MaxWorkingBytes = budgetCeiling
	got, err = dtw.Search(ctx, query, data, opts)
	mustNoError(t, err, "normalized workspace")
	if got == nil || !got.Complete || len(got.Hits) != 1 || got.Hits[0].SquaredDistance != 0 {
		t.Fatalf("translated series must match after population normalization: %+v", got)
	}
}

// TestCoverageSearch_WorkspaceAddressSpaceOverflow386 verifies representable budget arithmetic.
// On a 32-bit target, moderate input lengths can imply more than MaxInt bytes of workspace.
// Implementation:
//   - Stage 1: restrict this test to 32-bit int architectures.
//   - Stage 2: construct small borrowed slices and request one worker per admitted start.
//   - Stage 3: require numeric overflow before any worker allocation or candidate computation.
//
// Complexity: Time O(N+L), Space O(N+L) for the inputs; workspace must never be allocated.
// Notes: No hardcoded private struct sizes are needed: a DP cell contains at least one float64.
// AI-Hints: Do not run this workload on 64-bit targets where its workspace fits the int domain.
func TestCoverageSearch_WorkspaceAddressSpaceOverflow386(t *testing.T) {
	if strconv.IntSize != 32 {
		t.Skip("requires 32-bit int address-space accounting")
	}
	// There are exactly L complete starts. Two L-cell rows per worker require at least
	// 2*L*L*8 bytes, which exceeds MaxInt32 for L=16384 without allocating those rows.
	const length = 1 << 14
	query := make([]float64, length)
	data := make([]float64, 2*length-1)
	opts := dtw.DefaultSearchOptions()
	opts.Workers = length
	opts.Radius = -1
	// Unlimited MaxWorkingBytes does not waive representable-size validation.
	// A second-checkpoint cancellation is a safety fuse: if accounting regresses,
	// Search returns before creating a huge worker pool instead of exhausting memory.
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx := &coverageCheckpointContext{Context: parent, cancel: cancel, cancelAt: 2}
	got, err := dtw.Search(ctx, query, data, opts)
	mustCoverageSearchFailure(t, got, err, dtw.ErrOverflow, dtw.ErrNaNInf)
}

// TestCoverageSearch_SubnormalNormalizationCertifiedTie validates a scale-invariant ranking.
// A nonconstant subnormal query has positive population variance after scaling; it must
// match increasing candidates exactly, reject a descending candidate, and preserve the first tie.
// Implementation:
//   - Stage 1: construct three disjoint, nonconstant windows and a subnormal query.
//   - Stage 2: use strict diagonal squared costs and one worker for deterministic accounting.
//   - Stage 3: assert exact top-K, completed/pruned accounting and strict cutoff equality.
//
// Complexity: Time O(C*L), Space O(L+K) for this radius-zero workload.
// AI-Hints: Do not normalize through unscaled variance, which underflows on this query.
func TestCoverageSearch_SubnormalNormalizationCertifiedTie(t *testing.T) {
	query := []float64{0, math.SmallestNonzeroFloat64}
	data := []float64{1, 2, 10, 5, 2, 4}
	opts := dtw.DefaultSearchOptions()
	opts.Normalization = dtw.SearchZPopulation
	opts.Radius, opts.Stride = 0, len(query)
	got, err := dtw.Search(context.Background(), query, data, opts)
	mustNoError(t, err, "subnormal normalized search")
	if got == nil || !got.Complete || got.CandidatesTotal != 3 || len(got.Hits) != 1 {
		t.Fatalf("unexpected normalized ranking: %+v", got)
	}
	want := dtw.SearchHit{Window: dtw.IndexRange{Start: 0, End: 2}, SquaredDistance: 0}
	if got.Hits[0] != want {
		t.Fatalf("normalized winner=%+v; want %+v", got.Hits[0], want)
	}
	// Every increasing window normalizes to [-1,1]; the descending one to [1,-1].
	// Its first envelope deviation alone exceeds the zero cutoff. The later exact tie
	// must be evaluated rather than pruned, because equality cannot certify rejection.
	if got.CandidatesCompleted != 2 || got.CandidatesPruned != 1 || got.CellsVisited != 4 {
		t.Fatalf("strict certified cutoff accounting mismatch: %+v", got)
	}
}
