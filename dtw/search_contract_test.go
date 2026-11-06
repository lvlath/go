// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test validates bounded alignment, resource accounting,
// error identities, deterministic search ranking, and caller ownership.
package dtw_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
)

// TestSearch_CutoffEqualityKeepsTie confirms canonical equal-score start order.
func TestSearch_CutoffEqualityKeepsTie(t *testing.T) {
	opts := dtw.DefaultSearchOptions()
	opts.Radius = 0
	result, err := dtw.Search(context.Background(), []float64{0, 0}, []float64{0, 0, 0}, opts)
	if err != nil || result == nil || !result.Complete {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	assertExactHits(t, result.Hits, []dtw.SearchHit{{Window: dtw.IndexRange{Start: 0, End: 2}, SquaredDistance: 0}})
}

// TestSearch_ExclusionHalfOpen confirms exclusions apply to starts, never sample overlap.
func TestSearch_ExclusionHalfOpen(t *testing.T) {
	original := []dtw.IndexRange{{Start: 1, End: 2}, {Start: 0, End: 1}, {Start: 2, End: 2}}
	opts := dtw.DefaultSearchOptions()
	opts.Radius, opts.ExcludedStarts = 0, original
	result, err := dtw.Search(context.Background(), []float64{0, 0}, []float64{0, 0, 0}, opts)
	if err != nil || result.CandidatesTotal != 0 || !result.Complete || len(result.Hits) != 0 {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if original[0] != (dtw.IndexRange{Start: 1, End: 2}) || original[1] != (dtw.IndexRange{Start: 0, End: 1}) {
		t.Fatalf("mutated input exclusion ranges: %v", original)
	}
	opts.ExcludedStarts = []dtw.IndexRange{{Start: 0, End: 1}}
	result, err = dtw.Search(context.Background(), []float64{0, 0}, []float64{0, 0, 0}, opts)
	if err != nil || result.CandidatesTotal != 1 {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	assertExactHits(t, result.Hits, []dtw.SearchHit{{Window: dtw.IndexRange{Start: 1, End: 3}, SquaredDistance: 0}})
}

// TestSearch_WorkersStable checks complete worker-count independence across all hits.
func TestSearch_WorkersStable(t *testing.T) {
	query := []float64{1, -2, 3, 1, 0, 2, 4}
	data := []float64{0, -1, 2, 4, 1, 0, 2, 3, -2, 1, 3, 4, 1, -1, 2}
	var reference *dtw.SearchResult
	for _, workers := range []int{1, 4} {
		opts := dtw.DefaultSearchOptions()
		opts.Workers, opts.TopK, opts.Radius = workers, 5, 2
		result, err := dtw.Search(context.Background(), query, data, opts)
		if err != nil || !result.Complete {
			t.Fatalf("workers=%d: result=%+v err=%v", workers, result, err)
		}
		if reference == nil {
			reference = result
			continue
		}
		assertExactHits(t, result.Hits, reference.Hits)
		if result.CandidatesCompleted != reference.CandidatesCompleted || result.CellsVisited != reference.CellsVisited {
			t.Fatalf("worker-count accounting drift: reference=%+v parallel=%+v", reference, result)
		}
	}
}

// TestSearch_PartialIsNotGlobalTopK checks the hard cell budget and completed-only Hits.
func TestSearch_PartialIsNotGlobalTopK(t *testing.T) {
	opts := dtw.DefaultSearchOptions()
	opts.Radius, opts.MaxCells = 0, 4
	result, err := dtw.Search(context.Background(), []float64{0, 0, 0}, []float64{0, 0, 0, 0}, opts)
	if !errors.Is(err, dtw.ErrBudget) || result == nil || result.Complete ||
		result.CellsVisited != 4 || result.CandidatesCompleted != 1 || result.CandidatesTotal != 2 {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	assertExactHits(t, result.Hits, []dtw.SearchHit{{Window: dtw.IndexRange{Start: 0, End: 3}, SquaredDistance: 0}})
}

// TestSearch_BudgetIncludesWorkerMemory confirms checked memory preflight is worker-sensitive.
func TestSearch_BudgetIncludesWorkerMemory(t *testing.T) {
	query := make([]float64, 16)
	data := make([]float64, 48)
	opts := dtw.DefaultSearchOptions()
	opts.Radius, opts.MaxWorkingBytes = 0, 200
	result, err := dtw.Search(context.Background(), query, data, opts)
	if err != nil || result == nil || !result.Complete {
		t.Fatalf("one worker should fit: result=%+v err=%v", result, err)
	}
	opts.Workers = 4
	result, err = dtw.Search(context.Background(), query, data, opts)
	if result != nil || !errors.Is(err, dtw.ErrBudget) {
		t.Fatalf("worker multiplication not budgeted: result=%+v err=%v", result, err)
	}
}

// TestSearch_ZPopulationRejectsConstantCandidateAndQuery verifies explicit degeneracy errors.
func TestSearch_ZPopulationRejectsConstantCandidateAndQuery(t *testing.T) {
	opts := dtw.DefaultSearchOptions()
	opts.Normalization = dtw.SearchZPopulation
	result, err := dtw.Search(context.Background(), []float64{3, 3}, []float64{0, 1, 2}, opts)
	if result != nil || !errors.Is(err, dtw.ErrZeroVariance) {
		t.Fatalf("constant query: result=%+v err=%v", result, err)
	}
	result, err = dtw.Search(context.Background(), []float64{0, 1}, []float64{0, 1, 2, 2}, opts)
	if result == nil || result.Complete || !errors.Is(err, dtw.ErrZeroVariance) || result.CandidatesCompleted != 2 {
		t.Fatalf("constant trailing candidate: result=%+v err=%v", result, err)
	}
}

// TestSearch_ValidationPrecedence checks ctx/options/input/shape/numeric ordering.
func TestSearch_ValidationPrecedence(t *testing.T) {
	_, err := dtw.Search(nil, nil, nil, dtw.SearchOptions{})
	if !errors.Is(err, dtw.ErrInvalidOption) {
		t.Fatalf("nil context precedence: %v", err)
	}
	_, err = dtw.Search(context.Background(), nil, nil, dtw.SearchOptions{})
	if !errors.Is(err, dtw.ErrInvalidOption) {
		t.Fatalf("invalid options precedence: %v", err)
	}
	opts := dtw.DefaultSearchOptions()
	_, err = dtw.Search(context.Background(), nil, []float64{0}, opts)
	if !errors.Is(err, dtw.ErrNilInput) {
		t.Fatalf("nil input: %v", err)
	}
	_, err = dtw.Search(context.Background(), []float64{}, []float64{}, opts)
	if !errors.Is(err, dtw.ErrEmptyInput) {
		t.Fatalf("empty query: %v", err)
	}
	_, err = dtw.Search(context.Background(), []float64{0}, []float64{math.Inf(1)}, opts)
	if !errors.Is(err, dtw.ErrNonFinite) || !errors.Is(err, dtw.ErrNaNInf) {
		t.Fatalf("nonfinite family: %v", err)
	}
	result, err := dtw.Search(context.Background(), []float64{1, 2}, []float64{9}, opts)
	if err != nil || !result.Complete || result.CandidatesTotal != 0 || len(result.Hits) != 0 {
		t.Fatalf("shorter data: result=%+v err=%v", result, err)
	}
}

// TestSearch_RuntimeOverflowKeepsPartial ensures numeric errors are not bad matches.
func TestSearch_RuntimeOverflowKeepsPartial(t *testing.T) {
	opts := dtw.DefaultSearchOptions()
	result, err := dtw.Search(context.Background(), []float64{0, 0}, []float64{0, 0, 1e200}, opts)
	if result == nil || result.Complete || result.CandidatesCompleted != 1 ||
		!errors.Is(err, dtw.ErrOverflow) || !errors.Is(err, dtw.ErrNaNInf) {
		t.Fatalf("runtime overflow: result=%+v err=%v", result, err)
	}
}

// TestAlignBounded_StatusContracts verifies exact equality, strict cutoff, and no-path.
func TestAlignBounded_StatusContracts(t *testing.T) {
	ctx := context.Background()
	exact, err := dtw.AlignBounded(ctx, []float64{0, 1}, []float64{0, 2}, 1, dtw.WithReturnPath(true))
	if err != nil || exact == nil || exact.Status != dtw.BoundExact || exact.Distance != 1 || exact.LowerBound != 1 || len(exact.Path) == 0 {
		t.Fatalf("exact: %+v err=%v", exact, err)
	}
	above, err := dtw.AlignBounded(ctx, []float64{0, 1}, []float64{0, 2}, 0)
	if err != nil || above.Status != dtw.BoundAboveLimit || !math.IsNaN(above.Distance) || above.LowerBound <= 0 || len(above.Path) != 0 {
		t.Fatalf("above: %+v err=%v", above, err)
	}
	unreachable, err := dtw.AlignBounded(ctx, []float64{0, 1}, []float64{0, 1, 2}, 10, dtw.WithWindow(0), dtw.WithReturnPath(true))
	if err != nil || unreachable.Status != dtw.BoundUnreachable || !math.IsInf(unreachable.Distance, 1) {
		t.Fatalf("unreachable: %+v err=%v", unreachable, err)
	}
	_, err = dtw.AlignBounded(ctx, []float64{0}, []float64{0}, 0, dtw.WithReturnLocalCost(true))
	if !errors.Is(err, dtw.ErrInvalidOption) {
		t.Fatalf("unpublishable artifacts must be rejected: %v", err)
	}
}

// TestAlignBounded_FinalOverflowCannotBecomeAboveLimit preserves numeric identity.
func TestAlignBounded_FinalOverflowCannotBecomeAboveLimit(t *testing.T) {
	result, err := dtw.AlignBounded(context.Background(), []float64{0, 0}, []float64{0}, 0,
		dtw.WithCostFunc(func(_, _ float64) (float64, error) { return 1e308, nil }))
	if result != nil || !errors.Is(err, dtw.ErrOverflow) || !errors.Is(err, dtw.ErrNaNInf) {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

// TestLBKeoghSquared_DiagnosticNoPruning verifies value/shape errors and non-shortlist semantics.
func TestLBKeoghSquared_DiagnosticNoPruning(t *testing.T) {
	bound, err := dtw.LBKeoghSquared([]float64{0, 4, 8}, []float64{1, 2, 5}, []float64{2, 3, 6})
	if err != nil || bound != 6 {
		t.Fatalf("bound=%g err=%v; want 1+1+4", bound, err)
	}
	_, err = dtw.LBKeoghSquared([]float64{1}, []float64{2}, []float64{0})
	if !errors.Is(err, dtw.ErrShape) {
		t.Fatalf("reversed envelope: %v", err)
	}
}

// TestSearch_ResultOwnership checks completed-hit mutation cannot affect future invocations.
func TestSearch_ResultOwnership(t *testing.T) {
	query := []float64{1, 2, 3}
	data := []float64{0, 1, 2, 3, 4}
	opts := dtw.DefaultSearchOptions()
	first, err := dtw.Search(context.Background(), query, data, opts)
	if err != nil {
		t.Fatal(err)
	}
	first.Hits[0].SquaredDistance = 12345
	second, err := dtw.Search(context.Background(), query, data, opts)
	if err != nil || second.Hits[0].SquaredDistance == 12345 {
		t.Fatalf("alias leaked: result=%+v err=%v", second, err)
	}
	if query[0] != 1 || query[1] != 2 || query[2] != 3 || data[0] != 0 || data[4] != 4 {
		t.Fatal("mutated inputs")
	}
}

// TestLBKeogh_NeverFalsePrunesNextafterBoundary verifies exact search retention
// around a representable adjacent score: the diagnostic LB is never used to prune.
func TestLBKeogh_NeverFalsePrunesNextafterBoundary(t *testing.T) {
	adjacent := math.Nextafter(1, math.Inf(1))
	query := []float64{0, 1}
	data := []float64{0, adjacent, 0, 1}
	opts := dtw.DefaultSearchOptions()
	opts.Radius, opts.TopK = 0, 2
	result, err := dtw.Search(context.Background(), query, data, opts)
	if err != nil || !result.Complete || result.CandidatesPruned != 0 {
		t.Fatalf("nextafter boundary: result=%+v err=%v", result, err)
	}
	want := exactSearchOracle(query, data, 0, 1, 2)
	assertExactHits(t, result.Hits, want)

	lower := []float64{0, adjacent}
	upper := []float64{0, adjacent}
	bound, err := dtw.LBKeoghSquared(query, lower, upper)
	if err != nil || math.IsNaN(bound) || math.IsInf(bound, 0) {
		t.Fatalf("diagnostic bound=%g err=%v", bound, err)
	}
}

// TestSearch_ZPopulationExtremeScale verifies finite normalized matches
// when raw coordinates are far beyond the safe range for naive sum-of-squares.
func TestSearch_ZPopulationExtremeScale(t *testing.T) {
	large := 1e308
	low := -1e308
	query := []float64{large, 0, low}
	data := []float64{large, 0, low, large, 0, low}
	opts := dtw.DefaultSearchOptions()
	opts.Radius, opts.TopK = 0, 2
	opts.Normalization = dtw.SearchZPopulation
	result, err := dtw.Search(context.Background(), query, data, opts)
	if err != nil || !result.Complete || len(result.Hits) != 2 {
		t.Fatalf("extreme normalization: result=%+v err=%v", result, err)
	}
	if result.Hits[0].Window.Start != 0 || result.Hits[0].SquaredDistance != 0 ||
		result.Hits[1].Window.Start != 3 || result.Hits[1].SquaredDistance != 0 {
		t.Fatalf("unexpected normalized exact matches: %+v", result.Hits)
	}
}

// TestSearch_ZWorkersStable ensures complete z-normalized rankings do not depend
// on worker scheduling and that no normalized candidate scratch crosses goroutines.
func TestSearch_ZWorkersStable(t *testing.T) {
	query := []float64{2, 4, 9, 3, -1}
	data := []float64{1, 3, 8, 2, 0, 1, 5, 8, 2, -2, 4, 9, 0, 3, 7, 2}
	opts := dtw.DefaultSearchOptions()
	opts.Radius, opts.TopK, opts.Normalization = 2, 4, dtw.SearchZPopulation
	serial, err := dtw.Search(context.Background(), query, data, opts)
	if err != nil || !serial.Complete {
		t.Fatalf("serial normalized search: result=%+v err=%v", serial, err)
	}
	opts.Workers = 4
	parallel, err := dtw.Search(context.Background(), query, data, opts)
	if err != nil || !parallel.Complete {
		t.Fatalf("parallel normalized search: result=%+v err=%v", parallel, err)
	}
	assertExactHits(t, parallel.Hits, serial.Hits)
	if parallel.CellsVisited != serial.CellsVisited || parallel.CandidatesCompleted != serial.CandidatesCompleted {
		t.Fatalf("normalized worker accounting mismatch: serial=%+v parallel=%+v", serial, parallel)
	}
}

// TestSearch_ParallelHardCellLimit demonstrates MaxCells never overdraws
// even when multiple independent workers contend for their next cell credit.
func TestSearch_ParallelHardCellLimit(t *testing.T) {
	query := []float64{0, 1, 0, 1, 0, 1, 0, 1}
	data := make([]float64, 128)
	for i := range data {
		data[i] = float64(i % 3)
	}
	opts := dtw.DefaultSearchOptions()
	opts.Radius, opts.Workers, opts.MaxCells = 2, 4, 3
	result, err := dtw.Search(context.Background(), query, data, opts)
	if !errors.Is(err, dtw.ErrBudget) || result == nil || result.Complete || result.CellsVisited > 3 {
		t.Fatalf("atomic hard-budget breach: result=%+v err=%v", result, err)
	}
	if result.CandidatesCompleted+result.CandidatesPruned > result.CandidatesTotal {
		t.Fatalf("invalid completed accounting: %+v", result)
	}
}

// TestSearch_ExclusionCountsAgainstIndependentStartScan compares coalesced
// stride progression counts against an elementary direct enumeration.
func TestSearch_ExclusionCountsAgainstIndependentStartScan(t *testing.T) {
	query := []float64{1, 2, 3}
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	for _, stride := range []int{1, 2, 3, 5} {
		exclusions := []dtw.IndexRange{
			{Start: 7, End: 9},
			{Start: 1, End: 4},
			{Start: 3, End: 6},
			{Start: 0, End: 0},
		}
		expected := uint64(0)
		for start := 0; start+len(query) <= len(data); start += stride {
			blocked := false
			for _, exclusion := range exclusions {
				if start >= exclusion.Start && start < exclusion.End {
					blocked = true
				}
			}
			if !blocked {
				expected++
			}
		}
		opts := dtw.DefaultSearchOptions()
		opts.Radius, opts.Stride, opts.TopK = 0, stride, 20
		opts.ExcludedStarts = exclusions
		got, err := dtw.Search(context.Background(), query, data, opts)
		if err != nil || !got.Complete || got.CandidatesTotal != expected || got.CandidatesCompleted != expected {
			t.Fatalf("stride=%d expected=%d result=%+v err=%v", stride, expected, got, err)
		}
	}
}
