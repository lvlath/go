// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test demonstrates the new bounded and fixed-window exact-search contracts.
// Executable examples use the public facade and never reference private DP state.
package dtw_test

import (
	"context"
	"fmt"
	"math"

	"github.com/lvlath/go/dtw"
)

// ExampleSearch_fixedLengthTopK ranks two exact recurring three-frame signatures.
// It distinguishes global exact hits from a heuristic shortlist.
func ExampleSearch_fixedLengthTopK() {
	query := []float64{0, 1, 0}
	series := []float64{5, 0, 1, 0, 5, 0, 1, 0, 3}
	options := dtw.DefaultSearchOptions()
	options.Radius = 0
	options.TopK = 2

	result, err := dtw.Search(context.Background(), query, series, options)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	for _, hit := range result.Hits {
		fmt.Printf("window=[%d,%d) squared=%g\n", hit.Window.Start, hit.Window.End, hit.SquaredDistance)
	}
	fmt.Println("global-exact:", result.Complete)
	// Output:
	// window=[1,4) squared=0
	// window=[5,8) squared=0
	// global-exact: true
}

// ExampleSearch_zPopulation compares equally shaped signals at different offsets.
// Population normalization is explicit, independently fitted to each full window.
func ExampleSearch_zPopulation() {
	query := []float64{10, 11, 12}
	series := []float64{0, 1, 2, 10, 11, 12}
	options := dtw.DefaultSearchOptions()
	options.Normalization = dtw.SearchZPopulation
	options.Radius = 0
	options.TopK = 2

	result, err := dtw.Search(context.Background(), query, series, options)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("starts=%d,%d complete=%t\n",
		result.Hits[0].Window.Start, result.Hits[1].Window.Start, result.Complete)
	// Output:
	// starts=0,3 complete=true
}

// ExampleAlignBounded_inclusiveLimit explains why equality is exact rather than cutoff.
// AboveLimit never masquerades as a measured finite distance.
func ExampleAlignBounded_inclusiveLimit() {
	a := []float64{0, 1, 2}
	b := []float64{0, 2, 4}
	exact, err := dtw.AlignBounded(context.Background(), a, b, 3, dtw.WithWindow(0))
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	above, err := dtw.AlignBounded(context.Background(), a, b, 2, dtw.WithWindow(0))
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("exact=%t distance=%g\n", exact.Status == dtw.BoundExact, exact.Distance)
	fmt.Printf("above=%t unavailable=%t lower=%g\n",
		above.Status == dtw.BoundAboveLimit, math.IsNaN(above.Distance), above.LowerBound)
	// Output:
	// exact=true distance=3
	// above=true unavailable=true lower=3
}

// ExampleSearch_excludedStarts shows half-open exclusion semantics for candidate starts.
// Excluding start zero still permits a hit whose samples overlap that first window.
func ExampleSearch_excludedStarts() {
	options := dtw.DefaultSearchOptions()
	options.Radius = 0
	options.ExcludedStarts = []dtw.IndexRange{{Start: 0, End: 1}}

	result, err := dtw.Search(context.Background(), []float64{0, 0}, []float64{0, 0, 0}, options)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("start=%d total=%d complete=%t\n",
		result.Hits[0].Window.Start, result.CandidatesTotal, result.Complete)
	// Output:
	// start=1 total=1 complete=true
}

// ExampleLBKeoghSquared_envelopeDiagnostic computes a standalone envelope deviation.
// The diagnostic float64 bound is not automatically certified for pruning exact top-K.
func ExampleLBKeoghSquared_envelopeDiagnostic() {
	query := []float64{0, 4, 8}
	lower := []float64{1, 2, 5}
	upper := []float64{2, 3, 6}
	bound, err := dtw.LBKeoghSquared(query, lower, upper)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("squared-envelope-bound=%g\n", bound)
	// Output:
	// squared-envelope-bound=6
}

// ExampleSearch_certifiedPruning demonstrates a provably inferior candidate exclusion.
// A finite zero-cost template establishes the cutoff, while distant later windows
// are excluded only after verifying an envelope bound in the scorer's float64 model.
//
// Implementation:
//   - Stage 1: define one exact four-frame template followed by distant samples.
//   - Stage 2: search with one worker to make performance counters reproducible.
//   - Stage 3: inspect both the exact winning hit and the certificate accounting.
//
// Behavior highlights:
//   - The skipped windows are certified noncompetitive, not approximate matches.
//
// Inputs:
//   - No parameters; fixed finite scalar samples with a diagonal radius.
//
// Returns:
//   - Deterministic output from one complete SearchResult.
//
// Errors:
//   - Any classified search error prints diagnostics instead of a false success.
//
// Determinism:
//   - One worker processes candidate starts in increasing order.
//
// Complexity:
//   - Worst-case O(C*L) time for Radius=0, O(L) private workspace.
//
// Notes:
//   - The count of pruned candidates is a performance diagnostic, not a speed guarantee.
//
// AI-Hints:
//   - Never infer that pruning is safe from a rounded diagnostic LBKeoghSquared alone.
func ExampleSearch_certifiedPruning() {
	query := []float64{0, 1, 2, 3}
	series := []float64{0, 1, 2, 3, 100, 101, 102, 103, 104, 105}
	options := dtw.DefaultSearchOptions()
	options.Radius = 0
	options.Workers = 1

	result, err := dtw.Search(context.Background(), query, series, options)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("winner=[%d,%d) squared=%g\n", result.Hits[0].Window.Start, result.Hits[0].Window.End, result.Hits[0].SquaredDistance)
	fmt.Printf("complete=%t scored=%d pruned=%d cells=%d\n", result.Complete, result.CandidatesCompleted, result.CandidatesPruned, result.CellsVisited)
	// Output:
	// winner=[0,4) squared=0
	// complete=true scored=1 pruned=6 cells=4
}
