// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External executable examples demonstrate six independent series contracts.
// Inputs are fixed and results are printed in deterministic index order.
package series_test

import (
	"fmt"
	"math"

	"github.com/lvlath/go/series"
)

// ExampleValidate demonstrates a missing telemetry slot on an integer grid.
// Exact cadence identifies a gap without inserting a fabricated observation.
func ExampleValidate() {
	// A missing sample between 10 ns and 30 ns leaves three supplied samples.
	times := []int64{0, 10, 30}
	values := []float64{2, 4, 8}
	grid, err := series.Validate(times, values, series.GridOptions{CadenceNS: 10, AllowGaps: true})
	if err != nil {
		fmt.Println(err)
		return
	}
	// Gap{AfterIndex:1, MissingCount:1} describes the absent index slot.
	fmt.Println(grid.Samples, grid.Gaps, err)
	// Output: 3 [{1 1}] <nil>
}

// ExampleLag demonstrates a causal shift with a validity mask.
// Insufficient history remains in the original output length as zero/false.
func ExampleLag() {
	shifted, err := series.Lag([]float64{2, 4, 8}, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(shifted.Values, shifted.Valid)
	// Output: [0 2 4] [false true true]
}

// ExampleDifference computes differences over a two-sample horizon.
// The first two output positions are warmup; later positions use past values.
func ExampleDifference() {
	differences, err := series.Difference([]float64{2, 4, 8, 16}, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(differences.Values, differences.Valid)
	// Output: [0 0 6 12] [false false true true]
}

// ExampleSimpleReturns illustrates a valid terminal zero observation.
// The zero can be a current value even though it cannot later be a denominator.
func ExampleSimpleReturns() {
	returns, err := series.SimpleReturns([]float64{2, 4, 0})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(returns.Values, returns.Valid)
	// Output: [0 1 -1] [false true true]
}

// ExampleLogReturns illustrates extreme ratios and tiny observable changes.
// The kernel never forms an overflowing ratio and retains one-ULP increments.
func ExampleLogReturns() {
	// Both the extreme positive interval and nearby nextafter pair are valid.
	values := []float64{1e-300, 1e300, math.Nextafter(1e300, math.Inf(1))}
	returns, err := series.LogReturns(values)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("extreme finite=%t, tiny positive=%t, mask=%v\n",
		!math.IsInf(returns.Values[1], 0), returns.Values[2] > 0, returns.Valid)
	// Output: extreme finite=true, tiny positive=true, mask=[false true true]
}

// ExampleWindowRanges enumerates complete, half-open index intervals.
// The incomplete trailing two samples are not emitted as an additional window.
func ExampleWindowRanges() {
	ranges, err := series.WindowRanges(8, 3, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(ranges)
	// Output: [{0 3} {2 5} {4 7}]
}
