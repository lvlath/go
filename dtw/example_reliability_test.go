// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test demonstrates numerical and artifact contracts through public facades.
// These executable examples complement the longer scenario examples in example_test.go.
package dtw_test

import (
	"fmt"
	"math"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

// ExampleAlign_fullLocalArtifact demonstrates complete local diagnostics under a narrow band.
// A service comparing two already synchronized telemetry streams may prohibit
// temporal stretching while still needing the entire local-cost heatmap for review.
//
// Implementation:
//   - Stage 1: define two deterministic telemetry sequences and a diagonal-only band.
//   - Stage 2: ask Align to publish both the optimal path and full local-cost matrix.
//   - Stage 3: inspect a deliberately excluded cell without invoking any callback again.
//
// Behavior highlights:
//   - The window constrains DP paths, not the explicitly requested local artifact.
//
// Inputs:
//   - Two six-sample scalar series, WithWindow(0) and full artifact request.
//
// Returns:
//   - Printed finite distance, path length, artifact shape and an off-band cost.
//
// Errors:
//   - Unexpected input/storage errors are reported as example output.
//
// Determinism:
//   - Fixed sequences, row-major costs and diagonal-only witness.
//
// Complexity:
//   - Time O(n*m), Space O(n*m) for the requested LocalCost and path.
//
// Notes:
//   - Output matrices are detached from inputs and owned by the caller.
//
// AI-Hints:
//   - Do not infer zero local cost from cells outside the admitted window.
//   - Do not request LocalCost when only a low-memory distance is needed.
func ExampleAlign_fullLocalArtifact() {
	series := []float64{0, 1, 2, 3, 4, 5}
	res, err := dtw.Align(series, series,
		dtw.WithWindow(0),
		dtw.WithReturnPath(true),
		dtw.WithReturnLocalCost(true),
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	offBand, err := res.LocalCost.At(0, 5)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("distance=%g path-steps=%d\n", res.Distance, len(res.Path))
	fmt.Printf("local=%dx%d off-band=%g\n", res.LocalCost.Rows(), res.LocalCost.Cols(), offBand)

	// Output:
	// distance=0 path-steps=6
	// local=6x6 off-band=5
}

// ExampleAlignCostMatrix_discardedOverflow demonstrates finite optimality with overflowing alternatives.
// The precomputed local-cost surface contains extreme but finite penalties;
// its discarded non-diagonal branches may overflow without invalidating the optimum.
//
// Implementation:
//   - Stage 1: define four local costs and a finite slope penalty.
//   - Stage 2: request the optimal path and complete accumulated-cost diagnostic matrix.
//   - Stage 3: print the finite path and identify an overflowing intermediate cell.
//
// Behavior highlights:
//   - A finite optimal result is valid even when an unused DP branch overflows.
//
// Inputs:
//   - Two-by-two local surface with 1e308 off-diagonal costs.
//
// Returns:
//   - Printed finite optimal distance, deterministic path, and artifact overflow state.
//
// Errors:
//   - ErrOverflow would indicate the terminal minimum itself is unrepresentable.
//
// Determinism:
//   - Strict diagonal/up/left predecessor order and zero-based coordinates.
//
// Complexity:
//   - Time O(4) DP cells and O(9) accumulated artifact cells including boundaries.
//
// Notes:
//   - An accumulated +Inf cell can mean numeric overflow, not just absence.
//
// AI-Hints:
//   - Never return ErrOverflow solely because a discarded candidate overflowed.
//   - Keep +Inf artifact interpretation distinct from the result Reachable flag.
func ExampleAlignCostMatrix_discardedOverflow() {
	local, err := matrix.NewPreparedDense(2, 2)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	if err = local.Fill([]float64{1, 1e308, 1e308, 1}); err != nil {
		fmt.Println("error:", err)
		return
	}

	res, err := dtw.AlignCostMatrix(local,
		dtw.WithSlopePenalty(1e308),
		dtw.WithReturnPath(true),
		dtw.WithReturnAccumulated(true),
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	unusedCost, err := res.Accumulated.At(1, 2)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("distance=%g reachable=%t path=%v\n", res.Distance, res.Reachable, res.Path)
	fmt.Printf("overflowing-intermediate=%t\n", math.IsInf(unusedCost, 1))

	// Output:
	// distance=2 reachable=true path=[{0 0} {1 1}]
	// overflowing-intermediate=true
}
