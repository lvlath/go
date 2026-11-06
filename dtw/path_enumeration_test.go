// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies path optimality by enumerating all admissible monotone paths.
// This oracle does not call the production recurrence, predecessor picker, or cost helpers.
package dtw_test

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
)

// oracleStepRank orders one predecessor transition independently of the DP implementation.
// Smaller ranks represent diagonal, vertical/up, and horizontal/left steps,
// in exactly that order as seen when walking a path backward from its endpoint.
//
// Implementation:
//   - Stage 1: read the previous and current coordinates.
//   - Stage 2: classify their monotone delta with explicit structural checks.
//   - Stage 3: return the tie preference rank.
//
// Behavior highlights:
//   - Only three valid DTW step types are accepted by this oracle.
//
// Inputs:
//   - before, after: two adjacent zero-based coordinates on a legal path.
//
// Returns:
//   - int: rank 0 (diagonal), 1 (up), or 2 (left).
//
// Errors:
//   - None; invalid oracle paths panic to expose malformed fixture construction.
//
// Determinism:
//   - A pure rank based solely on integer coordinate differences.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The panic is test-only and cannot affect the production API.
//
// AI-Hints:
//   - Compare path ties backward from the terminal, not forward from the origin.
//   - Do not call production predecessor helpers from an independent oracle.
func oracleStepRank(before, after dtw.Coord) int {
	di, dj := after.I-before.I, after.J-before.J
	switch {
	case di == 1 && dj == 1:
		return 0
	case di == 1 && dj == 0:
		return 1
	case di == 0 && dj == 1:
		return 2
	default:
		panic("oracle received a non-DTW step")
	}
}

// oraclePrefersWitness resolves equal-cost paths by terminal-first predecessor order.
// It compares the recorded coordinate steps in reverse, without using any
// accumulated-cost matrix or production predecessor code.
//
// Implementation:
//   - Stage 1: walk both candidate paths backward from their final coordinate.
//   - Stage 2: compare step ranks at the first differing predecessor choice.
//   - Stage 3: retain the earlier path when every predecessor choice is equal.
//
// Behavior highlights:
//   - Equivalent paths retain their existing representative.
//
// Inputs:
//   - candidate, existing: legal paths with the same endpoints and rounded score.
//
// Returns:
//   - bool: whether candidate wins the prescribed deterministic tie.
//
// Errors:
//   - None for legal oracle paths.
//
// Determinism:
//   - Terminal-to-origin comparison of diagonal/up/left rank.
//
// Complexity:
//   - Time O(n+m), Space O(1).
//
// Notes:
//   - Intended only for exact equal-cost ties in small exhaustive fixtures.
//
// AI-Hints:
//   - Comparing forward lexicographic coordinates does not model DP tie order.
//   - Epsilon must not convert a strictly more expensive path into a tie.
func oraclePrefersWitness(candidate, existing dtw.Path) bool {
	ci, ei := len(candidate)-1, len(existing)-1
	for ci > 0 && ei > 0 {
		cr := oracleStepRank(candidate[ci-1], candidate[ci])
		er := oracleStepRank(existing[ei-1], existing[ei])
		if cr != er {
			return cr < er
		}
		ci--
		ei--
	}
	return false
}

// enumerateDTWPathsOracle exhaustively searches admissible monotone warping paths.
// It computes each complete path's objective by forward float64 additions,
// independent of the package's DP recurrence and classification machinery.
//
// Implementation:
//   - Stage 1: begin at (0,0) with its single local cost.
//   - Stage 2: recursively enumerate right, down, and diagonal steps within window;
//     each step adds its slope penalty first, then the destination local cost.
//   - Stage 3: compare complete path objectives strictly and resolve exact ties
//     by terminal-to-origin diagonal/up/left rank.
//
// Behavior highlights:
//   - Enumerates every admissible path; it does not prune by DP state.
//   - Tests deliberately keep costs small and exactly representable to avoid
//     rounding-induced global tie ambiguity in objective oracle comparisons.
//
// Inputs:
//   - local: non-empty rectangular finite non-negative cost surface.
//   - penalty: finite non-negative slope cost.
//   - window: -1 for unconstrained or non-negative Sakoe-Chiba radius.
//
// Returns:
//   - float64: minimal enumerated forward objective.
//   - dtw.Path: independent selected witness, detached from recursion storage.
//   - bool: whether any admissible terminal path exists.
//
// Errors:
//   - None; exhaustive fixtures satisfy the domain preconditions.
//
// Determinism:
//   - Recursion order fixed; exact path ties resolved by reverse step ranks.
//
// Complexity:
//   - Exponential in sequence sizes (bounded here to n,m <= 6).
//   - Working recursion O(n+m); selected witness O(n+m).
//
// Notes:
//   - This is deliberately not production code; it is a differential oracle.
//
// AI-Hints:
//   - Never reuse production recurrence logic in the oracle.
//   - The oracle must compare full legal paths, not numeric epsilon proximity.
func enumerateDTWPathsOracle(local [][]float64, penalty float64, window int) (float64, dtw.Path, bool) {
	rows, cols := len(local), len(local[0])
	var selected dtw.Path
	minimum := math.Inf(1)
	path := dtw.Path{{I: 0, J: 0}}

	var visit func(i, j int, objective float64)
	visit = func(i, j int, objective float64) {
		if i == rows-1 && j == cols-1 {
			if selected == nil || objective < minimum || (objective == minimum && oraclePrefersWitness(path, selected)) {
				minimum = objective
				selected = append(dtw.Path(nil), path...)
			}
			return
		}

		for _, step := range [][2]int{{0, 1}, {1, 0}, {1, 1}} {
			ni, nj := i+step[0], j+step[1]
			if ni >= rows || nj >= cols || window >= 0 && absOracle(ni-nj) > window {
				continue
			}

			nextObjective := objective
			if step != ([2]int{1, 1}) {
				nextObjective += penalty
			}
			nextObjective += local[ni][nj]
			path = append(path, dtw.Coord{I: ni, J: nj})
			visit(ni, nj, nextObjective)
			path = path[:len(path)-1]
		}
	}
	visit(0, 0, local[0][0])
	return minimum, selected, selected != nil
}

// absOracle returns the non-negative displacement of two small test indices.
// The helper is deliberately independent of the production window predicate.
//
// Implementation:
//   - Stage 1: check the difference's sign.
//   - Stage 2: negate a negative difference or keep a non-negative one.
//   - Stage 3: return the absolute index displacement.
//
// Behavior highlights:
//   - Test indices are small, so the integer MinInt case is outside this domain.
//
// Inputs:
//   - difference: difference of two small zero-based fixture indices.
//
// Returns:
//   - int: absolute displacement in time-step units.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure integer arithmetic, no dependency on the DTW package internals.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Only fixture indices up to five are passed by this oracle.
//
// AI-Hints:
//   - Do not reuse the private production window helper in an independent oracle.
//   - Window=0 permits only equal sequence indices.
func absOracle(difference int) int {
	if difference < 0 {
		return -difference
	}
	return difference
}

// TestAlign_PathCostEqualsForwardObjective checks independent path optimality and witness parity.
// Exhaustively enumerated paths must agree with both the scalar callback and
// precomputed matrix adapters on the distance and deterministic representative.
//
// Implementation:
//   - Stage 1: build several rectangular deterministic local-cost surfaces up to 6x6.
//   - Stage 2: enumerate all admissible paths with no production DP calls.
//   - Stage 3: run both public facades with matching costs, penalty, and window;
//     compare numeric optimum, path coordinates, and explicit no-path protocol.
//
// Behavior highlights:
//   - Costs are small exact binary fractions, separating tie policy from rounding.
//   - The scalar callback maps sample labels directly to the same local surface.
//
// Inputs:
//   - Exhaustive fixtures with n,m in [2,6], windows -1/0/1 and penalties 0/0.5.
//
// Returns:
//   - Test failure on distance, witness, or structural-unreachability mismatch.
//
// Errors:
//   - ErrNoPath is expected only for an inadmissible terminal window.
//
// Determinism:
//   - Fixed fixture loops and independently ranked backward ties.
//
// Complexity:
//   - Exponential oracle time per fixture; production remains O(n*m).
//
// Notes:
//   - This test is bounded and intentionally avoids exponential runtime on large n.
//
// AI-Hints:
//   - Do not change a mathematically correct oracle just to satisfy production output.
//   - Compare scalar/flat only with the same explicitly supplied local costs.
func TestAlign_PathCostEqualsForwardObjective(t *testing.T) {
	for rows := 2; rows <= 6; rows++ {
		for cols := 2; cols <= 6; cols++ {
			local := make([][]float64, rows)
			flat := make([]float64, 0, rows*cols)
			for i := 0; i < rows; i++ {
				local[i] = make([]float64, cols)
				for j := 0; j < cols; j++ {
					local[i][j] = float64((7*i+11*j+i*j)%5) * 0.25
					flat = append(flat, local[i][j])
				}
			}
			costMatrix := regressionDense(t, rows, cols, flat...)
			sequenceA := make([]float64, rows)
			sequenceB := make([]float64, cols)
			for i := range sequenceA {
				sequenceA[i] = float64(i)
			}
			for j := range sequenceB {
				sequenceB[j] = float64(j)
			}

			for _, window := range []int{-1, 0, 1} {
				for _, penalty := range []float64{0, 0.5} {
					name := fmt.Sprintf("%dx%d/window=%d/penalty=%g", rows, cols, window, penalty)
					t.Run(name, func(t *testing.T) {
						wantCost, wantPath, reachable := enumerateDTWPathsOracle(local, penalty, window)
						opts := []dtw.Option{dtw.WithWindow(window), dtw.WithSlopePenalty(penalty), dtw.WithReturnPath(true)}

						matrixResult, matrixErr := dtw.AlignCostMatrix(costMatrix, opts...)
						scalarResult, scalarErr := dtw.Align(sequenceA, sequenceB, append(opts, dtw.WithCostFunc(func(a, b float64) (float64, error) {
							return local[int(a)][int(b)], nil
						}))...)
						if !reachable {
							for _, got := range []struct {
								result *dtw.Result
								err    error
							}{{matrixResult, matrixErr}, {scalarResult, scalarErr}} {
								if !errors.Is(got.err, dtw.ErrNoPath) || got.result == nil || got.result.Reachable || !math.IsInf(got.result.Distance, 1) {
									t.Fatalf("unreachable: result=%+v err=%v", got.result, got.err)
								}
							}
							return
						}

						for _, got := range []struct {
							label  string
							result *dtw.Result
							err    error
						}{{"matrix", matrixResult, matrixErr}, {"scalar", scalarResult, scalarErr}} {
							if got.err != nil || got.result == nil || !got.result.Reachable || got.result.Distance != wantCost {
								t.Fatalf("%s: result=%+v err=%v, oracle=%g", got.label, got.result, got.err, wantCost)
							}
							mustPathEqual(t, got.result.Path, wantPath, got.label+" independent oracle witness")
						}
					})
				}
			}
		}
	}
}
