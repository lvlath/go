// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies range classification and exact path witnesses independently.
// This file uses public interfaces only; no private production recurrence is invoked.
package dtw_test

import (
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

// rationalSumExceedsMax checks exact binary64 operand values without float64 addition.
// The test oracle deliberately uses big.Rat so cases just above MaxFloat64
// cannot be misclassified merely because the rounded sum is still finite.
//
// Implementation:
//   - Stage 1: encode each original float64 operand as its exact binary rational.
//   - Stage 2: add in rational arithmetic with no rounding.
//   - Stage 3: compare the result to the exact rational MaxFloat64 boundary.
//
// Inputs:
//   - a, b: finite non-negative original binary64 values.
//
// Returns:
//   - bool: whether the exact real sum exceeds the largest finite float64.
//
// Errors:
//   - None; test fixtures establish the preconditions.
//
// Determinism:
//   - Exact rational addition and comparison.
//
// Complexity:
//   - O(bit length of the operands) arithmetic and allocation per call.
//
// Notes:
//   - This oracle must never be called by a production kernel.
//
// AI-Hints:
//   - Decimal string conversions are not an exact float64 input oracle.
//   - Check exact overflow even when a+b has rounded to MaxFloat64.
func rationalSumExceedsMax(a, b float64) bool {
	left := new(big.Rat).SetFloat64(a)
	right := new(big.Rat).SetFloat64(b)
	boundary := new(big.Rat).SetFloat64(math.MaxFloat64)
	return new(big.Rat).Add(left, right).Cmp(boundary) > 0
}

// TestAlign_MaxFloatBoundaryRationalOracle checks exact overflow near the float64 limit.
// It exercises a sum that is mathematically above MaxFloat64 but rounds back
// to a finite value, as well as exactly representable boundary cases.
//
// Implementation:
//   - Stage 1: classify input pairs independently with exact big.Rat arithmetic.
//   - Stage 2: align each pair as two mandatory costs in a single-column matrix.
//   - Stage 3: verify exact error identity or finite boundary result.
//
// Behavior highlights:
//   - A discarded rounded-overflow loophole must not publish a false finite optimum.
//
// Inputs:
//   - Fixed finite binary64 boundary operands.
//
// Returns:
//   - Test failure if mathematical range classification differs from the oracle.
//
// Errors:
//   - ErrOverflow and ErrNaNInf must both classify unrepresentable totals.
//
// Determinism:
//   - Fixed table iteration order; comparisons are exact.
//
// Complexity:
//   - O(1) tested DP cells and exact-rational operations per fixture.
//
// Notes:
//   - This tests range classification, not an arbitrary-precision DTW optimization mode.
//
// AI-Hints:
//   - The expected outcome is derived from rational operands, never production DP.
//   - Do not weaken the test to errors.Is(err, ErrNaNInf) alone.
func TestAlign_MaxFloatBoundaryRationalOracle(t *testing.T) {
	previous := math.Nextafter(math.MaxFloat64, 0)
	for _, tc := range []struct {
		name, comment string
		first, second float64
	}{
		{"exact-max", "exactly largest finite", math.MaxFloat64, 0},
		{"one-subnormal-above", "rounded sum still equals MaxFloat64", math.MaxFloat64, math.SmallestNonzeroFloat64},
		{"one-ulp-to-max", "exact restoration of MaxFloat64", previous, math.MaxFloat64 - previous},
		{"ordinary-overflow", "infinite rounded sum", math.MaxFloat64, 1e308},
	} {
		t.Run(tc.name, func(t *testing.T) {
			oracleOverflow := rationalSumExceedsMax(tc.first, tc.second)
			local := regressionDense(t, 2, 1, tc.first, tc.second)
			result, err := dtw.AlignCostMatrix(local)
			if oracleOverflow {
				if result != nil || !errors.Is(err, dtw.ErrOverflow) || !errors.Is(err, dtw.ErrNaNInf) {
					t.Fatalf("%s: result=%+v err=%v; want both numerical sentinels", tc.comment, result, err)
				}
				return
			}
			if err != nil || result == nil || !result.Reachable || result.Distance != math.MaxFloat64 {
				t.Fatalf("%s: result=%+v err=%v; want finite MaxFloat64", tc.comment, result, err)
			}
		})
	}
}

// TestAlignCostMatrix_OverflowFinalMinimum requires nil result for final overflow.
// Two mandatory large finite costs exceed the float64 range even though each
// individual local-cost cell is valid input to the algorithm.
//
// Implementation:
//   - Stage 1: create a two-row single-column matrix of finite local costs.
//   - Stage 2: align its only admissible path.
//   - Stage 3: verify nil result and both numeric error classifications.
//
// Behavior highlights:
//   - A structurally reachable overflowing terminal must never look unreachable.
//
// Inputs:
//   - Two local costs of 1e308 each.
//
// Returns:
//   - Test failure for a false no-path result or absent overflow sentinel.
//
// Errors:
//   - ErrOverflow joined with ErrNaNInf.
//
// Determinism:
//   - Exactly one structural path in a 2x1 local surface.
//
// Complexity:
//   - Time O(1), Space O(1) fixture size.
//
// Notes:
//   - The input Dense remains finite despite accumulated arithmetic overflow.
//
// AI-Hints:
//   - Do not classify a final +Inf accumulated value via Reachable=false.
//   - Check both numeric sentinels via errors.Is.
func TestAlignCostMatrix_OverflowFinalMinimum(t *testing.T) {
	local := regressionDense(t, 2, 1, 1e308, 1e308)
	got, err := dtw.AlignCostMatrix(local)
	if got != nil || !errors.Is(err, dtw.ErrOverflow) || !errors.Is(err, dtw.ErrNaNInf) {
		t.Fatalf("result=%+v err=%v, want nil result and both numerical sentinels", got, err)
	}
}

// TestAlignMatrix_ExtremeSquaredL2 validates finite and unrepresentable feature distances.
// The fixtures separately exercise coordinate cancellation, large/small scale
// accumulation, and squared-distance overflow with valid finite input coordinates.
//
// Implementation:
//   - Stage 1: construct explicit row vectors using the public matrix API.
//   - Stage 2: align each single-row pair through AlignMatrix.
//   - Stage 3: compare finite exact values or both numeric error sentinels.
//
// Behavior highlights:
//   - True unrepresentable local squared distances are artifact numeric errors.
//
// Inputs:
//   - Fixed 1-by-2 multivariate rows of finite float64 values.
//
// Returns:
//   - Test failure on loss of precision or silent overflow.
//
// Errors:
//   - ErrOverflow joined with ErrNaNInf for unrepresentable local cost.
//
// Determinism:
//   - Single-cell DTW with ascending feature order.
//
// Complexity:
//   - Time O(d), Space O(d) fixtures.
//
// Notes:
//   - Does not change scalar Align's absolute-cost default.
//
// AI-Hints:
//   - Use coordinate differences, not squared-coordinate dot-product identities.
//   - Preserve representable subnormals without an absolute epsilon clamp.
func TestAlignMatrix_ExtremeSquaredL2(t *testing.T) {
	for _, tc := range []struct {
		name     string
		x, y     []float64
		want     float64
		overflow bool
	}{
		{"mixed-scale", []float64{1e100, 1e-100}, []float64{0, 0}, 1e200, false},
		{"equal-huge", []float64{1e200, 1e200}, []float64{1e200, 1e200}, 0, false},
		{"unrepresentable", []float64{1e200, 0}, []float64{0, 0}, 0, true},
		{"subnormal-square", []float64{math.Ldexp(1, -537), 0}, []float64{0, 0}, math.SmallestNonzeroFloat64, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			x := regressionDense(t, 1, 2, tc.x...)
			y := regressionDense(t, 1, 2, tc.y...)
			result, err := dtw.AlignMatrix(x, y)
			if tc.overflow {
				if result != nil || !errors.Is(err, dtw.ErrOverflow) || !errors.Is(err, dtw.ErrNaNInf) {
					t.Fatalf("result=%+v err=%v; want classified local overflow", result, err)
				}
				return
			}
			if err != nil || result == nil || !result.Reachable || result.Distance != tc.want {
				t.Fatalf("result=%+v err=%v; want %g", result, err, tc.want)
			}
		})
	}
}

// genericCostMatrix implements Matrix through embedding, not a Dense type assertion.
// A delegated At call intentionally forces the generic interface code path.
type genericCostMatrix struct{ matrix.Matrix }

// At forwards one element read through the embedded generic Matrix interface.
// The wrapper exercises the same numerical adapter without the concrete
// *matrix.Dense dynamic type used by the direct adapter case.
//
// Implementation:
//   - Stage 1: preserve the caller's zero-based row and column indices.
//   - Stage 2: delegate to the embedded Matrix.At implementation exactly once.
//   - Stage 3: return the underlying value and error unchanged.
//
// Behavior highlights:
//   - No caching or alternate numerical kernel is introduced by the wrapper.
//
// Inputs:
//   - i, j: zero-based row and column indices in the embedded Matrix.
//
// Returns:
//   - float64: the same value as the embedded Matrix.At.
//   - error: the embedded implementation's original error identity.
//
// Errors:
//   - Underlying Matrix.At errors propagate without translation.
//
// Determinism:
//   - Exactly one deterministic delegated read per invocation.
//
// Complexity:
//   - Time O(1) for Dense-backed fixture, Space O(1).
//
// Notes:
//   - The embedded fixture is read-only during parity tests.
//
// AI-Hints:
//   - Do not detect this wrapper by concrete Dense type assertion.
//   - The generic and Dense results must be exactly equal for fixed inputs.
func (g genericCostMatrix) At(i, j int) (float64, error) { return g.Matrix.At(i, j) }

// TestAlignMatrix_GenericAndDenseParity proves identical squared costs across backends.
// Wrapping Dense in a generic Matrix implementation must not change cost values,
// overflow identity, path choices, or detached local-cost artifacts.
//
// Implementation:
//   - Stage 1: construct fixed small multivariate Dense inputs.
//   - Stage 2: compare Dense and generic Matrix adapters under the same options.
//   - Stage 3: compare distance, path, and every published local-cost cell.
//
// Behavior highlights:
//   - Both adapters use the same ordered coordinate-difference law.
//
// Inputs:
//   - Two multivariate sequences of 3 and 4 rows, 2 features each.
//
// Returns:
//   - Test failure on any observable backend-specific mathematical difference.
//
// Errors:
//   - Any unexpected adapter or DP error fails this test.
//
// Determinism:
//   - Fixed row-major matrix and path traversal.
//
// Complexity:
//   - O(nx*ny*d) per adapter, plus O(nx*ny) artifact comparisons.
//
// Notes:
//   - Neither input matrix is modified.
//
// AI-Hints:
//   - Never optimize Dense by introducing an alternative squared-cost formula.
//   - Keep Coordinate order {I in x, J in y} on both paths.
func TestAlignMatrix_GenericAndDenseParity(t *testing.T) {
	x := regressionDense(t, 3, 2, 1e9, 2, 1e9+1, 3, 1e9+2, 4)
	y := regressionDense(t, 4, 2, 1e9, 2, 1e9, 3, 1e9+1, 3, 1e9+2, 4)
	options := []dtw.Option{dtw.WithReturnPath(true), dtw.WithReturnLocalCost(true)}
	plain, err := dtw.AlignMatrix(x, y, options...)
	if err != nil {
		t.Fatalf("dense: %v", err)
	}
	generic, err := dtw.AlignMatrix(genericCostMatrix{x}, genericCostMatrix{y}, options...)
	if err != nil {
		t.Fatalf("generic: %v", err)
	}
	if plain.Distance != generic.Distance || len(plain.Path) != len(generic.Path) {
		t.Fatalf("dense=%+v, generic=%+v", plain, generic)
	}
	mustPathEqual(t, generic.Path, plain.Path, "matrix generic/Dense path parity")
	for i := 0; i < plain.LocalCost.Rows(); i++ {
		for j := 0; j < plain.LocalCost.Cols(); j++ {
			a, errA := plain.LocalCost.At(i, j)
			b, errB := generic.LocalCost.At(i, j)
			if errA != nil || errB != nil || a != b {
				t.Fatalf("local[%d,%d] dense=%g generic=%g errA=%v errB=%v", i, j, a, b, errA, errB)
			}
		}
	}
}

// TestAlign_StrictTieDoesNotPickMoreExpensivePath proves epsilon cannot replace strict order.
// A locally almost-equal diagonal candidate is still more expensive than
// a horizontal predecessor, even if their costs differ by less than 1e-9.
//
// Implementation:
//   - Stage 1: define a 2-by-3 local-cost surface with a near-tie terminal.
//   - Stage 2: request a selected path and compare its exact cells.
//   - Stage 3: recompute that witness objective independently from the local surface.
//
// Behavior highlights:
//   - The optimal path goes through (1,1), not the near-tie (0,1).
//
// Inputs:
//   - Fixed, finite, non-negative costs and zero penalty.
//
// Returns:
//   - Test failure if path objective exceeds the returned optimum.
//
// Errors:
//   - Any path reconstruction error is unexpected.
//
// Determinism:
//   - Strict minimum, diagonal/up/left only for actual numeric ties.
//
// Complexity:
//   - Time O(6) DP cells, Space O(6) fixture size.
//
// Notes:
//   - No modification of the user-facing tie option is required.
//
// AI-Hints:
//   - Never compare predecessor costs using approximate equality.
//   - Do not reconstruct from (current-local), which loses rounding information.
func TestAlign_StrictTieDoesNotPickMoreExpensivePath(t *testing.T) {
	local := regressionDense(t, 2, 3,
		0, 5e-10, 20,
		20, 0, 0,
	)
	res, err := dtw.AlignCostMatrix(local, dtw.WithReturnPath(true))
	if err != nil {
		t.Fatalf("AlignCostMatrix: %v", err)
	}
	want := dtw.Path{{I: 0, J: 0}, {I: 1, J: 1}, {I: 1, J: 2}}
	if res.Distance != 0 {
		t.Fatalf("distance=%g, want zero", res.Distance)
	}
	mustPathEqual(t, res.Path, want, "strict tie path")
}

// TestAlign_PathCancellationRoundoff1e16 checks scalar callback path ownership and rounding.
// A loss of low bits in the accumulated cost must not change the stored
// predecessor or cause callback re-evaluation during backtracking.
//
// Implementation:
//   - Stage 1: construct scalar labels and a pure asymmetric cost callback.
//   - Stage 2: align with path tracking enabled.
//   - Stage 3: compare the selected diagonal witness and finite rounded objective.
//
// Behavior highlights:
//   - The scalar and precomputed-cost adapters select the same diagonal witness.
//
// Inputs:
//   - Two two-element scalar sequences with costs 1 and 1e16.
//
// Returns:
//   - Test failure on incomplete or non-optimal path.
//
// Errors:
//   - ErrIncompletePath is forbidden for a reachable finite final state.
//
// Determinism:
//   - Fixed row-major callback evaluation and predecessor tie order.
//
// Complexity:
//   - Time O(4) DP cells, Space O(4) path buffer.
//
// Notes:
//   - Callback is pure and does not depend on mutable business state.
//
// AI-Hints:
//   - The accumulated rounded distance 1e16 does not justify reverse subtraction.
//   - Preserve the actual choice made during forward DP.
func TestAlign_PathCancellationRoundoff1e16(t *testing.T) {
	cost := func(a, b float64) (float64, error) {
		if a == 0 && b == 0 {
			return 1, nil
		}
		return 1e16, nil
	}
	res, err := dtw.Align([]float64{0, 1}, []float64{0, 1}, dtw.WithReturnPath(true), dtw.WithCostFunc(cost))
	if err != nil || res == nil || res.Distance != 1e16 {
		t.Fatalf("result=%+v err=%v; want distance 1e16", res, err)
	}
	want := dtw.Path{{I: 0, J: 0}, {I: 1, J: 1}}
	mustPathEqual(t, res.Path, want, "cancellation path")
}

// TestAlign_CallbackOnlyAdmittedCellsWithoutLocalArtifact checks windowed evaluation.
// When LocalCost is not requested, the scalar kernel must not invoke
// CostFunc on structurally excluded cells or during path reconstruction.
//
// Implementation:
//   - Stage 1: configure a strict diagonal window and a pure counted callback.
//   - Stage 2: run with path tracking but no full local artifact.
//   - Stage 3: compare the count to the number of admitted cells.
//
// Behavior highlights:
//   - Exactly three cells are admitted in a 3-by-3 diagonal window.
//
// Inputs:
//   - Two identical three-element sequences and Window=0.
//
// Returns:
//   - Test failure if off-band or backtrack callbacks are invoked.
//
// Errors:
//   - Any error for the finite diagonal alignment fails the test.
//
// Determinism:
//   - Counted calls occur in ascending diagonal order.
//
// Complexity:
//   - DP scans O(9) cells, while CostFunc is evaluated O(3) times.
//
// Notes:
//   - The counted callback is used solely to observe frequency, not output order.
//
// AI-Hints:
//   - Do not evaluate full local costs without an explicit artifact request.
//   - Do not recompute local costs when reconstructing the path.
func TestAlign_CallbackOnlyAdmittedCellsWithoutLocalArtifact(t *testing.T) {
	calls := 0
	res, err := dtw.Align([]float64{0, 1, 2}, []float64{0, 1, 2},
		dtw.WithWindow(0), dtw.WithReturnPath(true),
		dtw.WithCostFunc(func(a, b float64) (float64, error) {
			calls++
			return math.Abs(a - b), nil
		}),
	)
	if err != nil || res == nil || !res.Reachable || calls != 3 {
		t.Fatalf("result=%+v err=%v calls=%d, want 3", res, err, calls)
	}
}

// TestAlign_AccumulatedArtifactOverflowCellDoesNotInvalidateFiniteOptimum protects representation semantics.
// A full accumulated-cost matrix may contain +Inf at overflowing intermediate
// states even when the selected terminal optimum is finite and reachable.
//
// Implementation:
//   - Stage 1: set expensive off-diagonal local costs and slope penalties.
//   - Stage 2: explicitly request the accumulated artifact.
//   - Stage 3: assert finite diagonal terminal optimum and +Inf intermediate cells.
//
// Behavior highlights:
//   - Public +Inf artifact values require contextual interpretation.
//
// Inputs:
//   - Fixed 2x2 local costs and a positive slope penalty.
//
// Returns:
//   - Test failure on spurious overflow error or lost artifact state.
//
// Errors:
//   - No final overflow error is expected.
//
// Determinism:
//   - Fixed diagonal optimum and row-major artifact positions.
//
// Complexity:
//   - Time O(4), artifact Space O(9) including boundary cells.
//
// Notes:
//   - There is intentionally no new public state-classification field in Result.
//
// AI-Hints:
//   - Do not read every accumulated +Inf as structural absence.
//   - Keep overflow information private unless a separately versioned artifact is designed.
func TestAlign_AccumulatedArtifactOverflowCellDoesNotInvalidateFiniteOptimum(t *testing.T) {
	local := regressionDense(t, 2, 2, 1, 1e308, 1e308, 1)
	res, err := dtw.AlignCostMatrix(local, dtw.WithSlopePenalty(1e308), dtw.WithReturnAccumulated(true))
	if err != nil || res == nil || !res.Reachable || res.Distance != 2 || res.Accumulated == nil {
		t.Fatalf("result=%+v err=%v, want finite 2 with accumulated artifact", res, err)
	}
	for _, ij := range [][2]int{{1, 2}, {2, 1}} {
		value, readErr := res.Accumulated.At(ij[0], ij[1])
		if readErr != nil || !math.IsInf(value, 1) {
			t.Fatalf("accumulated[%d,%d]=%g err=%v, want +Inf", ij[0], ij[1], value, readErr)
		}
	}
}

// TestAlign_LocalArtifactRejectsInvalidOffBandCost verifies full artifact validation.
// An explicitly published LocalCost matrix cannot hide an invalid off-band
// callback cost merely because the optimal DTW path would not visit that cell.
//
// Implementation:
//   - Stage 1: establish a finite strict diagonal alignment.
//   - Stage 2: return +Inf on one otherwise excluded cell while requesting LocalCost.
//   - Stage 3: demand a nil result with the original numeric error identity.
//
// Behavior highlights:
//   - The artifact contract is stronger than the admitted-window recurrence domain.
//
// Inputs:
//   - 2-by-2 scalar samples and an explicit finite-only local callback.
//
// Returns:
//   - Test failure if a partial/invalid artifact is published.
//
// Errors:
//   - ErrNaNInf for the invalid local cost, independent of DP reachability.
//
// Determinism:
//   - Full local costs are visited once each in row-major order until failure.
//
// Complexity:
//   - O(4) callback evaluations at most; O(4) artifact capacity.
//
// Notes:
//   - No assertion is made about callback calls after the first failure.
//
// AI-Hints:
//   - Never hide invalid off-band cells in a full published local artifact.
//   - Do not return the otherwise finite DP result after artifact failure.
func TestAlign_LocalArtifactRejectsInvalidOffBandCost(t *testing.T) {
	res, err := dtw.Align([]float64{0, 1}, []float64{0, 1}, dtw.WithWindow(0),
		dtw.WithReturnLocalCost(true), dtw.WithCostFunc(func(a, b float64) (float64, error) {
			if a == 0 && b == 1 {
				return math.Inf(1), nil
			}
			return 0, nil
		}))
	if res != nil || !errors.Is(err, dtw.ErrNaNInf) {
		t.Fatalf("result=%+v err=%v, want nil result and ErrNaNInf", res, err)
	}
}
