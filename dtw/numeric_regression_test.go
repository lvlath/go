// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dtw_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

// regressionDense constructs a fully initialized, detached matrix fixture.
// It uses only the exported Matrix factory and Fill API, so independent tests
// do not rely on private row-major storage or production adapter helpers.
//
// Implementation:
//   - Stage 1: allocate a Dense matrix with the supplied dimensions.
//   - Stage 2: initialize all values in row-major order through Fill.
//   - Stage 3: fail the invoking test immediately if fixture construction fails.
//
// Inputs:
//   - t: active test context; rows, cols: matrix dimensions; values: row-major data.
//
// Returns:
//   - *matrix.Dense: independently allocated caller-owned fixture.
//
// Errors:
//   - Allocation and initialization errors fail t, not production test conditions.
//
// Determinism:
//   - Fixed row-major Fill semantics.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols).
//
// Notes:
//   - Does not mutate caller slices or invoke the DTW implementation.
//
// AI-Hints:
//   - Keep regression fixtures separate from the tested DTW recurrence.
//   - Never add test-only exports to the production package for fixture setup.
func regressionDense(t *testing.T, rows, cols int, values ...float64) *matrix.Dense {
	t.Helper()
	m, err := matrix.NewPreparedDense(rows, cols)
	if err != nil {
		t.Fatalf("NewPreparedDense: %v", err)
	}
	if err = m.Fill(values); err != nil {
		t.Fatalf("Fill: %v", err)
	}
	return m
}

// TestAlignMatrix_CancellationSquaredL2 verifies pre-subtraction cancellation and equal huge features.
// It protects a caller-visible numerical invariant rather than an implementation detail.
//
// Implementation:
//   - Stage 1: construct the fixed independent counterexample.
//   - Stage 2: invoke the published DTW facade using explicit options.
//   - Stage 3: compare result class, exact objective, and applicable artifact or witness.
//
// Behavior highlights:
//   - The fixture exercises pre-subtraction cancellation and equal huge features.
//
// Inputs:
//   - a one-cell squared-L2 local cost that must remain 1 or 0.
//
// Returns:
//   - Test failure if the documented contract is violated.
//
// Errors:
//   - Classifies numerical failures through errors.Is when errors are expected.
//
// Determinism:
//   - Uses fixed input order and exact finite classifications.
//
// Complexity:
//   - O(1) DP cells per subtest; no randomized execution.
//
// Notes:
//   - This regression does not reimplement production recurrence logic.
//
// AI-Hints:
//   - Do not weaken assertions to match historically incorrect results.
//   - Preserve the selected error and ownership laws across adapter refactors.
func TestAlignMatrix_CancellationSquaredL2(t *testing.T) {
	for _, tc := range []struct {
		name       string
		a, b, want float64
	}{
		{"near-large", 1e9, 1e9 + 1, 1},
		{"equal-huge", 1e200, 1e200, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			x := regressionDense(t, 1, 1, tc.a)
			y := regressionDense(t, 1, 1, tc.b)
			got, err := dtw.AlignMatrix(x, y)
			if err != nil {
				t.Fatalf("AlignMatrix: %v", err)
			}
			if got == nil || !got.Reachable || got.Distance != tc.want {
				t.Fatalf("result=%+v, want reachable distance %g", got, tc.want)
			}
		})
	}
}

// TestAlign_OverflowFinalMinimum verifies structural reachability despite an unrepresentable terminal optimum.
// It protects a caller-visible numerical invariant rather than an implementation detail.
//
// Implementation:
//   - Stage 1: construct the fixed independent counterexample.
//   - Stage 2: invoke the published DTW facade using explicit options.
//   - Stage 3: compare result class, exact objective, and applicable artifact or witness.
//
// Behavior highlights:
//   - The fixture exercises structural reachability despite an unrepresentable terminal optimum.
//
// Inputs:
//   - two mandatory 1e308 local costs with no alternative path.
//
// Returns:
//   - Test failure if the documented contract is violated.
//
// Errors:
//   - Classifies numerical failures through errors.Is when errors are expected.
//
// Determinism:
//   - Uses fixed input order and exact finite classifications.
//
// Complexity:
//   - O(2) DP cells; no randomized execution.
//
// Notes:
//   - This regression does not reimplement production recurrence logic.
//
// AI-Hints:
//   - Do not weaken assertions to match historically incorrect results.
//   - Preserve the selected error and ownership laws across adapter refactors.
func TestAlign_OverflowFinalMinimum(t *testing.T) {
	got, err := dtw.Align([]float64{0, 0}, []float64{0},
		dtw.WithCostFunc(func(_, _ float64) (float64, error) { return 1e308, nil }))
	if got != nil {
		t.Fatalf("overflow must not publish a result: %+v", got)
	}
	if !errors.Is(err, dtw.ErrOverflow) || !errors.Is(err, dtw.ErrNaNInf) {
		t.Fatalf("overflow must preserve both numeric identities: %v", err)
	}
}

// TestAlign_DiscardedOverflowKeepsFiniteOptimum verifies finite diagonal optimum despite overflow in discarded transitions.
// It protects a caller-visible numerical invariant rather than an implementation detail.
//
// Implementation:
//   - Stage 1: construct the fixed independent counterexample.
//   - Stage 2: invoke the published DTW facade using explicit options.
//   - Stage 3: compare result class, exact objective, and applicable artifact or witness.
//
// Behavior highlights:
//   - The fixture exercises finite diagonal optimum despite overflow in discarded transitions.
//
// Inputs:
//   - two-by-two scalar labels and a pure custom local-cost function.
//
// Returns:
//   - Test failure if the documented contract is violated.
//
// Errors:
//   - Classifies numerical failures through errors.Is when errors are expected.
//
// Determinism:
//   - Uses fixed input order and exact finite classifications.
//
// Complexity:
//   - O(4) DP cells; no randomized execution.
//
// Notes:
//   - This regression does not reimplement production recurrence logic.
//
// AI-Hints:
//   - Do not weaken assertions to match historically incorrect results.
//   - Preserve the selected error and ownership laws across adapter refactors.
func TestAlign_DiscardedOverflowKeepsFiniteOptimum(t *testing.T) {
	a, b := []float64{0, 1}, []float64{0, 1}
	got, err := dtw.Align(a, b,
		dtw.WithSlopePenalty(1e308),
		dtw.WithCostFunc(func(ai, bj float64) (float64, error) {
			if ai == bj {
				return 1, nil
			}
			return 1e308, nil
		}))
	if err != nil || got == nil || !got.Reachable || got.Distance != 2 {
		t.Fatalf("result=%+v err=%v; want finite optimum 2", got, err)
	}
}

// TestAlignCostMatrix_DiscardedOverflowKeepsFiniteOptimum verifies the same overflow law for the precomputed-cost adapter.
// It protects a caller-visible numerical invariant rather than an implementation detail.
//
// Implementation:
//   - Stage 1: construct the fixed independent counterexample.
//   - Stage 2: invoke the published DTW facade using explicit options.
//   - Stage 3: compare result class, exact objective, and applicable artifact or witness.
//
// Behavior highlights:
//   - The fixture exercises the same overflow law for the precomputed-cost adapter.
//
// Inputs:
//   - finite two-by-two local-cost Dense and an extreme slope penalty.
//
// Returns:
//   - Test failure if the documented contract is violated.
//
// Errors:
//   - Classifies numerical failures through errors.Is when errors are expected.
//
// Determinism:
//   - Uses fixed input order and exact finite classifications.
//
// Complexity:
//   - O(4) DP cells; no randomized execution.
//
// Notes:
//   - This regression does not reimplement production recurrence logic.
//
// AI-Hints:
//   - Do not weaken assertions to match historically incorrect results.
//   - Preserve the selected error and ownership laws across adapter refactors.
func TestAlignCostMatrix_DiscardedOverflowKeepsFiniteOptimum(t *testing.T) {
	local := regressionDense(t, 2, 2, 1, 1e308, 1e308, 1)
	got, err := dtw.AlignCostMatrix(local, dtw.WithSlopePenalty(1e308))
	if err != nil || got == nil || !got.Reachable || got.Distance != 2 {
		t.Fatalf("result=%+v err=%v; want finite optimum 2", got, err)
	}
}

// TestAlign_WindowUnreachableIsNotOverflow verifies structural no-path under strict diagonal window.
// It protects a caller-visible numerical invariant rather than an implementation detail.
//
// Implementation:
//   - Stage 1: construct the fixed independent counterexample.
//   - Stage 2: invoke the published DTW facade using explicit options.
//   - Stage 3: compare result class, exact objective, and applicable artifact or witness.
//
// Behavior highlights:
//   - The fixture exercises structural no-path under strict diagonal window.
//
// Inputs:
//   - different sequence lengths with Window=0.
//
// Returns:
//   - Test failure if the documented contract is violated.
//
// Errors:
//   - Classifies numerical failures through errors.Is when errors are expected.
//
// Determinism:
//   - Uses fixed input order and exact finite classifications.
//
// Complexity:
//   - O(6) grid cells; no randomized execution.
//
// Notes:
//   - This regression does not reimplement production recurrence logic.
//
// AI-Hints:
//   - Do not weaken assertions to match historically incorrect results.
//   - Preserve the selected error and ownership laws across adapter refactors.
func TestAlign_WindowUnreachableIsNotOverflow(t *testing.T) {
	got, err := dtw.Align([]float64{0, 1}, []float64{0, 1, 2}, dtw.WithWindow(0))
	if err != nil || got == nil || got.Reachable || !math.IsInf(got.Distance, 1) {
		t.Fatalf("result=%+v err=%v; want unreachable +Inf without error", got, err)
	}
}

// TestAlign_LocalCostFullOutsideBand verifies complete local matrix publication independently of the DP band.
// It protects a caller-visible numerical invariant rather than an implementation detail.
//
// Implementation:
//   - Stage 1: construct the fixed independent counterexample.
//   - Stage 2: invoke the published DTW facade using explicit options.
//   - Stage 3: compare result class, exact objective, and applicable artifact or witness.
//
// Behavior highlights:
//   - The fixture exercises complete local matrix publication independently of the DP band.
//
// Inputs:
//   - two-element sequences, Window=0, explicit LocalCost artifact.
//
// Returns:
//   - Test failure if the documented contract is violated.
//
// Errors:
//   - Classifies numerical failures through errors.Is when errors are expected.
//
// Determinism:
//   - Uses fixed input order and exact finite classifications.
//
// Complexity:
//   - O(4) local cells; no randomized execution.
//
// Notes:
//   - This regression does not reimplement production recurrence logic.
//
// AI-Hints:
//   - Do not weaken assertions to match historically incorrect results.
//   - Preserve the selected error and ownership laws across adapter refactors.
func TestAlign_LocalCostFullOutsideBand(t *testing.T) {
	got, err := dtw.Align([]float64{0, 10}, []float64{0, 10},
		dtw.WithWindow(0), dtw.WithReturnLocalCost(true))
	if err != nil || got == nil || got.LocalCost == nil {
		t.Fatalf("result=%+v err=%v", got, err)
	}
	for _, ij := range [][2]int{{0, 1}, {1, 0}} {
		v, err := got.LocalCost.At(ij[0], ij[1])
		if err != nil || v != 10 {
			t.Fatalf("local[%d,%d]=%g err=%v; want 10", ij[0], ij[1], v, err)
		}
	}
}

// TestAlign_CallbackPureSingleEvaluationPerPublishedCell verifies exactly one cost callback per published local cell.
// It protects a caller-visible numerical invariant rather than an implementation detail.
//
// Implementation:
//   - Stage 1: construct the fixed independent counterexample.
//   - Stage 2: invoke the published DTW facade using explicit options.
//   - Stage 3: compare result class, exact objective, and applicable artifact or witness.
//
// Behavior highlights:
//   - The fixture exercises exactly one cost callback per published local cell.
//
// Inputs:
//   - three-element series, strict band, full LocalCost and path.
//
// Returns:
//   - Test failure if the documented contract is violated.
//
// Errors:
//   - Classifies numerical failures through errors.Is when errors are expected.
//
// Determinism:
//   - Uses fixed input order and exact finite classifications.
//
// Complexity:
//   - O(9) callback evaluations; no randomized execution.
//
// Notes:
//   - This regression does not reimplement production recurrence logic.
//
// AI-Hints:
//   - Do not weaken assertions to match historically incorrect results.
//   - Preserve the selected error and ownership laws across adapter refactors.
func TestAlign_CallbackPureSingleEvaluationPerPublishedCell(t *testing.T) {
	calls := 0
	got, err := dtw.Align([]float64{0, 1, 2}, []float64{0, 1, 2},
		dtw.WithWindow(0), dtw.WithReturnPath(true), dtw.WithReturnLocalCost(true),
		dtw.WithCostFunc(func(a, b float64) (float64, error) {
			calls++
			return math.Abs(a - b), nil
		}))
	if err != nil || got == nil || !got.Reachable || got.LocalCost == nil {
		t.Fatalf("result=%+v err=%v", got, err)
	}
	if calls != 9 {
		t.Fatalf("callback calls=%d, want 9", calls)
	}
}

// TestAlignCostMatrix_PathCancellationRoundoff1e16 verifies correct forward predecessor witness under cancellation.
// It protects a caller-visible numerical invariant rather than an implementation detail.
//
// Implementation:
//   - Stage 1: construct the fixed independent counterexample.
//   - Stage 2: invoke the published DTW facade using explicit options.
//   - Stage 3: compare result class, exact objective, and applicable artifact or witness.
//
// Behavior highlights:
//   - The fixture exercises correct forward predecessor witness under cancellation.
//
// Inputs:
//   - two-by-two local matrix with initial 1 and large other costs.
//
// Returns:
//   - Test failure if the documented contract is violated.
//
// Errors:
//   - Classifies numerical failures through errors.Is when errors are expected.
//
// Determinism:
//   - Uses fixed input order and exact finite classifications.
//
// Complexity:
//   - O(4) DP cells; no randomized execution.
//
// Notes:
//   - This regression does not reimplement production recurrence logic.
//
// AI-Hints:
//   - Do not weaken assertions to match historically incorrect results.
//   - Preserve the selected error and ownership laws across adapter refactors.
func TestAlignCostMatrix_PathCancellationRoundoff1e16(t *testing.T) {
	local := regressionDense(t, 2, 2, 1, 1e16, 1e16, 1e16)
	got, err := dtw.AlignCostMatrix(local, dtw.WithReturnPath(true))
	if err != nil || got == nil || got.Distance != 1e16 {
		t.Fatalf("result=%+v err=%v; want distance 1e16", got, err)
	}
	want := dtw.Path{{I: 0, J: 0}, {I: 1, J: 1}}
	if len(got.Path) != len(want) {
		t.Fatalf("path=%v want=%v", got.Path, want)
	}
	for k := range want {
		if got.Path[k] != want[k] {
			t.Fatalf("path=%v want=%v", got.Path, want)
		}
	}
}
