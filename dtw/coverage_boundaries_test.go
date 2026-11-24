// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies bounded alignment and numeric lower-bound contracts.
// The cases distinguish input rejection, runtime errors and successful result statuses.
// AI-Hints:
//   - A cutoff cannot turn an overflow or a cancellation into a successful result.
//   - The diagnostic LB_Keogh value is not a substitute for the certified pruning kernel.
package dtw_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
)

// TestCoverageAlignBounded_ValidationPrecedence verifies the public validation order.
// Context and option errors precede scalar input validation, which precedes limit validation.
// Implementation:
//   - Stage 1: prepare independent invalid context, option, input and limit fixtures.
//   - Stage 2: combine invalid values to establish the applicable first failure.
//   - Stage 3: require nil result and the expected errors.Is identity.
//
// Complexity: Time and Space O(1) for each one-cell fixture.
// AI-Hints: Do not conflate an invalid limit with a valid above-limit alignment.
func TestCoverageAlignBounded_ValidationPrecedence(t *testing.T) {
	ctx := context.Background()
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	valid := []float64{1}
	for _, tc := range []struct {
		name  string
		ctx   context.Context
		a, b  []float64
		limit float64
		opts  []dtw.Option
		want  error
	}{
		{"nil-context", nil, nil, nil, -1, nil, dtw.ErrInvalidOption},
		{"canceled-context", canceled, nil, nil, -1, []dtw.Option{nil}, context.Canceled},
		{"nil-option-before-input", ctx, nil, nil, -1, []dtw.Option{nil}, dtw.ErrNilOption},
		{"invalid-window", ctx, valid, valid, 0, []dtw.Option{dtw.WithWindow(-2)}, dtw.ErrInvalidWindow},
		{"nil-input-before-limit", ctx, nil, valid, -1, nil, dtw.ErrNilInput},
		{"empty-input", ctx, []float64{}, valid, 0, nil, dtw.ErrEmptyInput},
		{"nonfinite-input-before-limit", ctx, valid, []float64{math.Inf(1)}, -1, nil, dtw.ErrNaNInf},
		{"negative-limit", ctx, valid, valid, -1, nil, dtw.ErrInvalidOption},
		{"negative-infinity-limit", ctx, valid, valid, math.Inf(-1), nil, dtw.ErrInvalidOption},
		{"nan-limit", ctx, valid, valid, math.NaN(), nil, dtw.ErrInvalidOption},
		{"unsupported-accumulated-artifact", ctx, valid, valid, 0, []dtw.Option{dtw.WithReturnAccumulated(true)}, dtw.ErrInvalidOption},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := dtw.AlignBounded(tc.ctx, tc.a, tc.b, tc.limit, tc.opts...)
			mustErrorIs(t, err, tc.want, tc.name)
			if got != nil {
				t.Fatalf("invalid request published %+v", got)
			}
		})
	}
}

// TestCoverageAlignBounded_CancellationAfterCompute checks deterministic cancellation.
// The one-cell callback returns a valid cost while triggering a real context cancellation.
// Implementation: cancel during the only local evaluation, then require nil and context.Canceled.
// Complexity: Time O(1), Space O(1); no sleeps, deadline races or worker goroutines.
// Notes: The callback side effect is a test-only cancellation trigger, not a cost-order contract.
func TestCoverageAlignBounded_CancellationAfterCompute(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	got, err := dtw.AlignBounded(ctx, []float64{1}, []float64{2}, math.Inf(1),
		dtw.WithCostFunc(func(_, _ float64) (float64, error) {
			cancel()
			return 1, nil
		}))
	mustErrorIs(t, err, context.Canceled, "cancellation before publication")
	if got != nil {
		t.Fatalf("canceled alignment published %+v", got)
	}
}

// TestCoverageAlignBounded_CallbackFailurePreservesIdentity protects the runtime error boundary.
// A failed cost source must not publish either an exact distance or an above-limit status.
// Implementation: inject a sentinel from a one-cell cost function and inspect the returned error.
// Complexity: Time O(1), Space O(1).
func TestCoverageAlignBounded_CallbackFailurePreservesIdentity(t *testing.T) {
	cause := errors.New("local cost model unavailable")
	got, err := dtw.AlignBounded(context.Background(), []float64{1}, []float64{2}, 0,
		dtw.WithCostFunc(func(_, _ float64) (float64, error) { return 0, cause }))
	mustErrorIs(t, err, cause, "callback runtime failure")
	if got != nil {
		t.Fatalf("failed callback published %+v", got)
	}
}

// TestCoverageAlignBounded_AdmittedCellsOracle checks rectangular bands and endpoint reachability.
// The oracle enumerates each zero-based coordinate independently of production row-bound arithmetic.
// Implementation:
//   - Stage 1: enumerate small rectangular shapes and clipped/unbounded radii.
//   - Stage 2: count all cells satisfying the mathematical band predicate.
//   - Stage 3: compare CellsVisited, reachability status and the zero-cost objective.
//
// Determinism: nested ascending loops with zero-valued inputs require no random seed.
// Complexity: Time O(S^4*R) across shapes up to S and R radii; each run uses O(S) DP memory.
// AI-Hints: Do not replace the oracle with the production row-interval formula.
func TestCoverageAlignBounded_AdmittedCellsOracle(t *testing.T) {
	const maxSide = 6
	for n := 1; n <= maxSide; n++ {
		for m := 1; m <= maxSide; m++ {
			for _, radius := range []int{-1, 0, 1, 2, math.MaxInt} {
				var wantCells uint64
				// Count admitted cells directly; rows beyond the band may contain zero cells.
				for i := 0; i < n; i++ {
					for j := 0; j < m; j++ {
						if radius < 0 || (i-j <= radius && j-i <= radius) {
							wantCells++
						}
					}
				}
				got, err := dtw.AlignBounded(context.Background(), make([]float64, n), make([]float64, m), 0, dtw.WithWindow(radius))
				mustNoError(t, err, "zero-cost rectangular bounded alignment")
				if got == nil || got.CellsVisited != wantCells {
					t.Fatalf("shape=%dx%d radius=%d: result=%+v want cells=%d", n, m, radius, got, wantCells)
				}
				// A monotone path exists exactly when the band admits the terminal coordinate.
				if radius >= 0 && (n-m > radius || m-n > radius) {
					if got.Status != dtw.BoundUnreachable || !math.IsInf(got.Distance, 1) || !math.IsInf(got.LowerBound, 1) || got.Path != nil {
						t.Fatalf("shape=%dx%d radius=%d: expected structural no-path; got %+v", n, m, radius, got)
					}
				} else if got.Status != dtw.BoundExact || got.Distance != 0 || got.LowerBound != 0 || got.Path != nil {
					t.Fatalf("shape=%dx%d radius=%d: expected exact zero; got %+v", n, m, radius, got)
				}
			}
		}
	}
}

// TestCoverageLBKeoghSquared_ValidationAndOverflow verifies each public numeric failure family.
// A finite local deviation can overflow during squaring, while finite local squares can overflow
// only when their accumulated sum exceeds the supported binary64 range.
// Implementation:
//   - Stage 1: construct nil, empty, mismatched, dirty and reversed-envelope cases.
//   - Stage 2: construct two independent overflow cases using exact powers of two.
//   - Stage 3: check nil-error finite behavior separately from error-classified results.
//
// Complexity: Time O(L), Space O(1) inside LBKeoghSquared, with constant-sized fixtures.
// AI-Hints: Use 2^512 for square overflow and four 2^511 deviations for sum overflow.
func TestCoverageLBKeoghSquared_ValidationAndOverflow(t *testing.T) {
	zero := []float64{0}
	one := []float64{1}
	localOverflow := math.Ldexp(1, 512)
	finiteDeviation := math.Ldexp(1, 511)
	for _, tc := range []struct {
		name                string
		query, lower, upper []float64
		want                []error
	}{
		{"nil-query", nil, zero, one, []error{dtw.ErrNilInput}},
		{"nil-lower", zero, nil, one, []error{dtw.ErrNilInput}},
		{"nil-upper", zero, zero, nil, []error{dtw.ErrNilInput}},
		{"empty", []float64{}, []float64{}, []float64{}, []error{dtw.ErrEmptyInput}},
		{"lower-shape", zero, []float64{}, one, []error{dtw.ErrShape}},
		{"upper-shape", zero, zero, []float64{}, []error{dtw.ErrShape}},
		{"nan-query", []float64{math.NaN()}, zero, one, []error{dtw.ErrNonFinite, dtw.ErrNaNInf}},
		{"positive-inf-lower", zero, []float64{math.Inf(1)}, one, []error{dtw.ErrNonFinite, dtw.ErrNaNInf}},
		{"negative-inf-upper", zero, zero, []float64{math.Inf(-1)}, []error{dtw.ErrNonFinite, dtw.ErrNaNInf}},
		{"reversed-envelope", zero, one, zero, []error{dtw.ErrShape}},
		{"local-square-overflow", []float64{localOverflow}, zero, zero, []error{dtw.ErrOverflow, dtw.ErrNaNInf}},
		{"difference-overflow", []float64{math.MaxFloat64}, []float64{-math.MaxFloat64}, []float64{-math.MaxFloat64}, []error{dtw.ErrOverflow, dtw.ErrNaNInf}},
		{"sum-overflow", []float64{finiteDeviation, finiteDeviation, finiteDeviation, finiteDeviation}, make([]float64, 4), make([]float64, 4), []error{dtw.ErrOverflow, dtw.ErrNaNInf}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := dtw.LBKeoghSquared(tc.query, tc.lower, tc.upper)
			for _, target := range tc.want {
				mustErrorIs(t, err, target, tc.name)
			}
		})
	}
	// A below-envelope, inside-envelope and above-envelope term contribute 1, 0 and 4.
	got, err := dtw.LBKeoghSquared([]float64{0, 3, 7}, []float64{1, 2, 4}, []float64{2, 4, 5})
	mustNoError(t, err, "finite envelope deviation")
	mustFloatEqual(t, got, 5, 0, "squared deviation sum")
}

// TestCoverageAlignBounded_AboveLimitSuppressesRequestedPath checks the outcome-specific witness.
// An internally computed path is published only for BoundExact, never for BoundAboveLimit.
// Implementation: compare the same unique diagonal alignment at below, equal and unbounded limits.
// Complexity: Time O(n*m), Space O(n*m) because explicit path tracking is requested.
// AI-Hints: A finite LowerBound above the limit is not a finite published Distance.
func TestCoverageAlignBounded_AboveLimitSuppressesRequestedPath(t *testing.T) {
	ctx := context.Background()
	a, b := []float64{0, 2}, []float64{0, 4}
	opts := []dtw.Option{dtw.WithWindow(0), dtw.WithReturnPath(true)}
	above, err := dtw.AlignBounded(ctx, a, b, 1, opts...)
	mustNoError(t, err, "above-limit alignment")
	if above == nil || above.Status != dtw.BoundAboveLimit || !math.IsNaN(above.Distance) ||
		above.LowerBound != 2 || above.Path != nil || above.CellsVisited != 2 {
		t.Fatalf("above-limit outcome leaked or lost state: %+v", above)
	}
	for _, limit := range []float64{2, math.Inf(1)} {
		exact, alignErr := dtw.AlignBounded(ctx, a, b, limit, opts...)
		mustNoError(t, alignErr, "inclusive or unlimited bound")
		if exact == nil || exact.Status != dtw.BoundExact || exact.Distance != 2 || exact.LowerBound != 2 || exact.CellsVisited != 2 {
			t.Fatalf("limit=%v: exact outcome mismatch: %+v", limit, exact)
		}
		mustPathEqual(t, exact.Path, dtw.Path{{I: 0, J: 0}, {I: 1, J: 1}}, "inclusive bounded witness")
	}
}
