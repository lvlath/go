// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols_test exercises public OLS failure, cancellation, and numeric boundaries.
// Every fitted model is obtained through OLS, and all matrix access stays public.
package ols_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ols"
	"github.com/lvlath/go/stats"
)

// TestOLSFitCancellationAtEveryCheckpoint verifies cancellation before and after
// training-scale fitting, design allocation, QR work, and residual verification.
//
// Scenario: an overdetermined, full-rank, two-feature calibration batch fits in
// all four intercept/scaling modes. Every observed context poll is replayed
// with real cancellation; no partially prepared model may become visible.
// Complexity: O(C*T) per mode, with C observed polls and a fixed-size fit cost T.
func TestOLSFitCancellationAtEveryCheckpoint(t *testing.T) {
	// Keep a small non-collinear matrix and finite, nontrivial responses.
	x := mustDense(t, [][]float64{{1, 0}, {0, 1}, {2, 1}, {1, 3}})
	y := []float64{7, 8, 13, 16}
	for _, tc := range []struct {
		name string
		opts ols.Options
	}{
		{"centered", ols.Options{}},
		{"centered_population_scaled", ols.Options{Standardize: true}},
		{"through_origin", ols.Options{NoIntercept: true}},
		{"through_origin_scale_only", ols.Options{NoIntercept: true, Standardize: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The closure exposes only public result presence and error identity.
			exerciseCancellation(t, func(ctx context.Context) (bool, error) {
				result, err := ols.Fit(ctx, x, y, tc.opts)
				return result != nil, err
			})
		})
	}
}

// TestOLSFitReadFaultsAtEveryStage verifies error propagation from a custom Matrix
// after successful initial validation, not just on the first input read.
//
// Scenario: observations are stable but a storage-backed accessor can fail during
// centering, stats preprocessing, design assembly, or post-fit SSE. Replaying
// all observed read ordinals verifies root identity and atomic publication.
// Complexity: O(R*T) per mode for R reads and a fixed-size fit cost T.
func TestOLSFitReadFaultsAtEveryStage(t *testing.T) {
	// The fixture has full affine and origin-constrained column rank.
	data := [][]float64{{1, 0}, {0, 1}, {2, 1}, {1, 3}}
	x := mustDense(t, data)
	y := []float64{7, 8, 13, 16}
	for _, tc := range []struct {
		name string
		opts ols.Options
	}{
		{"centered", ols.Options{}},
		{"centered_population_scaled", ols.Options{Standardize: true}},
		{"through_origin", ols.Options{NoIntercept: true}},
		{"through_origin_scale_only", ols.Options{NoIntercept: true, Standardize: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Only At may fail; dimensions and all successful reads are unchanged.
			exerciseReadFaults(t, x, func(input matrix.Matrix) (bool, error) {
				result, err := ols.Fit(context.Background(), input, y, tc.opts)
				return result != nil, err
			})
		})
	}
	// Failed calls must not leave the original caller-owned source changed.
	mustMatrixValues(t, x, data)
}

// TestOLSPredictCancellationAtEveryCheckpoint exercises both numerical paths
// and the final publication check, including a valid zero-row batch.
//
// Scenario: a previously fitted identity model accepts a normal row and a row
// with a tiny surviving term between large opposing terms. A late cancellation
// must discard the entire batch, including any already calculated prefix.
// Complexity: O(C*T) for each fixed-size inference batch.
func TestOLSPredictCancellationAtEveryCheckpoint(t *testing.T) {
	model := mustUnitSlopeModel(t)
	// A shape-preserving empty matrix exercises publication without row loops.
	empty, err := matrix.NewDense(0, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name  string
		input matrix.Matrix
	}{
		{"ordinary", mustDense(t, [][]float64{{1, 2, 3}, {4, 5, 6}})},
		{"exact_fallback", mustDense(t, [][]float64{{1, 2, 3}, {1e16, 1, -1e16}})},
		{"empty_batch", empty},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The first successful run observes the actual poll sequence.
			exerciseCancellation(t, func(ctx context.Context) (bool, error) {
				predictions, err := model.Predict(ctx, tc.input)
				return predictions != nil, err
			})
		})
	}
}

// TestOLSPredictReadFaultsAtEveryStage checks faults during validation, the
// compensated dot product, and the exact fallback's rereading of a stable row.
//
// Scenario: the first prediction is ordinary, while the second needs exact
// cancellation recovery. An error from any read must reject the entire output.
// Complexity: O(R*T) for the fixed two-row batch, including exact rereads.
func TestOLSPredictReadFaultsAtEveryStage(t *testing.T) {
	model := mustUnitSlopeModel(t)
	data := [][]float64{{1, 2, 3}, {1e16, 1, -1e16}}
	x := mustDense(t, data)
	// Every observed access is independently faulted without changing the source.
	exerciseReadFaults(t, x, func(input matrix.Matrix) (bool, error) {
		predictions, err := model.Predict(context.Background(), input)
		return predictions != nil, err
	})
	mustMatrixValues(t, x, data)
	// The same model and source still predict correctly after all failed calls.
	predictions, err := model.Predict(context.Background(), x)
	if err != nil || len(predictions) != 2 || predictions[0] != 6 || predictions[1] != 1 {
		t.Fatalf("prediction after read failures: %v, %v; want [6 1]", predictions, err)
	}
}

// TestOLSPredictDefensiveNonFiniteReread verifies that a hostile custom backend
// cannot turn a nonfinite cell into a successful prediction after preflight.
//
// Scenario: the underlying matrix is stable and finite, but a synthetic accessor
// substitutes NaN during a selected reread. This tests defensive error handling;
// it does not declare concurrent input mutation legal or promise snapshot isolation.
// Complexity: O(p) for the ordinary row; the exact row adds rational arithmetic.
func TestOLSPredictDefensiveNonFiniteReread(t *testing.T) {
	model := mustUnitSlopeModel(t)
	for _, tc := range []struct {
		name          string
		features      []float64
		completeReads int
		invalid       float64
	}{
		// One complete preflight precedes the ordinary dot product.
		{"ordinary_nan", []float64{1, 2, 3}, 1, math.NaN()},
		// Severe cancellation needs a complete compensated pass before exact reread.
		{"exact_positive_inf", []float64{1e16, 1, -1e16}, 2, math.Inf(1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source := mustDense(t, [][]float64{tc.features})
			// Derive the first read of the selected pass from the fixture width.
			probe := &readProbeMatrix{
				Matrix:     source,
				faultAt:    tc.completeReads*source.Cols() + 1,
				faultValue: tc.invalid,
			}
			predictions, err := model.Predict(context.Background(), probe)
			if predictions != nil {
				t.Fatalf("invalid backend published predictions: %v", predictions)
			}
			mustErrorIs(t, err, ols.ErrNonFinite)
			mustMatrixValues(t, source, [][]float64{tc.features})
		})
	}
}

// TestOLSPredictOverflowAfterCompensation targets a different range boundary
// from overflowing a product or the rounded sum of absolute terms.
//
// Scenario: half of the overflow midpoint is added twice to MaxFloat64. Each
// individual addition rounds to MaxFloat64; only the accumulated correction
// exposes the final overflow. The independent rational oracle verifies both
// the representable control and the overflowing response in each sign.
// Complexity: O(p) ordinary work and exceptional exact arithmetic, p=3 here.
func TestOLSPredictOverflowAfterCompensation(t *testing.T) {
	model := mustUnitSlopeModel(t)
	// At MaxFloat64, one ULP is 2^971; the overflow midpoint is 2^970 above it.
	// A 2^969 term is below that midpoint, but two such terms reach the tie.
	const halfOverflowMidpoint = 0x1p969
	for _, tc := range []struct {
		name string
		sign float64
	}{
		{"positive", 1},
		{"negative", -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			limit := tc.sign * math.MaxFloat64
			increment := tc.sign * halfOverflowMidpoint
			control := []float64{limit, increment, 0}
			overflow := []float64{limit, increment, increment}
			// Check the mathematical fixture independently of the fitted implementation.
			if want := exactOLSResponse(t, model, control); want != limit {
				t.Fatalf("representable control oracle=%g, want %g", want, limit)
			}
			if want := exactOLSResponse(t, model, overflow); !math.IsInf(want, 0) || math.Signbit(want) != math.Signbit(limit) {
				t.Fatalf("overflow oracle=%g, want signed infinity", want)
			}
			predictions, err := model.Predict(context.Background(), mustDense(t, [][]float64{control}))
			if err != nil || len(predictions) != 1 || predictions[0] != limit {
				t.Fatalf("finite boundary prediction: %v, %v; want [%g]", predictions, err, limit)
			}
			// Include a valid prefix; true overflow may not publish any prefix values.
			predictions, err = model.Predict(context.Background(), mustDense(t, [][]float64{control, overflow}))
			if predictions != nil {
				t.Fatalf("overflow published a partial batch: %v", predictions)
			}
			mustErrorIs(t, err, ols.ErrOverflow)
		})
	}
}

// TestOLSFitDependencyRangeErrors distinguishes three independent failure stages.
//
// Scenario: a positive population scale underflows to zero; a QR coefficient
// exceeds the finite range; and an otherwise solvable centered design requires
// an unrepresentable physical intercept. The first two preserve dependency
// sentinels, whereas the last is classified at the OLS conversion boundary.
// Complexity: fixed-size fits with no external data or random inputs.
func TestOLSFitDependencyRangeErrors(t *testing.T) {
	t.Run("population_scale_underflow", func(t *testing.T) {
		// Population sigma=sqrt(2)/3 * minSubnormal: strictly positive, but it
		// rounds to zero. It is not a constant column and cannot define a scale.
		x := mustDense(t, [][]float64{{0}, {0}, {math.SmallestNonzeroFloat64}})
		result, err := ols.FitThroughOriginStandardized(context.Background(), x, []float64{0, 0, 0})
		if result != nil {
			t.Fatal("unrepresentable positive scale published a fitted model")
		}
		mustErrorIs(t, err, ols.ErrOverflow)
		mustErrorIs(t, err, stats.ErrOverflow)
	})

	t.Run("qr_coefficient_overflow", func(t *testing.T) {
		// A one-dimensional design with condition one admits the finite control
		// beta=MaxFloat64. Doubling its RHS requires beta=2*MaxFloat64 instead.
		x := mustDense(t, [][]float64{{0.5}})
		control, err := ols.FitThroughOrigin(context.Background(), x, []float64{math.MaxFloat64 / 2})
		if err != nil || control == nil || control.Model == nil {
			t.Fatalf("finite QR control failed: %v, %v", control, err)
		}
		beta, intercept := control.Model.Coefficients()
		if len(beta) != 1 || beta[0] != math.MaxFloat64 || intercept != 0 || control.ResidualSumSquares != 0 {
			t.Fatalf("finite QR control: beta=%v b=%g SSE=%g", beta, intercept, control.ResidualSumSquares)
		}
		result, err := ols.FitThroughOrigin(context.Background(), x, []float64{math.MaxFloat64})
		if result != nil {
			t.Fatal("unrepresentable QR coefficient published a fitted model")
		}
		mustErrorIs(t, err, ols.ErrOverflow)
		mustErrorIs(t, err, matrix.ErrOverflow)
	})

	t.Run("physical_intercept_overflow", func(t *testing.T) {
		// Center x=[2,4] at mean 3. The centered QR solution has theta0=0
		// and theta1=2^1023, both finite. The required original intercept
		// b=-3*2^1023 is genuinely outside binary64, independently of QR rounding.
		const magnitude = 0x1p1023
		x := mustDense(t, [][]float64{{2}, {4}})
		result, err := ols.FitDefault(context.Background(), x, []float64{-magnitude, magnitude})
		if result != nil {
			t.Fatal("unrepresentable physical intercept published a fitted model")
		}
		mustErrorIs(t, err, ols.ErrOverflow)
		// A matrix.ErrOverflow would mean the QR stage failed too early and
		// would not demonstrate the intended coefficient-conversion boundary.
		if errors.Is(err, matrix.ErrOverflow) || errors.Is(err, stats.ErrOverflow) {
			t.Fatalf("overflow came from a dependency rather than OLS conversion: %v", err)
		}
	})
}

// TestOLSFitResidualSubtractionOverflow separates an unrepresentable physical
// residual from an overflowing QR residual norm or coefficient conversion.
//
// Scenario: a five-observation origin-constrained fit has one very large
// response opposite the model prediction, and four exactly representable
// calibration responses. Its exact least-squares slope rounds to -1.5.
// The QR diagnostic uses exact products, but SSE uses the published rounded
// predictions. The first rounded residual crosses the float64 overflow midpoint.
// Complexity: a fixed-size QR fit plus the independent rational test oracle.
func TestOLSFitResidualSubtractionOverflow(t *testing.T) {
	const (
		overflowMidpointDelta = 0x1p970
		calibrationFeature    = 0x1p1023
		physicalSlope         = -1.5
	)
	// Division rounds this feature down. Multiplying it by -1.5 rounds to
	// -2^970 even though the exact product is slightly closer to zero.
	first := overflowMidpointDelta / -physicalSlope
	data := [][]float64{{first}, {calibrationFeature}, {calibrationFeature}, {calibrationFeature}, {calibrationFeature}}
	calibrated := physicalSlope * calibrationFeature
	y := []float64{math.MaxFloat64, calibrated, calibrated, calibrated, calibrated}

	// Reuse the independent exact-rational training oracle rather than assuming
	// that an arbitrarily selected extreme design has the desired coefficient.
	coefficients, sse := rationalFitOracle(t, data, y, true)
	if coefficients[0] != physicalSlope || !math.IsInf(sse, 1) {
		t.Fatalf("invalid residual boundary fixture: beta=%v exact SSE=%g", coefficients, sse)
	}
	// FMA recovers the signed rounding residual of the first product: the
	// exact QR residual is below the overflow midpoint, while subtraction
	// using the rounded physical prediction is precisely at that midpoint.
	prediction := first * physicalSlope
	productError := math.FMA(first, physicalSlope, -prediction)
	if prediction != -overflowMidpointDelta || productError <= 0 || !math.IsInf(y[0]-prediction, 1) {
		t.Fatalf("fixture did not straddle the residual boundary: prediction=%g productError=%g", prediction, productError)
	}

	// All training values remain stable. A successful QR stage is insufficient
	// to publish a Model when its own original-coordinate SSE is unrepresentable.
	result, err := ols.FitThroughOrigin(context.Background(), mustDense(t, data), y)
	if result != nil {
		t.Fatal("unrepresentable physical residual published a fitted model")
	}
	mustErrorIs(t, err, ols.ErrOverflow)
	if errors.Is(err, matrix.ErrOverflow) || errors.Is(err, stats.ErrOverflow) {
		t.Fatalf("overflow occurred before physical residual verification: %v", err)
	}
}
