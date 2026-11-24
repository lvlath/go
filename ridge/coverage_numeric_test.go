// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test verifies numerical range boundaries through the public API.
// Exact scalar and affine oracles distinguish final overflow from intermediate loss.
package ridge_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ridge"
	"github.com/lvlath/go/stats"
)

// TestRidge_ExtremeConstantColumnCenteringRegression preserves a representable
// mean even when independently rounded weighted contributions would overflow.
// Implementation:
//   - Stage 1: construct three identical, extreme finite observations.
//   - Stage 2: fit an unscaled affine model with positive regularization.
//   - Stage 3: verify the intercept-only analytical solution and original-unit SSE.
//
// Behavior highlights: a constant feature has exactly zero centered coordinates.
// Inputs: three identical positive or negative MaxFloat64 features; y=[2,4,6].
// Returns: no value; the optimum has beta=0, b=4, SSE=8, penalty=0.
// Errors: none; every required coefficient and diagnostic is representable.
// Determinism: positive fixture followed by its sign-reversed counterpart.
// Complexity: fixed 3-by-1 training input; O(1) oracle work.
// Notes: the response is deliberately nonconstant to verify residual diagnostics.
// AI-Hints: do not classify a finite convex combination as overflow merely because
// two separately rounded weighted terms overflow when added.
func TestRidge_ExtremeConstantColumnCenteringRegression(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		name  string
		value float64
	}{
		{"positive_max", math.MaxFloat64},
		{"negative_max", -math.MaxFloat64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Positive Lambda removes the unconstrained constant slope; the intercept
			// remains unpenalized and equals the ordinary response mean.
			x := dense(t, [][]float64{{tc.value}, {tc.value}, {tc.value}})
			fit, err := ridge.FitWithLambda(ctx, x, []float64{2, 4, 6}, 1)
			if err != nil {
				t.Fatal(err)
			}
			beta, intercept := fit.Model.Coefficients()
			if len(beta) != 1 || beta[0] != 0 {
				t.Fatalf("constant column must have zero slope, got %v", beta)
			}
			closeTo(t, intercept, 4)
			closeTo(t, fit.ResidualSumSquares, 8)
			if fit.Penalty != 0 {
				t.Fatalf("constant column penalty=%g, want zero", fit.Penalty)
			}
			closeTo(t, fit.Objective, 4.0/3.0)
			values, err := fit.Model.Predict(ctx, dense(t, [][]float64{{0}, {tc.value}, {-tc.value}}))
			if err != nil || len(values) != 3 {
				t.Fatalf("constant predictor: values=%v err=%v", values, err)
			}
			for _, value := range values {
				closeTo(t, value, 4)
			}
		})
	}
}

// TestRidge_OppositeSignExtremeCentering checks the complementary mean path.
// The raw difference between opposite finite extrema overflows, while their
// mean and centered observations remain representable.
// Implementation: fit symmetric extreme features and verify physical inference.
// Behavior highlights: the weighted mean fallback must not reject this design.
// Inputs: x=[-MaxFloat64,MaxFloat64], y=[-1,1], Lambda=1, unscaled affine fit.
// Returns: no value; a nonzero subnormal slope and a near-zero intercept.
// Errors: none; the accepted rank and all required diagnostics are finite.
// Determinism: two opposite-signed observations in fixed order.
// Complexity: fixed scalar fit and O(1) analytic oracle.
// Notes: the exact slope is M/(M*M+1), with M=MaxFloat64; its binary64
// rounding is 1/M. Never materialize M*M when checking that oracle.
// AI-Hints: preserve the weighted update when value-mean itself overflows.
func TestRidge_OppositeSignExtremeCentering(t *testing.T) {
	ctx := context.Background()
	maximum := math.MaxFloat64
	x := dense(t, [][]float64{{-maximum}, {maximum}})
	fit, err := ridge.FitWithLambda(ctx, x, []float64{-1, 1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	beta, intercept := fit.Model.Coefficients()
	want := 1 / maximum
	// A relative comparison is essential: an absolute epsilon would accept zero.
	if len(beta) != 1 || beta[0] == 0 || math.IsNaN(beta[0]) || math.Abs(beta[0]/want-1) > 1e-9 {
		t.Fatalf("extreme slope=%v want=%.17g", beta, want)
	}
	closeTo(t, intercept, 0)
	values, err := fit.Model.Predict(ctx, x)
	if err != nil || len(values) != 2 {
		t.Fatalf("extreme inference=%v err=%v", values, err)
	}
	closeTo(t, values[0], -1)
	closeTo(t, values[1], 1)
	closeTo(t, fit.ResidualSumSquares, 0)
	closeTo(t, fit.Objective, 0)
}

// TestRidge_TrainingScaleUnderflowPreservesStatsError checks that a nonconstant
// column whose population scale rounds to zero is not treated as constant.
// Implementation: fit both affine and origin-standardized models to [0,2^-1074].
// Behavior highlights: exact population standard deviation is 2^-1075, which
// rounds to zero under binary64 nearest-even; a positive scale cannot be stored.
// Inputs: two finite, distinct observations and ordinary finite responses.
// Returns: no value; both calls must return a nil FitResult.
// Errors: ridge.ErrOverflow and the underlying stats.ErrOverflow must both match.
// Determinism: two fixed model policies, no random sampling.
// Complexity: fixed scalar fixture and O(1) independent mathematical reasoning.
// Notes: UnitScaleConstant applies to true constants, not rounded-away variance.
// AI-Hints: never weaken this test by substituting a unit scale for a nonconstant column.
func TestRidge_TrainingScaleUnderflowPreservesStatsError(t *testing.T) {
	x := dense(t, [][]float64{{0}, {math.SmallestNonzeroFloat64}})
	for _, noIntercept := range []bool{false, true} {
		fit, err := ridge.Fit(context.Background(), x, []float64{1, 2}, ridge.Options{
			Lambda: 1, Standardize: true, NoIntercept: noIntercept,
		})
		if fit != nil || !errors.Is(err, ridge.ErrOverflow) || !errors.Is(err, stats.ErrOverflow) {
			t.Fatalf("origin=%t: fit=%v err=%v; want both overflow identities", noIntercept, fit, err)
		}
	}
}

// TestRidge_PhysicalCoefficientRange verifies that a finite standardized QR
// solution cannot publish an unrepresentable physical slope or affine intercept.
// Implementation:
//   - Stage 1: choose exact power-of-two scales and means with known optima.
//   - Stage 2: fit using population standardization and positive Lambda.
//   - Stage 3: reject conversion overflow without publishing a partial model.
//
// Behavior highlights: the dependency solve is finite in both fixtures; the
// physical-coordinate conversion, rather than training scale or QR rank, fails.
// Inputs: centered tiny features, and large-offset features with exact deviations.
// Returns: no value; both calls must return nil FitResult.
// Errors: ridge.ErrOverflow, not a scale, rank, or matrix-solver error.
// Determinism: power-of-two fixtures with exact real-number optima.
// Complexity: two fixed 2-by-1 fits.
// Notes: SSE is not a valid substitute for checking coefficient representability.
// AI-Hints: do not infer physical coefficient finiteness from finite QR coordinates.
func TestRidge_PhysicalCoefficientRange(t *testing.T) {
	ctx := context.Background()
	t.Run("physical_slope", func(t *testing.T) {
		// DDOF=0 scale is 2^-1022. Standardized features are +/-1.
		// At Lambda=1, theta=8 and beta=8/2^-1022=2^1025, outside binary64.
		scale := math.Ldexp(1, -1022)
		fit, err := ridge.FitThroughOriginStandardized(ctx,
			dense(t, [][]float64{{-scale}, {scale}}), []float64{-16, 16}, 1)
		if fit != nil || !errors.Is(err, ridge.ErrOverflow) || errors.Is(err, stats.ErrOverflow) || errors.Is(err, matrix.ErrOverflow) {
			t.Fatalf("physical slope overflow: fit=%v err=%v", fit, err)
		}
	})
	t.Run("physical_intercept", func(t *testing.T) {
		// Mean=2^900, scale=2^849, theta=2^999, beta=2^150.
		// The centered intercept is zero; physical b=-2^1050 cannot fit.
		mean := math.Ldexp(1, 900)
		scale := math.Ldexp(1, 849)
		response := math.Ldexp(1, 1000)
		fit, err := ridge.FitStandardized(ctx,
			dense(t, [][]float64{{mean - scale}, {mean + scale}}), []float64{-response, response}, 1)
		if fit != nil || !errors.Is(err, ridge.ErrOverflow) || errors.Is(err, stats.ErrOverflow) || errors.Is(err, matrix.ErrOverflow) {
			t.Fatalf("physical intercept overflow: fit=%v err=%v", fit, err)
		}
	})
}

// TestRidge_DiagnosticRangeBoundaries separates three mandatory output fields
// that can overflow even when the augmented QR solution and L2 norm are finite.
// Implementation:
//   - Stage 1: independently derive exact scalar SSE, penalty and objective.
//   - Stage 2: verify the expected representability of each diagnostic.
//   - Stage 3: call the public facade and require atomic ErrOverflow.
//
// Behavior highlights: an overflowing SSE must not be silently divided by n;
// individually finite SSE and penalty do not imply a finite objective.
// Inputs: one observation per fixture, origin-constrained unscaled positive-Lambda fit.
// Returns: no value; no diagnostic may be replaced with infinity or a saturated number.
// Errors: ridge.ErrOverflow without matrix.ErrOverflow or stats.ErrOverflow.
// Determinism: exact scalar oracle and fixed power-of-two data.
// Complexity: O(1) rational operations and constant-size production fits.
// Notes: the penalty-only case has enough margin to tolerate ordinary QR rounding.
// AI-Hints: do not square the response in float64 when preparing the oracle.
func TestRidge_DiagnosticRangeBoundaries(t *testing.T) {
	cases := []struct {
		name                                      string
		x, y, lambda                              float64
		finiteSSE, finitePenalty, finiteObjective bool
	}{
		// Residual=2^600, so its L2 norm is finite but its squared norm is not.
		{"sse", 0, math.Ldexp(1, 600), 1, false, true, false},
		// Near-interpolation: residual is small, but Lambda*beta^2/2 overflows.
		{"penalty", 1, math.Ldexp(1, 550), math.Ldexp(1, -65), true, false, false},
		// beta=2y/3; SSE=penalty=y^2/9; objective=y^2/6 > MaxFloat64.
		{"objective", 1, math.Ldexp(1.25, 513), 0.5, true, true, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Check the fixture using exact rational arithmetic before invoking Ridge.
			sse, penalty, objective := scalarOriginDiagnostics(tc.x, tc.y, tc.lambda)
			wantSSE, _ := sse.Float64()
			wantPenalty, _ := penalty.Float64()
			wantObjective, _ := objective.Float64()
			if (!math.IsInf(wantSSE, 0)) != tc.finiteSSE || (!math.IsInf(wantPenalty, 0)) != tc.finitePenalty || (!math.IsInf(wantObjective, 0)) != tc.finiteObjective {
				t.Fatalf("bad oracle fixture: SSE=%g penalty=%g objective=%g", wantSSE, wantPenalty, wantObjective)
			}
			fit, err := ridge.FitThroughOrigin(context.Background(), dense(t, [][]float64{{tc.x}}), []float64{tc.y}, tc.lambda)
			if fit != nil || !errors.Is(err, ridge.ErrOverflow) || errors.Is(err, matrix.ErrOverflow) || errors.Is(err, stats.ErrOverflow) {
				t.Fatalf("diagnostic overflow: fit=%v err=%v", fit, err)
			}
		})
	}
}

// TestRidge_AugmentedResidualOverflowPreservesMatrixError verifies the distinct
// dependency boundary where the augmented L2 norm itself cannot be represented.
// Implementation: use an all-zero two-row design and two MaxFloat64 responses.
// Behavior highlights: the optimum is beta=0 and the norm is sqrt(2)*MaxFloat64.
// Inputs: finite observations, positive Lambda and an origin-constrained model.
// Returns: no value; failure is atomic.
// Errors: ridge.ErrOverflow and matrix.ErrOverflow must both match.
// Determinism: exact zero feature column and fixed maximal responses.
// Complexity: one fixed 2-by-1 fit.
// Notes: this is a genuine dependency residual overflow, not just an overflowing SSE.
// AI-Hints: preserve the matrix error identity instead of inspecting its message.
func TestRidge_AugmentedResidualOverflowPreservesMatrixError(t *testing.T) {
	fit, err := ridge.FitThroughOrigin(context.Background(), dense(t, [][]float64{{0}, {0}}),
		[]float64{math.MaxFloat64, math.MaxFloat64}, 1)
	if fit != nil || !errors.Is(err, ridge.ErrOverflow) || !errors.Is(err, matrix.ErrOverflow) {
		t.Fatalf("augmented residual overflow: fit=%v err=%v", fit, err)
	}
}

// TestRidge_SmallestPositiveLambdaAcceptsFullRankDesign verifies the lower
// positive regularization boundary without requesting a rank-deficient solve.
// Implementation: fit x=y=[1,2] at the smallest positive binary64 Lambda.
// Behavior highlights: sqrt(n*Lambda) is positive even at subnormal Lambda.
// Inputs: finite scalar design with full numerical rank and positive Lambda.
// Returns: no value; slope is approximately one and intercept is exactly zero.
// Errors: none; all diagnostics are representable (the penalty may round to zero).
// Determinism: fixed two-row fixture and analytic limiting optimum.
// Complexity: fixed scalar QR and O(1) oracle work.
// Notes: this does not promise resolution of arbitrary collinear data at tiny Lambda.
// AI-Hints: a rounded-zero penalty is not permission to accept Lambda=0.
func TestRidge_SmallestPositiveLambdaAcceptsFullRankDesign(t *testing.T) {
	fit, err := ridge.FitThroughOrigin(context.Background(), dense(t, [][]float64{{1}, {2}}),
		[]float64{1, 2}, math.SmallestNonzeroFloat64)
	if err != nil {
		t.Fatal(err)
	}
	beta, intercept := fit.Model.Coefficients()
	if len(beta) != 1 || math.Float64bits(intercept) != math.Float64bits(0) {
		t.Fatalf("unexpected model beta=%v b=%g", beta, intercept)
	}
	closeTo(t, beta[0], 1)
	closeTo(t, fit.ResidualSumSquares, 0)
	closeTo(t, fit.Objective, 0)
	if math.IsNaN(fit.Penalty) || math.IsInf(fit.Penalty, 0) || fit.Penalty < 0 {
		t.Fatalf("invalid penalty %g", fit.Penalty)
	}
}

// TestRidge_PredictFiniteAfterPartialSumOverflow isolates overflow of a running
// sum even though every individual product and the exact final answer are finite.
// Implementation: train three independent slopes, then predict three large terms.
// Behavior highlights: ordinary sequential addition overflows before cancellation.
// Inputs: identity design, y=[4,4,-4], Lambda=1, and a large held-out row.
// Returns: no value; exact-binary64 rational oracle must match the output bits.
// Errors: none; rejecting the intermediate sum would be a false overflow.
// Determinism: fixed feature order and exact rational oracle.
// Complexity: fixed 3-feature fit and one O(p) exact inference row.
// Notes: validate the fixture using the published slopes, not ideal real coefficients.
// AI-Hints: do not use an ordinary dot product as an overflow-cancellation oracle.
func TestRidge_PredictFiniteAfterPartialSumOverflow(t *testing.T) {
	ctx := context.Background()
	fit, err := ridge.FitThroughOrigin(ctx, dense(t, [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}), []float64{4, 4, -4}, 1)
	if err != nil {
		t.Fatal(err)
	}
	beta, intercept := fit.Model.Coefficients()
	if len(beta) != 3 {
		t.Fatalf("unexpected slopes %v", beta)
	}
	feature := 0.75 * math.MaxFloat64
	features := []float64{feature, feature, feature}
	for _, slope := range beta {
		if math.IsInf(feature*slope, 0) || math.IsNaN(feature*slope) {
			t.Fatalf("fixture requires finite products, slopes=%v", beta)
		}
	}
	if !math.IsInf(features[0]*beta[0]+features[1]*beta[1], 1) {
		t.Fatal("fixture does not overflow the positive partial sum")
	}
	want := exactAffinePrediction(t, beta, intercept, features)
	if math.IsNaN(want) || math.IsInf(want, 0) {
		t.Fatalf("fixture requires finite exact result, got %g", want)
	}
	values, err := fit.Model.Predict(ctx, dense(t, [][]float64{features}))
	if err != nil || len(values) != 1 || math.Float64bits(values[0]) != math.Float64bits(want) {
		t.Fatalf("prediction=%v err=%v want=%g", values, err, want)
	}
}

// TestRidge_PredictFinalCompensatedOverflow verifies the final conversion gate
// when all products and ordinary additions are finite but their exact sum is not.
// Implementation:
//   - Stage 1: fit independent positive slopes on an identity design.
//   - Stage 2: construct one near-maximal term plus many quarter-ULP increments.
//   - Stage 3: confirm the finite naive sum and overflowing rational oracle.
//
// Behavior highlights: compensation exposes real overflow hidden by rounded adds.
// Inputs: a fixed 16-feature identity design and finite held-out observations.
// Returns: no value; prediction must fail atomically.
// Errors: ridge.ErrOverflow; a saturated MaxFloat64 prediction is incorrect.
// Determinism: fixed feature order; bounded adjustment of the leading feature.
// Complexity: one small QR fit and O(p) exact rational oracle work.
// Notes: the fixture adapts only to the learned rounded slopes, not private state.
// AI-Hints: avoid asserting overflow from an already overflowing ordinary sum;
// this fixture specifically distinguishes final compensation from partial-sum overflow.
func TestRidge_PredictFinalCompensatedOverflow(t *testing.T) {
	const featuresCount = 16
	const lambda = 1.0 / featuresCount
	const quarterMaxULP = 0x1p969
	ctx := context.Background()
	train := make([][]float64, featuresCount)
	y := make([]float64, featuresCount)
	for i := range train {
		train[i] = make([]float64, featuresCount)
		train[i][i] = 1
		y[i] = 4
	}
	fit, err := ridge.FitThroughOrigin(ctx, dense(t, train), y, lambda)
	if err != nil {
		t.Fatal(err)
	}
	beta, intercept := fit.Model.Coefficients()
	if len(beta) != featuresCount || intercept != 0 {
		t.Fatalf("invalid fitted coordinates: %v b=%g", beta, intercept)
	}
	for _, slope := range beta {
		if !(slope > 1 && slope < 3) {
			t.Fatalf("expected finite slopes near two, got %v", beta)
		}
	}
	features := make([]float64, featuresCount)
	features[0] = math.MaxFloat64 / beta[0]
	// A rounded quotient may produce an overflowing product. Step at most twice
	// toward zero; the resulting product remains within a few ULPs of the maximum.
	for attempt := 0; attempt < 2 && math.IsInf(features[0]*beta[0], 0); attempt++ {
		features[0] = math.Nextafter(features[0], 0)
	}
	for j := 1; j < featuresCount; j++ {
		features[j] = quarterMaxULP / beta[j]
	}
	// Prove the ordinary binary64 path does not overflow at any product or sum.
	naive := intercept
	for j, slope := range beta {
		term := features[j] * slope
		naive += term
		if math.IsInf(term, 0) || math.IsInf(naive, 0) || math.IsNaN(naive) {
			t.Fatalf("fixture overflowed before final compensation at feature %d", j)
		}
	}
	if want := exactAffinePrediction(t, beta, intercept, features); !math.IsInf(want, 1) {
		t.Fatalf("fixture does not have an overflowing exact affine sum: %g", want)
	}
	values, err := fit.Model.Predict(ctx, dense(t, [][]float64{features}))
	if values != nil || !errors.Is(err, ridge.ErrOverflow) {
		t.Fatalf("final overflow: values=%v err=%v", values, err)
	}
}
