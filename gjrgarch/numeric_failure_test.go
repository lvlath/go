// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test distinguishes finite-input arithmetic failures from
// invalid observations and verifies atomic publication through the public API.
package gjrgarch_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

// TestGJR_ForecastKnownShockOverflowAtomic protects every forecast facade.
// A valid posterior with an extreme innovation must not publish any horizon.
// Implementation:
//   - Stage 1: Construct a stationary model and a valid, finite posterior.
//   - Stage 2: Trigger an unrepresentable weighted square on the first forecast.
//   - Stage 3: Verify scalar, variance, and volatility errors and state immutability.
//
// Complexity: O(1) time and space; failed forecasts must not allocate an output.
// AI-Hints: A large finite residual is not ErrNonFinite; never saturate to MaxFloat64.
func TestGJR_ForecastKnownShockOverflowAtomic(t *testing.T) {
	model, err := gjrgarch.New(gjrgarch.Parameters{Omega: 1, Alpha: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	// The state contains a finite innovation; only its future weighted square overflows.
	state, err := model.InitialState(math.MaxFloat64, 1)
	if err != nil {
		t.Fatal(err)
	}
	before := state
	variance, err := model.ForecastVariance(state, 4)
	mustErrorIs(t, err, gjrgarch.ErrOverflow)
	if variance != nil {
		t.Fatal("failed variance forecast published output")
	}
	volatility, err := model.ForecastVolatility(state, 4)
	mustErrorIs(t, err, gjrgarch.ErrOverflow)
	if volatility != nil {
		t.Fatal("failed volatility forecast published output")
	}
	_, err = model.ForecastNextVariance(state)
	mustErrorIs(t, err, gjrgarch.ErrOverflow)
	if state != before {
		t.Fatal("forecast mutated the posterior")
	}
}

// TestGJR_ForecastExtremeFiniteMeanReversion checks a near-limit, valid long-run level.
// The independent closed form avoids reproducing the iterative forecast implementation.
// Complexity: O(H) time and O(H) output for a fixed 32-step horizon.
// AI-Hints: Scale the oracle before adding terms near MaxFloat64; do not assert
// an overflowing intermediate sum as the mathematical reference value.
func TestGJR_ForecastExtremeFiniteMeanReversion(t *testing.T) {
	const horizon = 32
	const rho = 0.75
	const omega = math.MaxFloat64 / 8
	model, err := gjrgarch.New(gjrgarch.Parameters{Omega: omega, Beta: rho})
	if err != nil {
		t.Fatal(err)
	}
	// The forecast starts above its long-run level and should decay toward it.
	state, err := model.InitialState(0, math.MaxFloat64)
	if err != nil {
		t.Fatal(err)
	}
	forecast, err := model.ForecastVariance(state, horizon)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, len(forecast), horizon, "forecast horizon")
	// In MaxFloat64 units, h_infinity = 0.5 and h_0 = 1.
	for i, got := range forecast {
		want := 0.5 + 0.5*math.Pow(rho, float64(i+1))
		mustNearRelative(t, got/math.MaxFloat64, want, 4e-15, "normalized closed-form forecast")
		if i > 0 && got > forecast[i-1] {
			t.Fatal("forecast increased above its decreasing mean-reversion path")
		}
	}
}

// TestGJR_QMLEFiniteInputArithmeticFailures isolates distinct numeric failure modes.
// Every candidate is a valid stationary model with a positive floor. The expected
// failure originates in likelihood arithmetic, not caller-input validation.
//
// Implementation:
//   - Stage 1: Independently construct each finite extreme-value witness.
//   - Stage 2: Check New accepts its parameters, then evaluate the public score.
//   - Stage 3: Assert ErrOverflow, nil evaluation, and unchanged observations.
//
// Complexity: O(T) per case; T is at most two for the selected witnesses.
// AI-Hints: Finite h and finite objective terms do not imply representable
// derivatives; do not substitute an infinite or truncated gradient.
func TestGJR_QMLEFiniteInputArithmeticFailures(t *testing.T) {
	cases := []struct {
		name       string
		parameters gjrgarch.Parameters
		returns    []float64
		floor      float64
	}{
		// h0 is finite (~1e306), but dh0/dBeta = h0/(1-rho) is too large.
		{"stationary_sensitivity", gjrgarch.Parameters{Omega: 1e300, Beta: 1 - 0x1p-20}, []float64{0}, 1},
		// Both return and mean are finite, but their difference is not.
		{"residual_difference", gjrgarch.Parameters{Mean: -math.MaxFloat64, Omega: 1}, []float64{math.MaxFloat64}, 1},
		// A tiny variance and huge residual overflow the standardized innovation.
		{"standardized_residual", gjrgarch.Parameters{Omega: 1e-300}, []float64{math.MaxFloat64}, 1e-300},
		// z itself is finite while z*z is outside the binary64 range.
		{"standardized_square", gjrgarch.Parameters{Omega: 1}, []float64{1e155}, 1},
		// Even z=0 has an unrepresentable variance score at a subnormal h.
		{"variance_score", gjrgarch.Parameters{Omega: 1e-310}, []float64{0}, 1e-310},
		// The variance score is finite, but its product with dh0/dOmega is not.
		{"gradient_product", gjrgarch.Parameters{Omega: 1e-150 * 0x1p-20, Beta: 1 - 0x1p-20}, []float64{100}, 1e-160},
		// z^2 is modest; advancing dAlpha requires the unrepresentable raw square.
		{"lagged_residual_square", gjrgarch.Parameters{Omega: 1e308}, []float64{1e155, 0}, 1},
		// The raw square is finite, but adding the lagged sensitivity overflows.
		{"lagged_sensitivity", gjrgarch.Parameters{Omega: 3e307, Beta: 0.5}, []float64{math.Sqrt(1.3e308), 0}, 1},
		// Alpha=1.5 is legal with Gamma=-1.5; a positive shock overflows h1.
		{"lagged_variance", gjrgarch.Parameters{Omega: 1e307, Alpha: 1.5, Gamma: -1.5}, []float64{math.Sqrt(1.2e308), 0}, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := gjrgarch.New(tc.parameters); err != nil {
				t.Fatalf("fixture must be a valid stationary model: %v", err)
			}
			before := append([]float64(nil), tc.returns...)
			result, err := gjrgarch.EvaluateGaussianQMLE(context.Background(), tc.returns, tc.parameters, tc.floor)
			mustErrorIs(t, err, gjrgarch.ErrOverflow)
			mustNilState(t, result, true, "failed QMLE evaluation")
			for i, observed := range tc.returns {
				if observed != before[i] {
					t.Fatalf("evaluation mutated observation %d", i)
				}
			}
		})
	}
}

// TestGJR_QMLETerminalObservationDoesNotAdvance verifies the lagged derivative law.
// The same huge residual is legal at the terminal index but not when a later
// likelihood term requires its unrepresentable derivative square.
// Complexity: O(1) for the one- and two-observation calls.
func TestGJR_QMLETerminalObservationDoesNotAdvance(t *testing.T) {
	params := gjrgarch.Parameters{Omega: 1e308}
	const residual = 1e155
	one, err := gjrgarch.EvaluateGaussianQMLE(context.Background(), []float64{residual}, params, 1)
	if err != nil || one == nil {
		t.Fatalf("final observation should not advance sensitivities: %v", err)
	}
	// The independently scaled z avoids squaring the large residual directly.
	z := residual / math.Sqrt(params.Omega)
	want := 0.5 * (math.Log(2*math.Pi) + math.Log(params.Omega) + z*z)
	mustNearRelative(t, one.Objective, want, 1e-14, "terminal density loss")
	mustEqualInt(t, one.Observations, 1, "terminal sample count")
	two, err := gjrgarch.EvaluateGaussianQMLE(context.Background(), []float64{residual, 0}, params, 1)
	mustErrorIs(t, err, gjrgarch.ErrOverflow)
	mustNilState(t, two, true, "nonterminal sensitivity overflow")
}

// TestGJR_FitUnrepresentableInitialArithmetic separates preparation and score errors.
// No feasible initial objective exists, so neither error may expose a candidate.
// Complexity: O(T) preparation/evaluation, with fixed singleton samples.
func TestGJR_FitUnrepresentableInitialArithmetic(t *testing.T) {
	cases := []struct {
		name    string
		returns []float64
		options gjrgarch.Options
	}{
		// The original floor is valid; dividing twice by the enormous scale loses it.
		{"normalized_floor_underflow", []float64{math.MaxFloat64}, gjrgarch.Options{Initial: gjrgarch.Parameters{Omega: 1}, VarianceFloor: 1e-12}},
		// Scale=1 leaves the finite subnormal floor intact, but the initial score fails.
		{"initial_score_overflow", []float64{1}, gjrgarch.Options{Initial: gjrgarch.Parameters{Omega: 1e-310}, VarianceFloor: 1e-310}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := gjrgarch.New(tc.options.Initial); err != nil {
				t.Fatalf("invalid fixture: %v", err)
			}
			fit, err := gjrgarch.Fit(context.Background(), tc.returns, tc.options)
			mustErrorIs(t, err, gjrgarch.ErrOverflow)
			mustNilState(t, fit, true, "fit before valid initial score")
		})
	}
}

// TestGJR_FilterSingletonAndLateArithmeticFailure anchors the batch boundary.
// A valid one-observation series must publish the supplied variance; extending
// it with a finite but unrepresentable residual must discard the whole result.
// Complexity: O(T) with two observations and O(1) independent state.
// AI-Hints: The final residual must be validated even when it has no successor.
func TestGJR_FilterSingletonAndLateArithmeticFailure(t *testing.T) {
	model, err := gjrgarch.New(gjrgarch.Parameters{Mean: -math.MaxFloat64, Omega: 1})
	if err != nil {
		t.Fatal(err)
	}
	first := -math.MaxFloat64
	const initialVariance = 0.25
	state, err := model.InitialState(first, initialVariance)
	if err != nil {
		t.Fatal(err)
	}
	one, err := model.Filter(context.Background(), []float64{first}, initialVariance)
	if err != nil || len(one) != 1 || one[0] != initialVariance {
		t.Fatalf("singleton filter must preserve h0: %v %v", one, err)
	}
	before := state
	_, err = model.Step(state, math.MaxFloat64)
	mustErrorIs(t, err, gjrgarch.ErrOverflow)
	if state != before {
		t.Fatal("failed Step mutated state")
	}
	observations := []float64{first, math.MaxFloat64}
	batch, err := model.Filter(context.Background(), observations, initialVariance)
	mustErrorIs(t, err, gjrgarch.ErrOverflow)
	if batch != nil {
		t.Fatal("failed late residual published a partial variance sequence")
	}
	volatility, err := model.FilterVolatility(context.Background(), observations, initialVariance)
	mustErrorIs(t, err, gjrgarch.ErrOverflow)
	if volatility != nil {
		t.Fatal("volatility wrapper must preserve batch atomicity")
	}
}

// TestGJR_RecursionSumOverflow distinguishes a finite shock from an overflowing sum.
// The model is stationary and the supplied posterior is valid, but adding its
// finite weighted shock to Omega exceeds the representable range.
// Complexity: O(1) time and space.
func TestGJR_RecursionSumOverflow(t *testing.T) {
	model, err := gjrgarch.New(gjrgarch.Parameters{Omega: 0.5 * math.MaxFloat64, Alpha: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	// The weighted shock is finite (1e308); adding Omega is not.
	state, err := model.InitialState(math.Sqrt(1e308)*math.Sqrt(2), 1)
	if err != nil {
		t.Fatal(err)
	}
	_, err = model.ForecastNextVariance(state)
	mustErrorIs(t, err, gjrgarch.ErrOverflow)
}

// TestGJR_AdmissionAndNilModelWrappers verifies consistent public preconditions.
// It separates invalid receiver, canceled context and excessive output size;
// no oversized allocation is attempted.
// Complexity: O(1) time and space.
func TestGJR_AdmissionAndNilModelWrappers(t *testing.T) {
	var missing *gjrgarch.Model
	_, err := missing.UnconditionalVariance()
	mustErrorIs(t, err, gjrgarch.ErrStateVersion)
	_, err = missing.InitialState(0, 1)
	mustErrorIs(t, err, gjrgarch.ErrStateVersion)
	batch, err := missing.Filter(context.Background(), []float64{0}, 1)
	mustErrorIs(t, err, gjrgarch.ErrStateVersion)
	if batch != nil {
		t.Fatal("nil model published a filter")
	}
	batch, err = missing.FilterStationary(context.Background(), []float64{0})
	mustErrorIs(t, err, gjrgarch.ErrStateVersion)
	if batch != nil {
		t.Fatal("nil model published a stationary filter")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	batch, err = missing.FilterStationary(ctx, nil)
	mustErrorIs(t, err, context.Canceled)
	if batch != nil {
		t.Fatal("canceled stationary filter published output")
	}
	model, err := gjrgarch.New(p())
	if err != nil {
		t.Fatal(err)
	}
	state, err := model.InitialState(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	batch, err = model.ForecastVariance(state, math.MaxInt/8+1)
	mustErrorIs(t, err, gjrgarch.ErrOverflow)
	if batch != nil {
		t.Fatal("unrepresentable allocation published forecast output")
	}
}

// TestGJR_PersistenceSumOverflow rejects arithmetic overflow after valid input scans.
// Both individual shock weights are finite, but adding the nonnegative
// persistence terms exceeds binary64 before the stationarity comparison.
func TestGJR_PersistenceSumOverflow(t *testing.T) {
	model, err := gjrgarch.New(gjrgarch.Parameters{Omega: 1, Alpha: math.MaxFloat64, Beta: math.MaxFloat64})
	mustErrorIs(t, err, gjrgarch.ErrOverflow)
	if model != nil {
		t.Fatal("overflowed persistence published a model")
	}
}
