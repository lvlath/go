// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test anchors numerical edge cases and failure atomicity.
// The fixtures exercise the exported surface rather than private kernels.
package gjrgarch_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

// stagedContext deterministically triggers cancellation on a chosen Err call.
// No timeouts, sleeps, or worker goroutine failures are needed for this regression.
type stagedContext struct {
	context.Context
	calls  int
	stopAt int
}

func (ctx *stagedContext) Err() error {
	ctx.calls++
	if ctx.calls >= ctx.stopAt {
		return context.Canceled
	}
	return nil
}

func TestGJR_RepresentableShockWithUnrepresentableResidualSquare(t *testing.T) {
	// The pure square (1e160)^2 overflows, but 3e-308 * (1e160)^2 = 3e12 is finite.
	params := gjrgarch.Parameters{Omega: 0.1, Alpha: 2e-308, Gamma: 1e-308, Beta: 0.4}
	model, err := gjrgarch.New(params)
	if err != nil {
		t.Fatal(err)
	}
	state, err := model.InitialState(-1e160, 1)
	if err != nil {
		t.Fatal(err)
	}
	variance, err := model.ForecastNextVariance(state)
	if err != nil {
		t.Fatal(err)
	}
	// Compute the independent mathematical value without squaring residual directly.
	mustNearRelative(t, variance, 3e12+0.5, 1e-12, "scaled shock product")
	positive, err := model.InitialState(1e160, 1)
	if err != nil {
		t.Fatal(err)
	}
	up, err := model.ForecastNextVariance(positive)
	if err != nil {
		t.Fatal(err)
	}
	mustNearRelative(t, up, 2e12+0.5, 1e-12, "asymmetric positive shock")
}

func TestGJR_NegativeGammaAndSignedZeroAreWellDefined(t *testing.T) {
	params := gjrgarch.Parameters{Omega: 0.1, Alpha: 0.18, Gamma: -0.17, Beta: 0.5}
	model, err := gjrgarch.New(params)
	if err != nil {
		t.Fatal(err)
	}
	negative, err := model.InitialState(-0.8, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	positive, err := model.InitialState(0.8, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	nextNegative, err := model.ForecastNextVariance(negative)
	if err != nil {
		t.Fatal(err)
	}
	nextPositive, err := model.ForecastNextVariance(positive)
	if err != nil {
		t.Fatal(err)
	}
	mustNearRelative(t, nextNegative, 0.1+0.01*0.64+0.5*0.2, 1e-12, "negative gamma innovation")
	mustNearRelative(t, nextPositive, 0.1+0.18*0.64+0.5*0.2, 1e-12, "positive gamma innovation")
	negativeZero := math.Copysign(0, -1)
	zero, err := model.InitialState(negativeZero, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	nextZero, err := model.ForecastNextVariance(zero)
	if err != nil {
		t.Fatal(err)
	}
	mustNearRelative(t, nextZero, 0.2, 1e-12, "negative zero indicator must be false")
}

func TestGJR_LateCancellationReturnsOnlyValidatedCandidate(t *testing.T) {
	initial := gjrgarch.Parameters{Omega: 0.1, Alpha: 0.11, Gamma: 0.16, Beta: 0.47}
	data := gradientFixture()
	// Fit calls ctx.Err before options and during the initial QMLE, then in run.
	cancel := &stagedContext{Context: context.Background(), stopAt: 3}
	result, err := gjrgarch.Fit(cancel, data, gjrgarch.Options{Initial: initial, MaxIterations: 10})
	mustErrorIs(t, err, context.Canceled)
	if result == nil || result.Model == nil || result.Diagnostics.Converged {
		t.Fatal("late cancellation must retain a valid nonconverged model")
	}
	mustEqualInt(t, result.Diagnostics.Iterations, 0, "no accepted iterations before cancellation")
	mustEqualInt(t, len(result.Diagnostics.Objective), 1, "initial objective preserved")

	// Cancellation before a successful evaluation must be atomic nil/error.
	cancelFirst := &stagedContext{Context: context.Background(), stopAt: 1}
	early, err := gjrgarch.Fit(cancelFirst, data, gjrgarch.Options{Initial: initial})
	mustErrorIs(t, err, context.Canceled)
	mustNilState(t, early, true, "early cancellation")
}

func TestGJR_FilterCancellationIsAtomic(t *testing.T) {
	model, err := gjrgarch.New(p())
	if err != nil {
		t.Fatal(err)
	}
	data := gradientFixture()
	cancel := &stagedContext{Context: context.Background(), stopAt: 4}
	variances, err := model.Filter(cancel, data, 1)
	mustErrorIs(t, err, context.Canceled)
	if variances != nil {
		t.Fatal("failed filter published partially processed variances")
	}
}

func TestGJR_FitDeterministicBudgetAndKKTFeasibility(t *testing.T) {
	data := []float64{-.023, .014, .007, -.035, .018, -.01, .021, .009, -.016, .01, -.028, .012, .007, -.011, .024, -.02}
	initial := gjrgarch.Parameters{Omega: 0.00001, Alpha: 0.1, Gamma: 0.15, Beta: 0.6}
	o := gjrgarch.Options{Initial: initial, MaxIterations: 1}
	first, firstErr := gjrgarch.Fit(context.Background(), data, o)
	second, secondErr := gjrgarch.Fit(context.Background(), data, o)
	if first == nil || second == nil || !errors.Is(firstErr, gjrgarch.ErrNotConverged) || !errors.Is(secondErr, gjrgarch.ErrNotConverged) {
		t.Fatalf("budget must retain a last feasible candidate: %v; %v", firstErr, secondErr)
	}
	mustEqualBool(t, first.Diagnostics.Converged, false, "budget convergence")
	if first.Model.Parameters() != second.Model.Parameters() || first.LogLikelihood != second.LogLikelihood {
		t.Fatal("repeated fits are not deterministically identical")
	}
	for i := range first.Diagnostics.Objective {
		if first.Diagnostics.Objective[i] != second.Diagnostics.Objective[i] {
			t.Fatal("accepted objective history is nondeterministic")
		}
	}
	parameters := first.Model.Parameters()
	rho := parameters.Alpha/2 + (parameters.Alpha+parameters.Gamma)/2 + parameters.Beta
	if parameters.Alpha < 0 || parameters.Beta < 0 || parameters.Alpha+parameters.Gamma < 0 ||
		rho > 1-64*(0x1p-52) {
		t.Fatalf("KKT accepted an infeasible point: %+v rho=%.17g", parameters, rho)
	}
}

func TestGJR_InitialAndStepRejectStateWithoutMutation(t *testing.T) {
	model, err := gjrgarch.New(p())
	if err != nil {
		t.Fatal(err)
	}
	state, err := model.InitialState(-0.8, 1)
	if err != nil {
		t.Fatal(err)
	}
	broken := state
	broken.LastVariance = math.NaN()
	_, err = model.Step(broken, 0.2)
	mustErrorIs(t, err, gjrgarch.ErrStateVersion)
	if state.LastVariance != 1 || state.Steps != 1 {
		t.Fatal("caller state was mutated by failed Step")
	}
	broken = state
	broken.LastVariance = math.Inf(1)
	_, err = model.ForecastVariance(broken, 1)
	mustErrorIs(t, err, gjrgarch.ErrStateVersion)
	broken = state
	broken.LastResidual = math.NaN()
	_, err = model.ForecastNextVariance(broken)
	mustErrorIs(t, err, gjrgarch.ErrStateVersion)
}

// TestGJR_CancellationWithinKKTProjection anchors late cancellation while
// searching the projected-gradient face rather than only between iterations.
func TestGJR_CancellationWithinKKTProjection(t *testing.T) {
	initial := gjrgarch.Parameters{Omega: 0.13, Alpha: 0.11, Gamma: 0.18, Beta: 0.46}
	// Check 1: entry; check 2: initial likelihood; check 3: optimizer loop;
	// check 4: KKT project entry. Only a valid initial candidate may survive.
	cancel := &stagedContext{Context: context.Background(), stopAt: 4}
	result, err := gjrgarch.Fit(cancel, gradientFixture(), gjrgarch.Options{
		Initial: initial, MaxIterations: 7,
	})
	mustErrorIs(t, err, context.Canceled)
	if result == nil || result.Model == nil || result.Diagnostics.Converged {
		t.Fatal("KKT cancellation failed to publish truthful last valid candidate")
	}
	mustEqualInt(t, result.Diagnostics.Iterations, 0, "no projection candidate accepted")
}

// TestGJR_GaussianObjectiveScaleEquivariance verifies the likelihood Jacobian
// and coordinate scaling from the public evaluator, not from a private helper.
func TestGJR_GaussianObjectiveScaleEquivariance(t *testing.T) {
	original := gjrgarch.Parameters{Mean: 0.035, Omega: 0.08, Alpha: 0.14, Gamma: -0.05, Beta: 0.43}
	base := gradientFixture()
	const factor = 7.0
	const varianceFloor = 0.01
	scaled := original
	scaled.Mean *= factor
	scaled.Omega *= factor * factor
	data := make([]float64, len(base))
	for i, value := range base {
		data[i] = factor * value
	}
	before, err := gjrgarch.EvaluateGaussianQMLE(context.Background(), base, original, varianceFloor)
	if err != nil {
		t.Fatal(err)
	}
	after, err := gjrgarch.EvaluateGaussianQMLE(context.Background(), data, scaled, varianceFloor*factor*factor)
	if err != nil {
		t.Fatal(err)
	}
	mustNearRelative(t, after.Objective, before.Objective+math.Log(factor), 1e-10, "Gaussian Jacobian")
	mustNearRelative(t, after.LogLikelihood, before.LogLikelihood-float64(len(base))*math.Log(factor), 1e-10, "total log-likelihood Jacobian")
	left := gradientCoordinates(before.Gradient)
	right := gradientCoordinates(after.Gradient)
	mustNearRelative(t, right[0], left[0]/factor, 1e-10, "mean gradient coordinate scaling")
	mustNearRelative(t, right[1], left[1]/(factor*factor), 1e-10, "omega gradient coordinate scaling")
	for j := 2; j < len(left); j++ {
		mustNearRelative(t, right[j], left[j], 1e-10, "dimensionless gradient coordinate scaling")
	}
}

// TestGJR_FilterSingletonResidualOverflow ensures a one-observation batch
// validates the same posterior as InitialState instead of skipping its residual.
// This protects the arithmetic-domain sentinel and batch/stream parity.
func TestGJR_FilterSingletonResidualOverflow(t *testing.T) {
	model, err := gjrgarch.New(gjrgarch.Parameters{
		Mean: -math.MaxFloat64, Omega: 0.1, Alpha: 0.1, Beta: 0.5,
	})
	if err != nil {
		t.Fatal(err)
	}
	returns := []float64{math.MaxFloat64}
	variances, err := model.Filter(context.Background(), returns, 1)
	if variances != nil {
		t.Fatalf("overflow must not publish variances: %v", variances)
	}
	mustErrorIs(t, err, gjrgarch.ErrOverflow)
	_, err = model.InitialState(returns[0], 1)
	mustErrorIs(t, err, gjrgarch.ErrOverflow)
}
