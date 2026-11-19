// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test verifies the public convenience facades against
// independent causal, ownership, and sentinel-first contract expectations.
package gjrgarch_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

func TestAPI_StationaryVolatilityWrappers(t *testing.T) {
	params := gjrgarch.Parameters{Mean: 0.002, Omega: 0.00002, Alpha: 0.08, Gamma: 0.12, Beta: 0.7}
	model, err := gjrgarch.New(params)
	if err != nil {
		t.Fatal(err)
	}
	stationary, err := model.UnconditionalVariance()
	if err != nil {
		t.Fatal(err)
	}
	mustNearRelative(t, stationary, params.Omega/(1-.08-.06-.7), 1e-12, "unconditional variance")
	returns := []float64{0.014, -0.006, -0.027, 0.008, -0.031, 0.019, 0.002, -0.01, 0.011, -0.007, 0.013, -0.022}
	variance, err := model.FilterStationary(context.Background(), returns)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := model.Filter(context.Background(), returns, stationary)
	if err != nil {
		t.Fatal(err)
	}
	volatility, err := model.FilterVolatility(context.Background(), returns, stationary)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, len(variance), len(returns), "stationary filter shape")
	for i := range variance {
		mustNearRelative(t, variance[i], baseline[i], 1e-13, "filter wrapper parity")
		mustNearRelative(t, volatility[i], math.Sqrt(baseline[i]), 1e-13, "filter volatility")
	}

	state, err := model.InitialState(returns[len(returns)-1], baseline[len(baseline)-1])
	if err != nil {
		t.Fatal(err)
	}
	one, err := model.ForecastNextVariance(state)
	if err != nil {
		t.Fatal(err)
	}
	forecast, err := model.ForecastVariance(state, 10)
	if err != nil {
		t.Fatal(err)
	}
	sigma, err := model.ForecastVolatility(state, 10)
	if err != nil {
		t.Fatal(err)
	}
	mustNearRelative(t, one, forecast[0], 1e-13, "scalar forecast kernel parity")
	for i := range sigma {
		mustNearRelative(t, sigma[i], math.Sqrt(forecast[i]), 1e-13, "forecast volatility")
	}

	// Mutating a published forecast cannot alter another call or model state.
	forecast[0] = -1
	again, err := model.ForecastVariance(state, 10)
	if err != nil || again[0] <= 0 {
		t.Fatalf("forecast output aliased model: %v", err)
	}
	variance[0] = -1
	if baseline[0] <= 0 {
		t.Fatal("independent output slices share storage")
	}
}

func TestAPI_FitWithInitialAndObjectiveScore(t *testing.T) {
	initial := gjrgarch.Parameters{Omega: 0.1, Alpha: 0.13, Gamma: 0.11, Beta: 0.52}
	observed := gradientFixture()
	fromWrapper, wrapperErr := gjrgarch.FitWithInitial(context.Background(), observed, initial)
	fromCanonical, canonicalErr := gjrgarch.Fit(context.Background(), observed, gjrgarch.Options{Initial: initial})
	if !sameClassification(wrapperErr, canonicalErr) || fromWrapper == nil || fromCanonical == nil {
		t.Fatalf("wrapper mismatch: wrapper=%v canonical=%v", wrapperErr, canonicalErr)
	}
	if fromWrapper.Model.Parameters() != fromCanonical.Model.Parameters() {
		t.Fatal("wrapper changed deterministic model")
	}
	mustEqualInt(t, len(fromWrapper.Diagnostics.Objective), len(fromCanonical.Diagnostics.Objective), "history parity")
	for i, value := range fromWrapper.Diagnostics.Objective {
		if value != fromCanonical.Diagnostics.Objective[i] {
			t.Fatal("wrapper changed objective ordering or arithmetic")
		}
	}

	// A separate evaluator is valuable for model-monitoring and independent audits.
	fixed, err := gjrgarch.EvaluateGaussianQMLE(context.Background(), observed, initial, 0)
	if err != nil {
		t.Fatal(err)
	}
	mustNearRelative(t, fixed.Objective, fromWrapper.Diagnostics.Objective[0], 1e-12, "fit/evaluation initial objective")
	mustNearRelative(t, fixed.LogLikelihood, -float64(len(observed))*fixed.Objective, 1e-12, "total vs average")
	mustNilState(t, fixed, false, "QMLE evaluation")
	var absent *gjrgarch.QMLEEvaluation
	mustNilState(t, absent, true, "typed-nil evaluation")
}

// sameClassification compares the only nonfatal unfinished-fit category.
// Context errors are checked by dedicated tests with their own independent contexts.
func sameClassification(a, b error) bool {
	return (a == nil && b == nil) ||
		(errors.Is(a, gjrgarch.ErrNotConverged) && errors.Is(b, gjrgarch.ErrNotConverged))
}

func TestFitResult_NilableAndDetachedClone(t *testing.T) {
	var absent *gjrgarch.FitResult
	mustNilState(t, absent, true, "typed nil")
	mustNilState(t, new(gjrgarch.FitResult), false, "allocated empty result")
	if absent.Clone() != nil {
		t.Fatal("nil result clone must be nil")
	}
	data := gradientFixture()
	res, err := gjrgarch.Fit(context.Background(), data, gjrgarch.Options{
		Initial:       gjrgarch.Parameters{Omega: 0.12, Alpha: 0.09, Gamma: 0.18, Beta: 0.5},
		MaxIterations: 2,
	})
	if err != nil && !errors.Is(err, gjrgarch.ErrNotConverged) {
		t.Fatal(err)
	}
	if res == nil || res.IsNil() {
		t.Fatal("valid pointer must be non-nilable")
	}
	clone := res.Clone()
	if clone == res || clone.Model != res.Model {
		t.Fatal("clone must own its result but may share immutable model")
	}
	first := res.Diagnostics.Objective[0]
	clone.Diagnostics.Objective[0] += 100
	if res.Diagnostics.Objective[0] != first {
		t.Fatal("result clone aliased mutable objective history")
	}
}

func TestAPI_EvaluatorRejectsInvalidOrInfeasibleCandidates(t *testing.T) {
	model := gjrgarch.Parameters{Omega: 0.1, Alpha: 0.1, Gamma: 0.2, Beta: 0.5}
	observations := []float64{0.13, -0.07, 0.03}
	cases := []struct {
		name       string
		ctx        context.Context
		data       []float64
		parameters gjrgarch.Parameters
		floor      float64
		sentinel   error
	}{
		{"missing context", nil, observations, model, 0, gjrgarch.ErrInvalidOption},
		{"invalid floor", context.Background(), observations, model, -1, gjrgarch.ErrInvalidOption},
		{"invalid model", context.Background(), observations, gjrgarch.Parameters{}, 0, gjrgarch.ErrDomain},
		{"nil returns", context.Background(), nil, model, 0, gjrgarch.ErrNilInput},
		{"empty returns", context.Background(), []float64{}, model, 0, gjrgarch.ErrEmptyInput},
		{"NaN returns", context.Background(), []float64{math.NaN()}, model, 0, gjrgarch.ErrNonFinite},
		{"omega below floor despite h0 above floor", context.Background(), observations, model, 0.12, gjrgarch.ErrDomain},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := gjrgarch.EvaluateGaussianQMLE(tc.ctx, tc.data, tc.parameters, tc.floor)
			if res != nil {
				t.Fatal("invalid likelihood published a result")
			}
			mustErrorIs(t, err, tc.sentinel)
		})
	}
}

func TestModel_ConcurrentReadOnlyFilteringAndForecasting(t *testing.T) {
	model, err := gjrgarch.New(gjrgarch.Parameters{Omega: 0.12, Alpha: 0.13, Gamma: -0.04, Beta: 0.42})
	if err != nil {
		t.Fatal(err)
	}
	returns := gradientFixture()
	baseline, err := model.FilterStationary(context.Background(), returns)
	if err != nil {
		t.Fatal(err)
	}
	const workers = 24
	errorChannel := make(chan error, workers)
	var group sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		group.Add(1)
		go func() {
			defer group.Done()
			result, e := model.FilterStationary(context.Background(), returns)
			if e != nil {
				errorChannel <- e
				return
			}
			for i := range result {
				if result[i] != baseline[i] {
					errorChannel <- gjrgarch.ErrOverflow // Identifies a deterministic output violation.
					return
				}
			}
			state, e := model.InitialState(returns[len(returns)-1], result[len(result)-1])
			if e == nil {
				_, e = model.ForecastVariance(state, 10)
			}
			if e != nil {
				errorChannel <- e
			}
		}()
	}
	group.Wait()
	close(errorChannel)
	for err := range errorChannel {
		t.Fatalf("concurrent immutable model failed: %v", err)
	}
}
