// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima_test verifies public facade wrappers without accessing private kernels.
// Each case compares canonical Fit/Forecast with its policy-only shortcut.
package arima_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/arima"
)

func TestFitPolicyWrappersMatchCanonical(t *testing.T) {
	ctx := context.Background()
	samples := syntheticARMA(120)
	tests := []struct {
		name      string
		order     arima.Order
		withDrift bool
		wrapper   func() (*arima.FitResult, error)
	}{
		{"with_drift", arima.Order{AR: 1}, true, func() (*arima.FitResult, error) {
			return arima.FitWithDrift(ctx, samples, arima.Order{AR: 1})
		}},
		{"without_drift", arima.Order{MA: 1}, false, func() (*arima.FitResult, error) {
			return arima.FitWithoutDrift(ctx, samples, arima.Order{MA: 1})
		}},
		{"arma", arima.Order{AR: 1, MA: 1}, true, func() (*arima.FitResult, error) {
			return arima.FitARMA(ctx, samples, 1, 1, true)
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			canonical, ce := arima.Fit(ctx, samples, arima.Options{Order: tc.order, IncludeDrift: tc.withDrift})
			wrapped, we := tc.wrapper()
			if (ce == nil) != (we == nil) || (ce != nil && !errors.Is(we, ce)) {
				t.Fatalf("canonical %v, wrapper %v", ce, we)
			}
			if canonical == nil || wrapped == nil {
				t.Fatal("missing accepted fit")
			}
			mustEqualString(t, wrapped.Model.ID(), canonical.Model.ID(), "wrapper model")
			mustClose(t, wrapped.LogLikelihood, canonical.LogLikelihood)
			mustVector(t, wrapped.Diagnostics.Objective, canonical.Diagnostics.Objective)
		})
	}
}

func TestFitIntegratedAnalyticAndNoDrift(t *testing.T) {
	ctx := context.Background()
	levels := []float64{100, 104, 110, 117, 127, 139, 152, 168, 186, 205, 227, 251}
	result, err := arima.FitIntegrated(ctx, levels, 2, true)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Diagnostics.Converged || result.Diagnostics.Iterations != 0 {
		t.Fatalf("analytic solution: %+v", result.Diagnostics)
	}
	if result.Model.Parameters().Order.Difference != 2 {
		t.Fatal("incorrect integration order")
	}
	forecast, err := result.Model.ForecastNext(ctx, levels)
	if err != nil {
		t.Fatal(err)
	}
	if forecast.Horizon != 1 || len(forecast.Mean) != 1 || len(forecast.Variance) != 1 {
		t.Fatal("bad one-step result shape")
	}
	batch, err := result.Model.Forecast(ctx, levels, 1)
	if err != nil {
		t.Fatal(err)
	}
	mustVector(t, batch.Mean, forecast.Mean)
	mustVector(t, batch.Variance, forecast.Variance)

	baseline, err := arima.FitIntegrated(ctx, levels, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if baseline.Model.Parameters().Drift != 0 {
		t.Fatal("unexpected implicit drift")
	}
}

func TestWrapperValidationAndErrorIdentity(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		name string
		call func() (*arima.FitResult, error)
		want error
	}{
		{"bad_ar", func() (*arima.FitResult, error) { return arima.FitARMA(ctx, []float64{1, 2, 3}, 9, 0, false) }, arima.ErrInvalidOption},
		{"bad_ma", func() (*arima.FitResult, error) { return arima.FitARMA(ctx, []float64{1, 2, 3}, 0, -1, false) }, arima.ErrInvalidOption},
		{"bad_difference", func() (*arima.FitResult, error) { return arima.FitIntegrated(ctx, []float64{1, 2, 3}, 3, true) }, arima.ErrInvalidOption},
		{"nil_input", func() (*arima.FitResult, error) { return arima.FitWithDrift(ctx, nil, arima.Order{}) }, arima.ErrNilInput},
		{"short_fit", func() (*arima.FitResult, error) {
			return arima.FitWithoutDrift(ctx, []float64{1, 2}, arima.Order{AR: 1})
		}, arima.ErrInsufficientData},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := tc.call()
			mustErrorIs(t, err, tc.want)
			if result != nil {
				t.Fatal("published a result on validation error")
			}
		})
	}
}

func TestUninitializedModelAndCanonicalZeroHash(t *testing.T) {
	var empty arima.Model
	if empty.ID() != "" {
		t.Fatal("unexpected initialized ID")
	}
	forecast, err := empty.Forecast(context.Background(), []float64{1, 2, 3}, 2)
	mustErrorIs(t, err, arima.ErrInvalidOption)
	if forecast != nil {
		t.Fatal("uninitialized model published result")
	}

	positive := mustModel(t, arima.Parameters{Drift: 0, InnovationVariance: 1})
	negative := mustModel(t, arima.Parameters{Drift: math.Copysign(0, -1), InnovationVariance: 1})
	mustEqualString(t, positive.ID(), negative.ID(), "signed zero canonicalization")
	if positive.ID() == mustModel(t, arima.Parameters{Drift: 0, InnovationVariance: 2}).ID() {
		t.Fatal("innovation variance absent from identity")
	}
}

func TestNilablePointersAreNotValidityChecks(t *testing.T) {
	var missingFit *arima.FitResult
	var missingForecast *arima.ForecastResult
	if !missingFit.IsNil() || !missingForecast.IsNil() {
		t.Fatal("typed nil misclassified")
	}
	if (&arima.FitResult{}).IsNil() || (&arima.ForecastResult{}).IsNil() {
		t.Fatal("non-nil receivers must not be reclassified by mutable fields")
	}
}
