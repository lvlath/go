// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
// Package arima_test anchors the AR01-AR09 analytical acceptance matrix.
// This file checks direct forecast oracles, domain boundaries, prefix causality and nilability.
package arima_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/arima"
)

// TestAR01ToAR05AnalyticalForecast anchors five exact mean/variance recurrences.
func TestAR01ToAR05AnalyticalForecast(t *testing.T) {
	tests := []struct {
		name                    string
		p                       arima.Parameters
		history, mean, variance []float64
	}{
		{"AR01", arima.Parameters{Drift: 2, InnovationVariance: 3, Order: arima.Order{}}, []float64{7}, []float64{2, 2, 2}, []float64{3, 3, 3}},
		{"AR02", arima.Parameters{AR: []float64{0.5}, Drift: 1, InnovationVariance: 4, Order: arima.Order{AR: 1}}, []float64{5, 10}, []float64{6, 4, 3}, []float64{4, 5, 5.25}},
		{"AR03", arima.Parameters{InnovationVariance: 2, Order: arima.Order{Difference: 1}}, []float64{4, 5}, []float64{5, 5, 5}, []float64{2, 4, 6}},
		{"AR04", arima.Parameters{InnovationVariance: 1, Order: arima.Order{Difference: 2}}, []float64{1, 3, 6}, []float64{9, 12, 15}, []float64{1, 5, 14}},
		{"AR05", arima.Parameters{MA: []float64{0.5}, InnovationVariance: 1, Order: arima.Order{MA: 1}}, []float64{0, 2}, []float64{1, 0}, []float64{1, 1.25}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := mustModel(t, tc.p)
			got, err := m.Forecast(context.Background(), tc.history, len(tc.mean))
			if err != nil {
				t.Fatal(err)
			}
			mustVector(t, got.Mean, tc.mean)
			mustVector(t, got.Variance, tc.variance)
		})
	}
}

// TestAR06AnalyticFit checks profiled Gaussian variance and likelihood without an optimizer.
func TestAR06AnalyticFit(t *testing.T) {
	r, err := arima.Fit(context.Background(), []float64{1, 2, 3}, arima.Options{IncludeDrift: true})
	if err != nil {
		t.Fatal(err)
	}
	p := r.Model.Parameters()
	mustClose(t, p.Drift, 2)
	mustClose(t, p.InnovationVariance, 2.0/3.0)
	mustClose(t, r.LogLikelihood, -1.5*(math.Log(2*math.Pi)+1+math.Log(2.0/3.0)))
	mustEqualBool(t, r.Diagnostics.Converged, true, "analytic")
	mustEqualInt(t, r.Diagnostics.EffectiveN, 3, "effective")
}

// TestAR07Boundary rejects exact unit AR and MA roots on both signs.
func TestAR07Boundary(t *testing.T) {
	for _, tc := range []struct {
		p   arima.Parameters
		err error
	}{
		{arima.Parameters{AR: []float64{1}, InnovationVariance: 1, Order: arima.Order{AR: 1}}, arima.ErrNonStationary},
		{arima.Parameters{MA: []float64{1}, InnovationVariance: 1, Order: arima.Order{MA: 1}}, arima.ErrNonInvertible},
		{arima.Parameters{AR: []float64{-1}, InnovationVariance: 1, Order: arima.Order{AR: 1}}, arima.ErrNonStationary},
		{arima.Parameters{MA: []float64{-1}, InnovationVariance: 1, Order: arima.Order{MA: 1}}, arima.ErrNonInvertible},
	} {
		_, err := arima.New(tc.p)
		mustErrorIs(t, err, tc.err)
	}
}

// TestAR08CausalOriginAndOwnership guards against look-ahead and shared slices.
func TestAR08CausalOriginAndOwnership(t *testing.T) {
	p := arima.Parameters{AR: []float64{0.3}, MA: []float64{0.2}, Drift: 0.4, InnovationVariance: 2, Order: arima.Order{AR: 1, MA: 1}}
	m := mustModel(t, p)
	history := []float64{2, 4, 5, 3, 8, 7}
	f, err := m.Forecast(context.Background(), history, 3)
	if err != nil {
		t.Fatal(err)
	}
	history = append(history, 1000, 2000)
	g, err := m.Forecast(context.Background(), history[:6], 3)
	if err != nil {
		t.Fatal(err)
	}
	mustVector(t, f.Mean, g.Mean)
	mustVector(t, f.Variance, g.Variance)
	id := m.ID()
	p.AR[0] = 1
	p.MA[0] = 1
	copyP := m.Parameters()
	copyP.AR[0] = 1
	copyP.MA[0] = 1
	if m.ID() != id || m.Parameters().AR[0] != 0.3 {
		t.Fatal("model mutated")
	}
	f.Mean[0] = 999
	if g.Mean[0] == 999 {
		t.Fatal("forecast alias")
	}
}

// TestAR09Singular prevents attempting a logarithm of zero effective SSE.
func TestAR09Singular(t *testing.T) {
	_, err := arima.Fit(context.Background(), []float64{5, 5, 5, 5}, arima.Options{IncludeDrift: true})
	mustErrorIs(t, err, arima.ErrSingular)
}

// TestResultIsNilInterface checks typed-nil detection and pointer-only nilability.
func TestResultIsNilInterface(t *testing.T) {
	var fit *arima.FitResult
	var forecast *arima.ForecastResult
	mustNilState(t, fit, true, "fit")
	mustNilState(t, forecast, true, "forecast")
	mustNilState(t, &arima.FitResult{}, false, "empty fit")
	mustNilState(t, &arima.ForecastResult{}, false, "empty forecast")
	if !isNilLike(fit) {
		t.Fatal("nilable")
	}
}

// TestErrorPrecedence verifies context, options and sample-validation ordering.
func TestErrorPrecedence(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := arima.Fit(ctx, nil, arima.Options{Order: arima.Order{AR: 9}})
	mustErrorIs(t, err, context.Canceled)
	_, err = arima.Fit(nil, nil, arima.Options{})
	mustErrorIs(t, err, arima.ErrInvalidOption)
	_, err = arima.Fit(context.Background(), nil, arima.Options{})
	mustErrorIs(t, err, arima.ErrNilInput)
	_, err = arima.Fit(context.Background(), []float64{}, arima.Options{})
	mustErrorIs(t, err, arima.ErrEmptyInput)
	_, err = arima.Fit(context.Background(), []float64{math.NaN(), 1}, arima.Options{})
	mustErrorIs(t, err, arima.ErrNonFinite)
	_, err = arima.Fit(context.Background(), []float64{1, 2, 3}, arima.Options{Order: arima.Order{AR: 8}})
	mustErrorIs(t, err, arima.ErrInsufficientData)
	_, err = arima.New(arima.Parameters{InnovationVariance: 1, Order: arima.Order{Difference: 3}})
	mustErrorIs(t, err, arima.ErrInvalidOption)
	if errors.Is(err, arima.ErrSingular) {
		t.Fatal("wrong sentinel")
	}
}
