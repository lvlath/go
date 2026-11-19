// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
// Package arima_test exercises accepted L-BFGS behavior using deterministic stationary fixtures.
// This file checks nonconvergence ownership and polynomial parameter validation.
package arima_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/arima"
)

// syntheticARMA builds a reproducible ARMA-like fixture without global randomness.
func syntheticARMA(n int) []float64 {
	y := make([]float64, n)
	previousResidual := 0.0
	for i := range y {
		residual := 0.4*math.Sin(float64(i)*0.63) + 0.6*math.Cos(float64(i)*1.71)
		y[i] = 1.2 + residual + 0.35*previousResidual
		if i > 0 {
			y[i] += 0.45 * y[i-1]
		}
		previousResidual = residual
	}
	return y
}

// TestFitARMARegression checks accepted fitting trajectories across polynomial orders.
func TestFitARMARegression(t *testing.T) {
	data := syntheticARMA(240)
	for _, order := range []arima.Order{{AR: 1, MA: 1}, {AR: 2, MA: 2}, {AR: 1, MA: 0}, {AR: 0, MA: 1}, {AR: 8, MA: 8}} {
		t.Run("fit", func(t *testing.T) {
			result, err := arima.Fit(context.Background(), data, arima.Options{Order: order, IncludeDrift: true, MaxIterations: 300, Tolerance: 1e-6})
			if err != nil && !errors.Is(err, arima.ErrNotConverged) {
				t.Fatal(err)
			}
			if result == nil || result.Model == nil {
				t.Fatal("no accepted model")
			}
			if result.Diagnostics.EffectiveN != len(data)-maxOrder(order.AR, order.MA) {
				t.Fatal("bad sample")
			}
			for i := 1; i < len(result.Diagnostics.Objective); i++ {
				if result.Diagnostics.Objective[i] > result.Diagnostics.Objective[i-1]+1e-8 {
					t.Fatal("objective increased")
				}
			}
			p := result.Model.Parameters()
			if len(p.AR) != order.AR || len(p.MA) != order.MA || p.InnovationVariance <= 0 {
				t.Fatal("bad model")
			}
			fc, e := result.Model.Forecast(context.Background(), data, 5)
			if e != nil {
				t.Fatal(e)
			}
			if len(fc.Mean) != 5 {
				t.Fatal("wrong horizon")
			}
		})
	}
}

// maxOrder is a test-only independent maximum used by likelihood oracles.
func maxOrder(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// TestNonConvergencePartial verifies the last accepted model survives an iteration limit.
func TestNonConvergencePartial(t *testing.T) {
	result, err := arima.Fit(context.Background(), syntheticARMA(100), arima.Options{Order: arima.Order{AR: 1, MA: 1}, IncludeDrift: true, MaxIterations: 1, Tolerance: 1e-15})
	mustErrorIs(t, err, arima.ErrNotConverged)
	if result == nil || result.IsNil() || result.Diagnostics.Converged {
		t.Fatal("missing partial valid model")
	}
	if result.Diagnostics.Iterations > 1 {
		t.Fatal("iteration budget violated")
	}
}

// TestStrictValidationAndDomain rejects invalid orders, parameters and histories.
func TestStrictValidationAndDomain(t *testing.T) {
	ctx := context.Background()
	invalidOrders := []arima.Order{{AR: -1}, {AR: 9}, {MA: -1}, {MA: 9}, {Difference: -1}, {Difference: 3}}
	for _, o := range invalidOrders {
		_, e := arima.Fit(ctx, []float64{1, 2, 3, 4}, arima.Options{Order: o})
		mustErrorIs(t, e, arima.ErrInvalidOption)
	}
	for _, opts := range []arima.Options{{MaxIterations: -1}, {Tolerance: -1}, {Tolerance: math.Inf(1)}, {Tolerance: math.NaN()}} {
		_, e := arima.Fit(ctx, []float64{1, 2, 3, 4}, opts)
		mustErrorIs(t, e, arima.ErrInvalidOption)
	}
	for _, p := range []arima.Parameters{
		{AR: []float64{0.3}, Order: arima.Order{}, InnovationVariance: 1},
		{AR: []float64{math.NaN()}, Order: arima.Order{AR: 1}, InnovationVariance: 1},
		{AR: []float64{math.Inf(1)}, Order: arima.Order{AR: 1}, InnovationVariance: 1},
		{MA: []float64{math.NaN()}, Order: arima.Order{MA: 1}, InnovationVariance: 1},
		{Drift: math.NaN(), InnovationVariance: 1},
		{InnovationVariance: math.NaN()},
		{InnovationVariance: math.Inf(1)},
		{InnovationVariance: 0},
	} {
		_, e := arima.New(p)
		if e == nil {
			t.Fatalf("accepted invalid parameters %+v", p)
		}
	}
	m := mustModel(t, arima.Parameters{InnovationVariance: 1})
	for _, tc := range []struct {
		history []float64
		h       int
		target  error
	}{
		{nil, 1, arima.ErrNilInput},
		{[]float64{}, 1, arima.ErrEmptyInput},
		{[]float64{math.NaN()}, 1, arima.ErrNonFinite},
		{[]float64{math.Inf(1)}, 1, arima.ErrNonFinite},
		{[]float64{1}, 0, arima.ErrInvalidOption},
		{[]float64{1}, -1, arima.ErrInvalidOption},
		{[]float64{1}, int(^uint(0) >> 1), arima.ErrOverflow},
	} {
		_, e := m.Forecast(ctx, tc.history, tc.h)
		mustErrorIs(t, e, tc.target)
	}
	var nilM *arima.Model
	_, e := nilM.Forecast(ctx, []float64{1}, 1)
	mustErrorIs(t, e, arima.ErrInvalidOption)
	_, e = m.Forecast(nil, []float64{1}, 1)
	mustErrorIs(t, e, arima.ErrInvalidOption)
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	_, e = m.Forecast(canceled, []float64{1}, 1)
	mustErrorIs(t, e, context.Canceled)
	m1 := mustModel(t, arima.Parameters{InnovationVariance: 1, Order: arima.Order{Difference: 2}})
	_, e = m1.Forecast(ctx, []float64{1, 2}, 1)
	mustErrorIs(t, e, arima.ErrInsufficientData)
	_, e = arima.Fit(ctx, []float64{1e308, -1e308, 1e308}, arima.Options{})
	mustErrorIs(t, e, arima.ErrOverflow)
	if nilM.ID() != "" || nilM.Parameters().InnovationVariance != 0 {
		t.Fatal("nil receiver")
	}
}

// TestSchurHigherOrder checks factored second-order stationary coefficients.
func TestSchurHigherOrder(t *testing.T) {
	// Polynomial (1-0.5z)(1-0.25z), both roots outside unit circle.
	p := arima.Parameters{AR: []float64{0.75, -0.125}, MA: []float64{-0.6, 0.08}, InnovationVariance: 1, Order: arima.Order{AR: 2, MA: 2}}
	m := mustModel(t, p)
	if m.Parameters().AR[1] != -0.125 {
		t.Fatal("wrong coefficients")
	}
	p.AR = []float64{1.4, -0.3} // Schur-invalid despite individually finite coefficients.
	_, err := arima.New(p)
	mustErrorIs(t, err, arima.ErrNonStationary)
}
