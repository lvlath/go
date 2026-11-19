// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
// Package arima_test checks marginal variance growth and anchored forecast mechanics.
// This file includes analytic AR/MA/integration results and numeric overflow classification.
package arima_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/arima"
)

// TestForecastVarianceMonotone checks cumulative nonnegative impulse variance increments.
func TestForecastVarianceMonotone(t *testing.T) {
	for d := 0; d <= 2; d++ {
		m := mustModel(t, arima.Parameters{AR: []float64{0.7}, MA: []float64{-0.3}, InnovationVariance: 1.25, Order: arima.Order{AR: 1, Difference: d, MA: 1}})
		r, e := m.Forecast(context.Background(), []float64{10, 11, 13, 16, 20}, 200)
		if e != nil {
			t.Fatal(e)
		}
		for i := 1; i < len(r.Variance); i++ {
			if r.Variance[i] < r.Variance[i-1] || !(!math.IsNaN(r.Variance[i]) && !math.IsInf(r.Variance[i], 0)) {
				t.Fatalf("d=%d i=%d variance", d, i)
			}
		}
	}
}

// TestForecastRejectsOverflow rejects unrepresentable anchored level growth.
func TestForecastRejectsOverflow(t *testing.T) {
	m := mustModel(t, arima.Parameters{Drift: 1e308, InnovationVariance: 1, Order: arima.Order{Difference: 1}})
	_, err := m.Forecast(context.Background(), []float64{1e308, 1e308}, 3)
	mustErrorIs(t, err, arima.ErrOverflow)
}

// TestAR1IntegratedInterceptIsNotMeanIncrement checks the differenced-space
// intercept and last-level anchor independently of the optimizer and its tests.
func TestAR1IntegratedInterceptIsNotMeanIncrement(t *testing.T) {
	model := mustModel(t, arima.Parameters{
		AR: []float64{0.5}, Drift: 1, InnovationVariance: 4,
		Order: arima.Order{AR: 1, Difference: 1},
	})
	forecast, err := model.Forecast(context.Background(), []float64{0, 10, 20}, 3)
	if err != nil {
		t.Fatal(err)
	}
	// Last difference=10, so future differences are 6,4,3, NOT c=1.
	// Actual last level is 20, therefore reconstructed levels are 26,30,33.
	mustVector(t, forecast.Mean, []float64{26, 30, 33})
	// Integrated psi=[1,1.5,1.75]; cumulative sigma^2*psi^2 gives 4,13,25.25.
	mustVector(t, forecast.Variance, []float64{4, 13, 25.25})
}

// TestDoubleIntegrationFourthHorizon rejects resetting slope or accumulating
// only the predicted acceleration instead of both original historical anchors.
func TestDoubleIntegrationFourthHorizon(t *testing.T) {
	model := mustModel(t, arima.Parameters{
		Drift: 2, InnovationVariance: 1,
		Order: arima.Order{Difference: 2},
	})
	forecast, err := model.Forecast(context.Background(), []float64{10, 13, 19}, 4)
	if err != nil {
		t.Fatal(err)
	}
	mustVector(t, forecast.Mean, []float64{27, 37, 49, 63})
	mustVector(t, forecast.Variance, []float64{1, 5, 14, 30})
}
