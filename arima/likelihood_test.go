// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
// Package arima_test checks likelihood signs and conditional initialization with scalar oracles.
// This file uses hand-computed residual recurrences independent of optimizer logic.
package arima_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/arima"
)

// Independent oracle for conditional Gaussian fit with known AR/MA coefficients:
// compute predictions and residuals by scalar hand recurrence, excluding burn-in.
// TestConditionalARMAInnovationOracle checks the sign and burn-in of lagged MA shocks.
func TestConditionalARMAInnovationOracle(t *testing.T) {
	p := arima.Parameters{AR: []float64{0.5}, MA: []float64{0.25}, Drift: 1, InnovationVariance: 2, Order: arima.Order{AR: 1, MA: 1}}
	m := mustModel(t, p)
	r, e := m.Forecast(context.Background(), []float64{2, 4, 6, 3}, 2)
	if e != nil {
		t.Fatal(e)
	}
	// e0=0, e1=4-1-1=2, e2=6-1-2-.5=2.5, e3=3-1-3-.625=-1.625.
	// y4=1+0.5*3+0.25*(-1.625)=2.09375; y5=1+0.5*y4.
	mustVector(t, r.Mean, []float64{2.09375, 2.046875})
	mustVector(t, r.Variance, []float64{2, 2 * (1 + 0.75*0.75)})
}

// TestAnalyticLikelihoodOracle compares closed-form Gaussian profile likelihood.
func TestAnalyticLikelihoodOracle(t *testing.T) {
	r, e := arima.Fit(context.Background(), []float64{-2, 0, 2, 4, 6}, arima.Options{IncludeDrift: true})
	if e != nil {
		t.Fatal(e)
	}
	mustClose(t, r.Model.Parameters().Drift, 2)
	mustClose(t, r.Model.Parameters().InnovationVariance, 8)
	want := -2.5 * (math.Log(2*math.Pi) + 1 + math.Log(8))
	mustClose(t, r.LogLikelihood, want)
}
