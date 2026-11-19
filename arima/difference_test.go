// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
// Package arima_test validates inverse differencing at nonzero historical origins.
// This file protects order-one and order-two level anchors from off-by-one drift.
package arima_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/arima"
)

// TestAnchorsExactWithNonzeroAcceleration verifies historical slope preservation.
func TestAnchorsExactWithNonzeroAcceleration(t *testing.T) {
	m := mustModel(t, arima.Parameters{Drift: 2, InnovationVariance: 1, Order: arima.Order{Difference: 2}})
	r, e := m.Forecast(context.Background(), []float64{10, 13, 19}, 3)
	if e != nil {
		t.Fatal(e)
	}
	mustVector(t, r.Mean, []float64{27, 37, 49})
	mustVector(t, r.Variance, []float64{1, 5, 14})
}

// TestFirstDifferenceDrift accumulates mean differences from the actual last level.
func TestFirstDifferenceDrift(t *testing.T) {
	m := mustModel(t, arima.Parameters{Drift: 3, InnovationVariance: 1, Order: arima.Order{Difference: 1}})
	r, e := m.Forecast(context.Background(), []float64{100, 104}, 3)
	if e != nil {
		t.Fatal(e)
	}
	mustVector(t, r.Mean, []float64{107, 110, 113})
}
