// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima_test maintains independent scalar oracles for conditional likelihood.
// The oracle implements the published equation directly and never calls private kernels.
package arima_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/arima"
)

// conditionalOracle evaluates the public sign and burn-in convention independently.
// A scalar residual loop is intentionally kept separate from the production implementation.
func conditionalOracle(levels []float64, parameters arima.Parameters) (float64, float64) {
	y := append([]float64(nil), levels...)
	for d := 0; d < parameters.Order.Difference; d++ {
		for i := 0; i < len(y)-1; i++ {
			y[i] = y[i+1] - y[i]
		}
		y = y[:len(y)-1]
	}
	burnIn := maxOrder(parameters.Order.AR, parameters.Order.MA)
	e := make([]float64, len(y))
	sse := 0.0
	for t := burnIn; t < len(y); t++ {
		prediction := parameters.Drift
		for lag, phi := range parameters.AR {
			prediction += phi * y[t-lag-1]
		}
		for lag, theta := range parameters.MA {
			prediction += theta * e[t-lag-1]
		}
		e[t] = y[t] - prediction
		sse += e[t] * e[t]
	}
	n := float64(len(y) - burnIn)
	variance := sse / n
	return -n / 2 * (math.Log(2*math.Pi) + 1 + math.Log(variance)), variance
}

func TestIndependentProfileLikelihoodForFittedARMA(t *testing.T) {
	data := syntheticARMA(240)
	for _, order := range []arima.Order{{AR: 1}, {MA: 1}, {AR: 1, MA: 1}, {AR: 2, MA: 2}} {
		result, err := arima.Fit(context.Background(), data, arima.Options{
			Order: order, IncludeDrift: true, MaxIterations: 400, Tolerance: 1e-6,
		})
		if err != nil && !errors.Is(err, arima.ErrNotConverged) {
			t.Fatal(err)
		}
		if result == nil || result.Model == nil {
			t.Fatal("no candidate model")
		}
		gotParams := result.Model.Parameters()
		wantLL, wantVariance := conditionalOracle(data, gotParams)
		mustClose(t, result.LogLikelihood, wantLL)
		mustClose(t, gotParams.InnovationVariance, wantVariance)
		if result.Diagnostics.EffectiveN != len(data)-maxOrder(order.AR, order.MA) {
			t.Fatal("burn-in samples entered effective likelihood")
		}
	}
}

func TestSchurPolynomialProductsAndNearUnitBoundary(t *testing.T) {
	// Stable AR roots are 2 and 4, from (1-0.5z)(1-0.25z).
	good := arima.Parameters{AR: []float64{0.75, -0.125}, InnovationVariance: 1, Order: arima.Order{AR: 2}}
	mustModel(t, good)
	// Replacing the second factor by (1-z) gives a strict unit root.
	good.AR = []float64{1.5, -0.5}
	_, err := arima.New(good)
	mustErrorIs(t, err, arima.ErrNonStationary)

	// Stable MA roots are 2 and 4, from (1-0.5z)(1-0.25z).
	ma := arima.Parameters{MA: []float64{-0.75, 0.125}, InnovationVariance: 1, Order: arima.Order{MA: 2}}
	mustModel(t, ma)
	ma.MA = []float64{-1.5, 0.5}
	_, err = arima.New(ma)
	mustErrorIs(t, err, arima.ErrNonInvertible)

	near := arima.Parameters{AR: []float64{math.Nextafter(1, 0)}, InnovationVariance: 1, Order: arima.Order{AR: 1}}
	mustModel(t, near) // Strictly stable, not rounded or clipped to one.
	near.AR[0] = 1
	_, err = arima.New(near)
	mustErrorIs(t, err, arima.ErrNonStationary)
}

func TestAnalyticWhiteNoiseSmallVarianceNotFalseSingular(t *testing.T) {
	observations := []float64{0, 1e-160, 3e-160, 2e-160, -1e-160}
	result, err := arima.FitWithDrift(context.Background(), observations, arima.Order{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Model.Parameters().InnovationVariance <= 0 || math.IsInf(result.LogLikelihood, 0) {
		t.Fatal("positive representable variance was treated as singular")
	}
}

func TestGradientStationarityAgainstIndependentFiniteDifferences(t *testing.T) {
	data := syntheticARMA(320)
	result, err := arima.Fit(context.Background(), data, arima.Options{
		Order: arima.Order{AR: 1, MA: 1}, IncludeDrift: true, MaxIterations: 400, Tolerance: 1e-7,
	})
	if err != nil {
		t.Fatalf("optimizer did not establish stationarity: %v", err)
	}
	parameters := result.Model.Parameters()
	for k := 0; k < 3; k++ {
		plus, minus := parameters, parameters
		plus.AR = append([]float64(nil), parameters.AR...)
		minus.AR = append([]float64(nil), parameters.AR...)
		plus.MA = append([]float64(nil), parameters.MA...)
		minus.MA = append([]float64(nil), parameters.MA...)
		switch k {
		case 0:
			plus.AR[0] += 1e-5
			minus.AR[0] -= 1e-5
		case 1:
			plus.MA[0] += 1e-5
			minus.MA[0] -= 1e-5
		default:
			plus.Drift += 1e-5
			minus.Drift -= 1e-5
		}
		plusLL, _ := conditionalOracle(data, plus)
		minusLL, _ := conditionalOracle(data, minus)
		finiteDifference := (plusLL - minusLL) / (2e-5)
		if math.Abs(finiteDifference) > 1e-3 {
			t.Fatalf("physical coordinate %d: finite-difference likelihood slope=%g", k, finiteDifference)
		}
	}
}
