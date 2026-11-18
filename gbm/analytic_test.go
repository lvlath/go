// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm_test uses independent closed-form Ito solutions as math oracles.
package gbm_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/rng"
)

// TestIrregularGridAnalyticZeroNormal tests the cumulative Ito drift on uneven dt.
func TestIrregularGridAnalyticZeroNormal(t *testing.T) {
	times := []float64{0, 0.125, 0.625, 2.0, 2.125, 3.5}
	params := gbm.Parameters{Initial: 120, Drift: 0.08, Volatility: 0.3, Unit: gbm.TimeYears}
	source := &scripted{}
	result, err := gbm.Simulate(context.Background(), params,
		gbm.TimeGrid{Values: times, Unit: params.Unit}, source)
	if err != nil {
		t.Fatal(err)
	}
	for index, elapsed := range times {
		// Closed-form independent oracle: Z_i=0 for every stochastic interval.
		expected := params.Initial * math.Exp((params.Drift-0.5*params.Volatility*params.Volatility)*elapsed)
		mustNear(t, result.Values[index], expected, 1e-13)
	}
	mustEqualInt(t, source.count, len(times)-1, "irregular grid RNG calls")
}

// TestNonzeroNormalSequenceUsesEveryIrregularStep checks order, not just count.
func TestNonzeroNormalSequenceUsesEveryIrregularStep(t *testing.T) {
	times := []float64{0, 0.125, 0.5, 0.625, 1.5}
	normals := []float64{-0.4, 1.3, 0.8, -1.2}
	params := gbm.Parameters{Initial: 72, Drift: -0.015, Volatility: 0.35, Unit: gbm.TimeYears}
	result, err := gbm.Simulate(context.Background(), params,
		gbm.TimeGrid{Values: times, Unit: params.Unit}, &scripted{draws: normals})
	if err != nil {
		t.Fatal(err)
	}

	// Use the ordinary closed-form Ito product as an independent oracle only
	// within a safely representable and well-conditioned moderate input regime.
	logReturn := 0.0
	for index, z := range normals {
		dt := times[index+1] - times[index]
		logReturn += (params.Drift-0.5*params.Volatility*params.Volatility)*dt +
			params.Volatility*math.Sqrt(dt)*z
		mustNear(t, result.Values[index+1], params.Initial*math.Exp(logReturn), 1e-13)
	}
}

// TestDeterministicNegativeDrift verifies the physical domain allows depreciation.
func TestDeterministicNegativeDrift(t *testing.T) {
	result, err := gbm.Simulate(context.Background(),
		gbm.Parameters{Initial: 100, Drift: -0.2},
		gbm.TimeGrid{Values: []float64{0, 0.5, 2, 5}}, &scripted{})
	if err != nil {
		t.Fatal(err)
	}
	mustNear(t, result.Values[3], 100*math.Exp(-1), 1e-13)
}

// TestFixedSeedMomentSmoke checks that the simulated distribution is consistent
// with E[S_T]=S0*exp(mu*T) and Var[S_T]=S0^2*exp(2mu*T)*expm1(sigma^2*T).
// Wide deterministic bounds avoid flaky acceptance based on a single 5% test.
func TestFixedSeedMomentSmoke(t *testing.T) {
	const paths = 4096
	params := gbm.Parameters{Initial: 100, Drift: 0.06, Volatility: 0.2, Unit: gbm.TimeYears}
	grid := gbm.TimeGrid{Values: []float64{0, 1}, Unit: gbm.TimeYears}
	var total, totalSquares float64
	for id := uint64(0); id < paths; id++ {
		result, err := gbm.Simulate(context.Background(), params, grid, rng.NewStream(14537, id))
		if err != nil {
			t.Fatal(err)
		}
		value := result.Values[1]
		total += value
		totalSquares += value * value
	}
	mean := total / paths
	variance := totalSquares/paths - mean*mean
	expectedMean := 100 * math.Exp(0.06)
	expectedVariance := 10000 * math.Exp(0.12) * math.Expm1(0.04)
	if math.Abs(mean-expectedMean) > 2.5 || math.Abs(variance-expectedVariance) > 140 {
		t.Fatalf("moments mean=%g variance=%g; expected mean=%g variance=%g",
			mean, variance, expectedMean, expectedVariance)
	}
}
