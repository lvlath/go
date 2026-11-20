// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf_test documents runnable PACF workflows on deterministic domain fixtures.
// Each example builds a realistic-sized signal, executes the public API, and consumes
// numerical evidence without treating finite-sample PACF as a statistical proof.
package pacf_test

import (
	"context"
	"fmt"
	"math"
	"math/rand"

	"github.com/lvlath/go/pacf"
)

// ExampleComputeCentered_arOrderIdentification explores an AR order for an equipment
// vibration residual model rather than assuming an exact finite-sample cutoff.
//
// Scenario:
//   - A monitor collects thousands of evenly spaced vibration residual samples.
//   - The latent process follows a stable AR(2) law with known coefficients solely
//     for fixture construction; the estimator is not given those coefficients.
//   - A lag cutoff is an exploratory model-order candidate, not a formal test.
//
// Process:
//   - Generate a reproducible AR(2) window with independent Gaussian innovations.
//   - Estimate the centered PACF through lag 12 and inspect all nonzero lags.
//   - Select the largest lag exceeding an explicit 3/sqrt(n) screening heuristic.
//
// AI-Hints:
//   - Estimate uncertainty and validate against held-out data before choosing ARIMA order.
//   - Never claim exact zero sample PACF beyond the true process order.
func ExampleComputeCentered_arOrderIdentification() {
	const observations = 8192
	const maxLag = 12
	const firstAR = 0.6
	const secondAR = -0.2

	// A private fixed-seed generator makes the demonstration reproducible.
	rng := rand.New(rand.NewSource(27))
	vibration := make([]float64, observations)
	for t := 2; t < len(vibration); t++ {
		vibration[t] = firstAR*vibration[t-1] + secondAR*vibration[t-2] + rng.NormFloat64()
	}

	profile, err := pacf.ComputeCentered(context.Background(), vibration, maxLag)
	if err != nil {
		fmt.Println("PACF estimation failed")
		return
	}

	// The screening threshold is an explicit engineering heuristic, not a p-value.
	cutoff := 3 / math.Sqrt(float64(profile.Count))
	candidate := 0
	for lag := 1; lag <= maxLag; lag++ {
		if math.Abs(profile.Values[lag]) > cutoff {
			candidate = lag
		}
	}
	fmt.Printf("vibration samples=%d AR candidate=%d PACF(1)=%.3f PACF(2)=%.3f\n",
		profile.Count, candidate, profile.Values[1], profile.Values[2])
	// Output:
	// vibration samples=8192 AR candidate=2 PACF(1)=0.502 PACF(2)=-0.192
}

// ExampleCompute_residualWhitening compares serial dependence before and after
// a calibrated AR(1) filter used to monitor industrial compressor residuals.
//
// Scenario:
//   - A maintenance pipeline already has a fixed AR coefficient learned from training.
//   - Raw compressor telemetry is serially dependent; filter innovations should not be.
//   - A whiteness diagnostic highlights remaining one-step memory, not independence.
//
// Process:
//   - Generate correlated telemetry with a fixed AR coefficient and seeded innovations.
//   - Compute centered lag-one PACF on the raw signal and filtered residuals separately.
//   - Check whether the resulting residual lag-one coefficient is small.
//
// AI-Hints:
//   - Never fit the AR coefficient on the same held-out residual window used for evaluation.
//   - A low PACF at one lag is not a replacement for broader residual diagnostics.
func ExampleCompute_residualWhitening() {
	const observations = 4096
	const calibratedAR = 0.78

	rng := rand.New(rand.NewSource(12))
	telemetry := make([]float64, observations)
	innovations := make([]float64, observations)
	for t := 1; t < observations; t++ {
		innovations[t] = rng.NormFloat64()
		telemetry[t] = calibratedAR*telemetry[t-1] + innovations[t]
	}

	// The first time step has no previous observation, so omit it from both windows.
	before, err := pacf.ComputeCentered(context.Background(), telemetry[1:], 6)
	if err != nil {
		fmt.Println("raw diagnostic failed")
		return
	}
	after, err := pacf.ComputeCentered(context.Background(), innovations[1:], 6)
	if err != nil {
		fmt.Println("residual diagnostic failed")
		return
	}

	fmt.Printf("compressor raw lag1=%.3f filtered lag1=%.3f residual small=%t\n",
		before.Values[1], after.Values[1], math.Abs(after.Values[1]) < 0.05)
	// Output:
	// compressor raw lag1=0.777 filtered lag1=0.005 residual small=true
}

// ExampleComputeUncentered_orderFlowMemory quantifies memory in fixed-zero
// signed order-book imbalance used by a market microstructure monitoring system.
//
// Scenario:
//   - Signed imbalance is positive for buy pressure and negative for sell pressure.
//   - Physical zero means balanced demand, so an arbitrary sample-mean subtraction
//     would change the intended reference rather than improve the estimator.
//   - The fixture includes persistent imbalance but does not assert a trading edge.
//
// Process:
//   - Build deterministic buy/sell volumes and a bounded persistent imbalance.
//   - Use the explicitly uncentered common-denominator PACF estimator.
//   - Compare lag-one memory with the reduction in linear prediction variance.
//
// AI-Hints:
//   - Sampling frequency and order-book event-time policy determine physical lag units.
//   - PACF is a dependence diagnostic, not a market-impact or profitability model.
func ExampleComputeUncentered_orderFlowMemory() {
	const events = 480
	imbalance := make([]float64, events)
	rng := rand.New(rand.NewSource(101))
	for t := 1; t < events; t++ {
		// Fixed-seed purchase and sale volumes stand in for reproducible event snapshots.
		buyVolume := 100 + rng.Intn(900)
		sellVolume := 100 + rng.Intn(900)
		instant := float64(buyVolume-sellVolume) / float64(buyVolume+sellVolume)
		imbalance[t] = 0.75*imbalance[t-1] + 0.25*instant
	}

	profile, err := pacf.ComputeUncentered(context.Background(), imbalance, 10)
	if err != nil {
		fmt.Println("order-flow diagnostic failed")
		return
	}

	varianceReduction := 1 - profile.PredictionVariance[1]/profile.PredictionVariance[0]
	fmt.Printf("order-flow events=%d PACF(1)=%.3f lag1 variance reduction=%.1f%%\n",
		profile.Count, profile.Values[1], 100*varianceReduction)
	// Output:
	// order-flow events=480 PACF(1)=0.757 lag1 variance reduction=57.2%
}

// ExampleCompute_predictionVariance analyzes how additional autoregressive
// orders change one-step linear prediction uncertainty in a sensor residual series.
//
// Scenario:
//   - Two residual lags influence a rotating-machine vibration trace.
//   - The variance curve supplies a physically interpretable in-window fit diagnostic.
//   - Diminishing gains above order two invite a smaller forecasting candidate.
//
// Process:
//   - Synthesize a reproducible AR(2) trace in fixed vibration units.
//   - Estimate the centered profile and calculate variance reductions by order.
//   - Report the first two improvements without promising held-out risk reduction.
//
// AI-Hints:
//   - V[m] is an in-sample linear prediction-error variance, not forecast RMSE.
//   - Compare candidate orders with held-out forecast errors before deployment.
func ExampleCompute_predictionVariance() {
	const samples = 6000
	rng := rand.New(rand.NewSource(43))
	residual := make([]float64, samples)
	for t := 2; t < samples; t++ {
		residual[t] = 0.5*residual[t-1] - 0.35*residual[t-2] + 0.8*rng.NormFloat64()
	}

	result, err := pacf.ComputeCentered(context.Background(), residual, 8)
	if err != nil {
		fmt.Println("variance diagnostic failed")
		return
	}

	firstGain := 100 * (1 - result.PredictionVariance[1]/result.PredictionVariance[0])
	secondGain := 100 * (1 - result.PredictionVariance[2]/result.PredictionVariance[1])
	fmt.Printf("vibration first-order gain=%.1f%% second-order gain=%.1f%%\n",
		firstGain, secondGain)
	// Output:
	// vibration first-order gain=13.3% second-order gain=11.6%
}

// ExampleComputeAtLag_sensorBiasPolicy demonstrates why nuisance DC offset
// must not silently become physical memory in a calibrated temperature channel.
//
// Scenario:
//   - A sensor reports a large fixed DC bias together with oscillating variation.
//   - The analyst wants the mean-adjusted temporal dependence, not the raw DC level.
//   - The same lag must be available from both the full result and its point query.
//
// Process:
//   - Build a 384-sample deterministic biased sensor trace.
//   - Compare uncentered and centered PACF on the same unchanged window.
//   - Verify ComputeAtLag agrees with the corresponding full-profile coefficient.
//
// AI-Hints:
//   - Specify centering by the physical meaning of the baseline, not by a magic default.
//   - A point query still computes every intermediate prediction order through that lag.
func ExampleComputeAtLag_sensorBiasPolicy() {
	const observations = 384
	sensor := make([]float64, observations)
	for i := range sensor {
		sensor[i] = 12 + 2.5*math.Sin(float64(i)*0.65) + 0.15*math.Cos(float64(i)*1.7)
	}

	raw, err := pacf.ComputeUncentered(context.Background(), sensor, 4)
	if err != nil {
		fmt.Println("raw PACF failed")
		return
	}
	centered, err := pacf.ComputeCentered(context.Background(), sensor, 4)
	if err != nil {
		fmt.Println("centered PACF failed")
		return
	}
	point, err := pacf.ComputeAtLag(context.Background(), sensor, 1, true)
	if err != nil {
		fmt.Println("point PACF failed")
		return
	}

	fmt.Printf("biased sensor raw lag1=%.3f centered lag1=%.3f point matches=%t\n",
		raw.Values[1], centered.Values[1], math.Abs(point-centered.Values[1]) < 1e-14)
	// Output:
	// biased sensor raw lag1=0.993 centered lag1=0.791 point matches=true
}
