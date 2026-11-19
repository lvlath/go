// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima_test provides deterministic operational examples of conditional ARIMA.
// All samples are illustrative engineering fixtures, not claimed historical market data.
package arima_test

import (
	"context"
	"fmt"
	"math"

	"github.com/lvlath/go/arima"
)

// ExampleFitWithDrift_quarterlyGDPPlanning models a normalized quarterly output index.
// Scenario: a public-sector planning team needs a four-quarter activity baseline
// with an anchored level forecast, not a mistaken accumulation starting at zero.
// Process:
//   - Calibrate ARIMA(1,1,0) with a differenced-space intercept on 32 index levels.
//   - Recover the long-run expected quarterly growth c/(1-phi), not simply c.
//   - Forecast four actual index levels and their innovation-only uncertainty.
//
// Limitation: the fixed illustrative series does not validate out-of-sample GDP accuracy.
func ExampleFitWithDrift_quarterlyGDPPlanning() {
	quarterlyIndex := []float64{
		100.0, 101.1, 102.6, 103.8, 104.5, 105.9, 107.3, 108.6,
		109.4, 110.8, 112.4, 113.5, 114.6, 116.2, 117.7, 118.9,
		120.1, 121.5, 122.6, 124.0, 125.8, 127.0, 128.0, 129.5,
		130.7, 132.2, 133.5, 134.8, 136.0, 137.6, 139.1, 140.5,
	}
	ctx := context.Background()
	fit, err := arima.FitWithDrift(ctx, quarterlyIndex, arima.Order{AR: 1, Difference: 1})
	if err != nil {
		fmt.Println("GDP calibration failed:", err)
		return
	}
	forecast, err := fit.Model.Forecast(ctx, quarterlyIndex, 4)
	if err != nil {
		fmt.Println("GDP projection failed:", err)
		return
	}
	parameters := fit.Model.Parameters()
	longRunGrowth := parameters.Drift / (1 - parameters.AR[0])
	upper95 := forecast.Mean[3] + 1.645*math.Sqrt(forecast.Variance[3])
	fmt.Printf("GDP growth %.2f index-points/quarter; Q+4 %.2f; one-sided-95 upper %.2f\n",
		longRunGrowth, forecast.Mean[3], upper95)
	// Output: GDP growth 1.31 index-points/quarter; Q+4 145.74; one-sided-95 upper 146.46
}

// ExampleModel_Forecast_liquidityRiskGate evaluates a fixed bid-ask spread model.
// Scenario: an execution service checks whether a normal-innovation upper tail
// exceeds a 1.80-basis-point liquidity budget before posting aggressive quotes.
// Process:
//   - Create a stationary, invertible ARMA(1,1) using preapproved coefficients.
//   - Condition its last innovations on 30 illustrative intraday spread snapshots.
//   - Use the marginal one-sided 95% tail as a review signal, not trading advice.
//
// Limitation: tails exclude calibration error, regime shifts and microstructure jumps.
func ExampleModel_Forecast_liquidityRiskGate() {
	spreadsBps := []float64{
		1.7, 1.6, 1.5, 1.8, 1.7, 1.6, 1.9, 1.7, 1.5, 1.6,
		1.8, 1.9, 1.6, 1.5, 1.7, 1.8, 1.6, 1.4, 1.6, 1.7,
		1.9, 1.8, 1.6, 1.5, 1.7, 1.8, 1.6, 1.5, 1.9, 2.0,
	}
	model, err := arima.New(arima.Parameters{
		AR: []float64{0.55}, MA: []float64{-0.30}, Drift: 0.72,
		InnovationVariance: 0.0225, Order: arima.Order{AR: 1, MA: 1},
	})
	if err != nil {
		fmt.Println("spread model invalid:", err)
		return
	}
	forecast, err := model.Forecast(context.Background(), spreadsBps, 6)
	if err != nil {
		fmt.Println("spread forecast failed:", err)
		return
	}
	upper95 := forecast.Mean[5] + 1.645*math.Sqrt(forecast.Variance[5])
	reviewQuote := upper95 > 1.80
	fmt.Printf("spread next %.3f bps; tick+6 upper95 %.3f bps; review=%t\n",
		forecast.Mean[0], upper95, reviewQuote)
	// Output: spread next 1.721 bps; tick+6 upper95 1.864 bps; review=true
}

// ExampleFitIntegrated_supportQueueCapacity projects an operational ticket backlog.
// Scenario: a support director forecasts a week of unassigned incidents from
// daily totals and checks if the final backlog exceeds a staffing threshold.
// Process:
//   - Fit a drift-enabled random walk with the analytic ARIMA(0,1,0) shortcut.
//   - Forecast seven anchored counts and inspect the marginal variance in tickets^2.
//   - Flag a planning threshold without treating variance as a capacity guarantee.
//
// Limitation: weekends, seasonal demand and intervention policies are not modeled.
func ExampleFitIntegrated_supportQueueCapacity() {
	backlog := []float64{
		180, 184, 191, 189, 196, 202, 207, 205, 213, 219,
		222, 229, 225, 232, 237, 243, 249, 246, 254, 260,
		263, 269, 276, 273, 281, 287, 292, 299,
	}
	fit, err := arima.FitIntegrated(context.Background(), backlog, 1, true)
	if err != nil {
		fmt.Println("queue calibration failed:", err)
		return
	}
	forecast, err := fit.Model.Forecast(context.Background(), backlog, 7)
	if err != nil {
		fmt.Println("queue forecast failed:", err)
		return
	}
	threshold := 325.0
	fmt.Printf("queue drift %.2f tickets/day; day+7 %.1f; sigma %.1f; plan-extra=%t\n",
		fit.Model.Parameters().Drift, forecast.Mean[6], math.Sqrt(forecast.Variance[6]), forecast.Mean[6] >= threshold)
	// Output: queue drift 4.41 tickets/day; day+7 329.9; sigma 9.7; plan-extra=true
}

// ExampleModel_Forecast_industrialEnergyAcceleration forecasts factory load levels.
// Scenario: a utility scheduler needs the next seven daily MWh requirements
// while preserving both yesterday's level and yesterday's first difference.
// Process:
//   - Fix a calibrated ARIMA(0,2,1) acceleration model with MA shock response.
//   - Recover future levels through two anchored integrations.
//   - Estimate a one-sided innovation-only reserve and check feeder capacity.
//
// Limitation: shutdowns, weather and parameter risk require separate controls.
func ExampleModel_Forecast_industrialEnergyAcceleration() {
	mwh := []float64{
		410, 413, 417, 423, 430, 438, 447, 457, 469, 482,
		496, 511, 527, 544, 563, 582, 602, 624, 647, 671,
		696, 722, 749, 778, 807, 838, 870, 903, 937, 973,
	}
	model, err := arima.New(arima.Parameters{
		MA: []float64{0.25}, Drift: 0.10, InnovationVariance: 0.49,
		Order: arima.Order{Difference: 2, MA: 1},
	})
	if err != nil {
		fmt.Println("energy model invalid:", err)
		return
	}
	forecast, err := model.Forecast(context.Background(), mwh, 7)
	if err != nil {
		fmt.Println("energy forecast failed:", err)
		return
	}
	reserve95 := forecast.Mean[6] + 1.645*math.Sqrt(forecast.Variance[6])
	fmt.Printf("energy day+1 %.1f MWh; day+7 %.1f; reserve95 %.1f; over-1245=%t\n",
		forecast.Mean[0], forecast.Mean[6], reserve95, reserve95 > 1245)
	// Output: energy day+1 1009.5 MWh; day+7 1230.8; reserve95 1247.2; over-1245=true
}

// ExampleFitARMA_datacenterTemperature forecasts a stationary deviation series.
// Scenario: operations monitors a data-center intake-temperature correction
// after fan-control normalization, with zero as a physical equilibrium.
// Process:
//   - Fit zero-intercept AR(1) to 36 illustrative correction readings.
//   - Forecast three readings and a Gaussian upper-tail alarm threshold.
//   - Preserve the zero-intercept model instead of silently centering the data.
//
// Limitation: calibrated innovation variance does not cover sensor failures.
func ExampleFitARMA_datacenterTemperature() {
	correctionCelsius := []float64{
		0.418, 0.218, 0.294, 0.096, -0.136, 0.290, 0.146, 0.209, -0.050,
		-0.113, 0.264, -0.036, -0.018, -0.263, -0.136, 0.155, -0.211, -0.106,
		-0.230, 0.061, 0.212, -0.158, 0.059, -0.003, 0.293, 0.209, -0.146,
		0.129, 0.028, 0.222, -0.060, -0.289, 0.050, -0.092, 0.047, -0.271,
	}
	fit, err := arima.FitARMA(context.Background(), correctionCelsius, 1, 0, false)
	if err != nil {
		fmt.Println("temperature calibration failed:", err)
		return
	}
	forecast, err := fit.Model.Forecast(context.Background(), correctionCelsius, 3)
	if err != nil {
		fmt.Println("temperature forecast failed:", err)
		return
	}
	upper95 := forecast.Mean[2] + 1.645*math.Sqrt(forecast.Variance[2])
	fmt.Printf("thermal phi %.2f; step+3 correction %.3f C; upper95 %.3f; inspect=%t\n",
		fit.Model.Parameters().AR[0], forecast.Mean[2], upper95, upper95 > 0.25)
	// Output: thermal phi 0.12; step+3 correction -0.000 C; upper95 0.289; inspect=true
}

// ExampleModel_Parameters_releaseFingerprint demonstrates immutable model governance.
// Scenario: two services must use the same shock model revision while operators
// edit a detached configuration snapshot during a staged rollout.
// Process:
//   - Create a validated ARIMA(1,0,1) model and record its fingerprint.
//   - Mutate original and returned slices, then verify the deployed model remains intact.
//   - Produce a next-step forecast without introducing shared mutable state.
//
// Limitation: a content hash detects revision equality but is not a digital signature.
func ExampleModel_Parameters_releaseFingerprint() {
	ar := []float64{0.42}
	ma := []float64{-0.16}
	model, err := arima.New(arima.Parameters{
		AR: ar, MA: ma, Drift: 0.87, InnovationVariance: 0.09,
		Order: arima.Order{AR: 1, MA: 1},
	})
	if err != nil {
		fmt.Println("release invalid:", err)
		return
	}
	fingerprint := model.ID()
	ar[0], ma[0] = 0.99, 0.99
	staged := model.Parameters()
	staged.AR[0], staged.MA[0] = 0.01, 0.01

	levels := []float64{1.32, 1.53, 1.57, 1.49, 1.68, 1.74, 1.60, 1.65,
		1.72, 1.81, 1.67, 1.76, 1.70, 1.88, 1.79, 1.75}
	forecast, err := model.ForecastNext(context.Background(), levels)
	if err != nil {
		fmt.Println("deployment forecast failed:", err)
		return
	}
	fmt.Printf("release sha256=%t unchanged=%t forecast-next=%.3f\n",
		len(fingerprint) == 64, model.ID() == fingerprint && model.Parameters().AR[0] == 0.42,
		forecast.Mean[0])
	// Output: release sha256=true unchanged=true forecast-next=1.580
}
