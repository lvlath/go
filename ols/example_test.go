// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols_test demonstrates realistic regression contracts using fixed,
// reproducible synthetic measurements. These are engineering illustrations,
// not live investment prices, certified sensor readings, or energy forecasts.
package ols_test

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ols"
)

// ExampleFit estimates the market exposure and monthly alpha of an asset from
// 24 synthetic monthly excess-return observations (CAPM-style regression).
//
// Scenario:
//   - A risk desk maintains a dated training window of market and asset excess
//     returns in decimal fractions, not raw prices or percentage-point values.
//   - A market-only factor model estimates sensitivity, residual tracking error,
//     and the forecast conditional on a separately supplied market scenario.
//   - SSE is an in-sample squared-return diagnostic, not a hypothesis test or a
//     confidence interval; autocorrelation and heteroskedasticity are not modeled.
//
// Process:
//   - Build the 24-by-1 factor matrix in chronological order.
//   - Fit an explicit intercept without inverting the normal equations.
//   - Consume beta, alpha, SSE, rank, and a heldout stress prediction.
//
// Complexity: O(n*q*q+q^3) time with q=2; O(n*q) transient space.
// AI-Hints: Never include the future scenario in training means or factor fitting.
func ExampleFit() {
	market := []float64{
		-.042, .021, -.013, .034, -.026, .041, .009, -.017,
		.028, -.033, .019, .006, .052, -.011, .025, -.038,
		.014, .031, -.024, .046, -.005, .023, -.019, .037,
	}
	// Small signed returns represent idiosyncratic realized variation. The
	// fixture is intentionally not an exact affine function of the factor.
	idiosyncratic := []float64{
		.0008, -.0005, .0002, -.0012, .0010, -.0008, .0003, .0004,
		-.0011, .0009, -.0004, .0007, .0005, -.0006, .0006, -.0003,
		.0012, -.0009, .0008, -.0010, .0005, -.0003, .0011, -.0004,
	}
	asset := make([]float64, len(market))
	for month, factor := range market {
		asset[month] = .0016 + 1.18*factor + idiosyncratic[month]
	}
	features, err := matrix.NewDense(len(market), 1)
	if err != nil {
		fmt.Println("matrix:", err)
		return
	}
	if err = features.Fill(market); err != nil {
		fmt.Println("market data:", err)
		return
	}

	fit, err := ols.FitDefault(context.Background(), features, asset)
	if err != nil {
		fmt.Println("CAPM fit:", err)
		return
	}
	beta, alpha := fit.Model.Coefficients()

	// A hypothetical -1.5% market excess return is *not* fitted into the
	// training sample. Predict uses the same frozen original-unit coefficients.
	stress, err := matrix.NewDense(1, 1)
	if err != nil {
		fmt.Println("stress matrix:", err)
		return
	}
	if err = stress.Set(0, 0, -.015); err != nil {
		fmt.Println("stress value:", err)
		return
	}
	forecast, err := fit.Model.Predict(context.Background(), stress)
	if err != nil {
		fmt.Println("stress prediction:", err)
		return
	}
	fmt.Printf("CAPM beta=%.2f alpha=%.1f bps/month SSE=%.6f rank=%d\n",
		beta[0], alpha*1e4, fit.ResidualSumSquares, fit.Rank)
	fmt.Printf("stress market=-1.50%% asset=%.2f%%\n", 100*forecast[0])

	// Output:
	// CAPM beta=1.16 alpha=17.6 bps/month SSE=0.000009 rank=2
	// stress market=-1.50% asset=-1.57%
}

// ExampleFit_sensorCalibration performs a two-factor bench calibration of an
// industrial pressure transducer with pressure gain and temperature drift.
//
// Scenario:
//   - 24 calibration points vary pressure (bar) and ambient temperature (C)
//     independently to identify the voltage gain and thermal sensitivity.
//   - A technician needs the voltage offset at a reference 25 C, even though
//     the model itself trains on *absolute* Celsius readings.
//   - A separate check compares scaled and unscaled QR fits in physical units.
//
// Process:
//   - Build a 24-by-2 design with noisy voltage measurements.
//   - Fit scaled and unscaled models; compare their original-unit responses.
//   - Predict a withheld 42-bar / 26-C point without refitting the scaler.
//
// Complexity: O(n*q*q+q^3) time with q=3 and O(n*q) work storage.
// AI-Hints: Use a real sensor uncertainty model before treating residuals as tolerances.
func ExampleFit_sensorCalibration() {
	pressureBar := []float64{
		0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55,
		60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115,
	}
	temperatureC := []float64{
		18, 30, 23, 34, 20, 28, 17, 32, 19, 27, 21, 35,
		22, 29, 16, 33, 24, 31, 26, 15, 36, 25, 29, 18,
	}
	voltageNoise := []float64{
		.002, -.001, .001, -.002, 0, .001, -.001, .002,
		-.002, .001, 0, .002, -.001, .001, -.002, 0,
		.002, -.001, .001, -.002, .001, 0, .002, -.001,
	}
	voltage := make([]float64, len(pressureBar))
	train, err := matrix.NewDense(len(pressureBar), 2)
	if err != nil {
		fmt.Println("matrix:", err)
		return
	}
	for i := range voltage {
		voltage[i] = .625 + .084*pressureBar[i] + .011*(temperatureC[i]-25) + voltageNoise[i]
		if err = train.Set(i, 0, pressureBar[i]); err != nil {
			fmt.Println("pressure:", err)
			return
		}
		if err = train.Set(i, 1, temperatureC[i]); err != nil {
			fmt.Println("temperature:", err)
			return
		}
	}

	scaled, err := ols.FitStandardized(context.Background(), train, voltage)
	if err != nil {
		fmt.Println("scaled calibration:", err)
		return
	}
	raw, err := ols.FitDefault(context.Background(), train, voltage)
	if err != nil {
		fmt.Println("raw calibration:", err)
		return
	}
	gainAndDrift, physicalIntercept := scaled.Model.Coefficients()
	zeroAt25C := math.FMA(25, gainAndDrift[1], physicalIntercept)

	holdout, err := matrix.NewDense(1, 2)
	if err != nil {
		fmt.Println("holdout matrix:", err)
		return
	}
	if err = holdout.Fill([]float64{42, 26}); err != nil {
		fmt.Println("holdout values:", err)
		return
	}
	scaledForecast, err := scaled.Model.Predict(context.Background(), holdout)
	if err != nil {
		fmt.Println("scaled forecast:", err)
		return
	}
	rawForecast, err := raw.Model.Predict(context.Background(), holdout)
	if err != nil {
		fmt.Println("raw forecast:", err)
		return
	}
	fmt.Printf("sensor gain=%.3f V/bar drift=%.3f V/C offset@25C=%.3f V\n",
		gainAndDrift[0], gainAndDrift[1], zeroAt25C)
	fmt.Printf("holdout=%.3f V scaled/raw agree within 1mV=%t\n",
		scaledForecast[0], math.Abs(scaledForecast[0]-rawForecast[0]) < .001)

	// Output:
	// sensor gain=0.084 V/bar drift=0.011 V/C offset@25C=0.625 V
	// holdout=4.164 V scaled/raw agree within 1mV=true
}

// ExampleFit_chargeEnergy estimates an origin-constrained charging-energy
// model with separate main-circuit and cooling-fan operating durations.
//
// Scenario:
//   - The engineering identity is zero delivered energy at zero operating
//     time for *both* channels; a learned intercept would be a modeling defect.
//   - The fan and charger have different effective powers. Scale-only QR keeps
//     those channel units comparable without hiding an intercept in centering.
//
// Process:
//   - Fit 16 readings in hours and kWh with NoIntercept+Standardize.
//   - Recover both powers in kW, then infer an off-grid duty-cycle scenario.
//   - Verify the exactly zero-origin output with a separate Predict call.
//
// Complexity: O(n*p*p+p^3) time with p=2 and O(n*p) temporary storage.
// AI-Hints: Do not apply a centered stats.Standardizer to this origin-constrained basis.
func ExampleFit_chargeEnergy() {
	chargerHours := []float64{
		.25, .50, .75, 1, 1.25, 1.5, 1.75, 2,
		2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4,
	}
	coolingHours := []float64{
		.1, .4, .3, .2, .6, .2, .7, .5,
		.8, .3, .9, .4, .7, 1, .6, 1.1,
	}
	measurementNoise := []float64{
		0, -.015, .010, -.020, .005, 0, .015, -.010,
		.008, -.006, .005, 0, -.005, .010, -.010, .005,
	}
	training, err := matrix.NewDense(len(chargerHours), 2)
	if err != nil {
		fmt.Println("matrix:", err)
		return
	}
	observedKWh := make([]float64, len(chargerHours))
	for i := range observedKWh {
		observedKWh[i] = 7.2*chargerHours[i] + .25*coolingHours[i] + measurementNoise[i]
		if err = training.Set(i, 0, chargerHours[i]); err != nil {
			fmt.Println("charger hours:", err)
			return
		}
		if err = training.Set(i, 1, coolingHours[i]); err != nil {
			fmt.Println("cooling hours:", err)
			return
		}
	}

	fit, err := ols.FitThroughOriginStandardized(context.Background(), training, observedKWh)
	if err != nil {
		fmt.Println("charging fit:", err)
		return
	}
	powerKW, intercept := fit.Model.Coefficients()
	future, err := matrix.NewDense(2, 2)
	if err != nil {
		fmt.Println("inference matrix:", err)
		return
	}
	if err = future.Fill([]float64{0, 0, 3.25, .8}); err != nil {
		fmt.Println("inference values:", err)
		return
	}
	predicted, err := fit.Model.Predict(context.Background(), future)
	if err != nil {
		fmt.Println("energy prediction:", err)
		return
	}
	fmt.Printf("charging main=%.2f kW cooling=%.2f kW rank=%d\n",
		powerKW[0], powerKW[1], fit.Rank)
	fmt.Printf("origin intercept=+0: %t, zero-case=+0: %t, next energy=%.2f kWh\n",
		intercept == 0 && !math.Signbit(intercept),
		predicted[0] == 0 && !math.Signbit(predicted[0]), predicted[1])

	// Output:
	// charging main=7.20 kW cooling=0.27 kW rank=2
	// origin intercept=+0: true, zero-case=+0: true, next energy=23.60 kWh
}

// ExampleFit_redundantTelemetry demonstrates an operational rank gate for
// duplicate telemetry channels before a latency regression is published.
//
// Scenario:
//   - A pipeline accidentally submits the same packet-rate channel twice,
//     under different names, next to a measured CPU temperature.
//   - A dependent design must fail; reducing the feature set is an explicit
//     *caller decision*, not an automatic feature drop inside OLS.
//
// Process:
//   - Show ErrRankDeficient on a 18-by-3 design with duplicate columns.
//   - Build a new 18-by-2 design that deliberately removes one duplicate.
//   - Refit and predict a new operating state only after the rank gate passes.
//
// Complexity: O(n*q*q+q^3) per fit; q=4 then q=3.
// AI-Hints: Never lower rank tolerance just to force collinear columns through QR.
func ExampleFit_redundantTelemetry() {
	packetRate := []float64{
		80, 120, 90, 160, 110, 200, 140, 240, 130,
		190, 170, 260, 150, 220, 280, 180, 300, 210,
	}
	temperatureC := []float64{
		24, 25, 30, 26, 31, 27, 33, 29, 36,
		28, 35, 32, 37, 30, 34, 38, 31, 40,
	}
	measurementNoise := []float64{
		.02, -.04, .03, -.02, .01, .05, -.03, .02, -.01,
		0, .01, -.04, .03, .02, -.02, .01, .04, -.03,
	}
	duplicated, err := matrix.NewDense(len(packetRate), 3)
	if err != nil {
		fmt.Println("matrix:", err)
		return
	}
	response := make([]float64, len(packetRate))
	for i := range response {
		response[i] = 1.5 + .02*packetRate[i] + .03*temperatureC[i] + measurementNoise[i]
		for col, value := range []float64{packetRate[i], packetRate[i], temperatureC[i]} {
			if err = duplicated.Set(i, col, value); err != nil {
				fmt.Println("duplicate data:", err)
				return
			}
		}
	}
	invalidFit, err := ols.FitStandardized(context.Background(), duplicated, response)
	fmt.Printf("duplicate gate: rank-deficient=%t published=%t\n",
		errors.Is(err, ols.ErrRankDeficient), invalidFit != nil)
	if !errors.Is(err, ols.ErrRankDeficient) {
		return // Do not invent an alternate fit on unrelated failures.
	}

	// The operator removes the known duplicate from the caller-owned feature
	// schema. OLS is invoked a second time with a separately valid design.
	selected, err := matrix.NewDense(len(packetRate), 2)
	if err != nil {
		fmt.Println("selected matrix:", err)
		return
	}
	for i := range response {
		if err = selected.Set(i, 0, packetRate[i]); err != nil {
			fmt.Println("selected packets:", err)
			return
		}
		if err = selected.Set(i, 1, temperatureC[i]); err != nil {
			fmt.Println("selected temperature:", err)
			return
		}
	}
	validFit, err := ols.FitStandardized(context.Background(), selected, response)
	if err != nil {
		fmt.Println("reduced fit:", err)
		return
	}
	next, err := matrix.NewDense(1, 2)
	if err != nil {
		fmt.Println("holdout matrix:", err)
		return
	}
	if err = next.Fill([]float64{250, 34}); err != nil {
		fmt.Println("holdout values:", err)
		return
	}
	predicted, err := validFit.Model.Predict(context.Background(), next)
	if err != nil {
		fmt.Println("latency prediction:", err)
		return
	}
	fmt.Printf("selected features=2 rank=%d predicted-latency=%.2f ms\n",
		validFit.Rank, predicted[0])

	// Output:
	// duplicate gate: rank-deficient=true published=false
	// selected features=2 rank=3 predicted-latency=7.52 ms
}

// ExampleModel_Predict validates a chronological holdout on a three-factor
// industrial energy-load model without fitting on future observations.
//
// Scenario:
//   - A facilities operator models daily kWh using heating-degree-hours,
//     production shift count, and machine runtime hours.
//   - Training is the first 18 days; the final six days form a strictly
//     later, untouched holdout batch with larger degree-hour extremes.
//   - A fixed operational 3-kWh MAE target is an illustrative business gate,
//     not a statistical confidence claim or extrapolation guarantee.
//
// Process:
//   - Fit once with train-only population standardization (q=4).
//   - Predict the six heldout rows and measure absolute prediction errors.
//   - Mutate an exported coefficient copy; confirm inference is unchanged.
//
// Complexity: O(n*q*q+q^3) training; O(h*p) inference, h=6, p=3.
// AI-Hints: A holdout error is not a proof of future performance or causal impact.
func ExampleModel_Predict() {
	degreeHours := []float64{
		20, 35, 28, 52, 41, 60, 47, 74, 55, 81, 63, 95,
		39, 68, 88, 32, 105, 57, 114, 70, 125, 84, 137, 99,
	}
	shifts := []float64{
		1, 2, 3, 2, 1, 3, 2, 1, 2, 3, 1, 2,
		3, 2, 1, 3, 2, 1, 3, 1, 2, 3, 1, 2,
	}
	runtimeHours := []float64{
		6, 4, 8, 3, 7, 5, 9, 2, 6, 4, 8, 3,
		7, 5, 9, 2, 6, 4, 8, 3, 7, 5, 9, 2,
	}
	meterNoise := []float64{
		2, -3, 1, -1, 0, 2, -2, 3, 1, -2, 0, 2,
		1, -1, 3, -3, 2, 0, -1, 2, -2, 1, 3, -1,
	}
	const trainDays = 18
	const holdoutDays = 6
	training, err := matrix.NewDense(trainDays, 3)
	if err != nil {
		fmt.Println("train matrix:", err)
		return
	}
	observedKWh := make([]float64, len(degreeHours))
	for day := range observedKWh {
		observedKWh[day] = 150 + 1.65*degreeHours[day] + 8.5*shifts[day] +
			.75*runtimeHours[day] + meterNoise[day]
		if day >= trainDays {
			continue // Holdout features are intentionally absent from the Fit call.
		}
		for col, value := range []float64{degreeHours[day], shifts[day], runtimeHours[day]} {
			if err = training.Set(day, col, value); err != nil {
				fmt.Println("train feature:", err)
				return
			}
		}
	}

	fit, err := ols.FitStandardized(context.Background(), training, observedKWh[:trainDays])
	if err != nil {
		fmt.Println("energy fit:", err)
		return
	}
	holdout, err := matrix.NewDense(holdoutDays, 3)
	if err != nil {
		fmt.Println("holdout matrix:", err)
		return
	}
	for day := 0; day < holdoutDays; day++ {
		index := trainDays + day
		for col, value := range []float64{degreeHours[index], shifts[index], runtimeHours[index]} {
			if err = holdout.Set(day, col, value); err != nil {
				fmt.Println("holdout feature:", err)
				return
			}
		}
	}
	forecast, err := fit.Model.Predict(context.Background(), holdout)
	if err != nil {
		fmt.Println("holdout prediction:", err)
		return
	}
	absoluteError := 0.0
	for day, estimate := range forecast {
		absoluteError += math.Abs(observedKWh[trainDays+day] - estimate)
	}
	maeKWh := absoluteError / float64(holdoutDays)

	// Coefficients returns a detached slice. Changing it must not tamper with
	// later forecasts or the model's frozen training parameters.
	exportedBeta, _ := fit.Model.Coefficients()
	exportedBeta[0] += 10000
	repeated, err := fit.Model.Predict(context.Background(), holdout)
	if err != nil {
		fmt.Println("repeat prediction:", err)
		return
	}
	stable := true
	for day := range forecast {
		stable = stable && forecast[day] == repeated[day]
	}
	fmt.Printf("energy train=%d holdout=%d rank=%d holdout-MAE=%.2f kWh\n",
		trainDays, holdoutDays, fit.Rank, maeKWh)
	fmt.Printf("operational MAE < 3 kWh=%t; frozen predictions=%t\n", maeKWh < 3, stable)

	// Output:
	// energy train=18 holdout=6 rank=4 holdout-MAE=1.97 kWh
	// operational MAE < 3 kWh=true; frozen predictions=true
}
