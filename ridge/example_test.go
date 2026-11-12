// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test demonstrates reproducible, decision-oriented Tikhonov
// regression pipelines. All observations are synthetic and have explicit units.
package ridge_test

import (
	"context"
	"fmt"
	"math"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ridge"
)

// exampleDense creates an owning matrix from a rectangular observation ledger.
// It refuses missing or inconsistent rows before calling any regression API.
//
// Implementation:
//   - Stage 1: validate the sample and feature counts.
//   - Stage 2: allocate one finite-only Dense and populate it in row-major order.
//   - Stage 3: return the complete matrix or the first construction error.
//
// Behavior highlights: never drops an invalid row or substitutes a zero feature.
// Inputs: rows are caller-owned read-only samples in their physical feature units.
// Returns: detached *matrix.Dense, or nil and a classified error.
// Errors: ridge.ErrEmptyInput, ridge.ErrShape, or underlying matrix errors.
// Determinism: outer row index, then inner feature index, both ascending.
// Complexity: Time O(n*p), Space O(n*p) for the detached matrix.
// Notes: Only tests and examples use this adapter; no production helper is exposed.
// AI-Hints: Check Set errors even for deterministic test data and use stable column order.
func exampleDense(rows [][]float64) (*matrix.Dense, error) {
	if len(rows) == 0 || len(rows[0]) == 0 {
		return nil, ridge.ErrEmptyInput
	}
	m, err := matrix.NewDense(len(rows), len(rows[0]))
	if err != nil {
		return nil, err
	}
	for i, row := range rows {
		if len(row) != m.Cols() {
			return nil, ridge.ErrShape
		}
		for j, value := range row {
			if err = m.Set(i, j, value); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}

// ExampleFitStandardized_financialHedge estimates a five-factor synthetic
// monthly portfolio return, deliberately including a duplicated macro index.
// A chronological 12-month holdout and two stress regimes check whether the
// fitted factor map is useful beyond minimizing the training objective.
func ExampleFitStandardized_financialHedge() {
	const (
		months      = 48
		trainMonths = 36
		lambda      = 0.12
	)
	ledger := make([][]float64, months)
	returns := make([]float64, months) // Monthly total returns, percentage points.
	for month := range ledger {
		growth := -1.5 + 0.1*float64((month*7)%31)
		rate := 2.4 + 0.12*float64((month*11)%17)
		energy := 70 + 2*float64((month*5)%19)
		credit := 1.1 + 0.09*float64((month*13)%15)
		ledger[month] = []float64{growth, growth, rate, energy, credit}
		// A deterministic seasonal disturbance keeps the holdout nontrivial.
		returns[month] = 0.4 + 0.35*growth - 0.2*rate +
			0.012*energy - 0.15*credit + 0.08*math.Sin(float64(month)*math.Pi/6)
	}
	training, err := exampleDense(ledger[:trainMonths])
	if err != nil {
		fmt.Println("training matrix failed")
		return
	}
	holdout, err := exampleDense(ledger[trainMonths:])
	if err != nil {
		fmt.Println("holdout matrix failed")
		return
	}
	fit, err := ridge.FitStandardized(context.Background(), training, returns[:trainMonths], lambda)
	if err != nil {
		fmt.Println("ridge fit failed")
		return
	}
	predicted, err := fit.Model.Predict(context.Background(), holdout)
	if err != nil {
		fmt.Println("holdout scoring failed")
		return
	}

	// Compare against a truly available training-only baseline, never the
	// holdout mean (which would leak future response information into the score).
	trainMean := 0.0
	for _, value := range returns[:trainMonths] {
		trainMean += value / float64(trainMonths)
	}
	ridgeMAE, baselineMAE := 0.0, 0.0
	for i, forecast := range predicted {
		actual := returns[trainMonths+i]
		ridgeMAE += math.Abs(forecast-actual) / float64(len(predicted))
		baselineMAE += math.Abs(trainMean-actual) / float64(len(predicted))
	}
	stress, err := exampleDense([][]float64{
		{-2.2, -2.2, 5.2, 105, 3.0}, // Rate and credit-spread shock.
		{1.5, 1.5, 2.1, 72, 1.1},    // Recovery with lower discount rates.
	})
	if err != nil {
		fmt.Println("stress matrix failed")
		return
	}
	stressResponse, err := fit.Model.Predict(context.Background(), stress)
	if err != nil {
		fmt.Println("stress prediction failed")
		return
	}
	beta, _ := fit.Model.Coefficients()
	fmt.Printf("portfolio: train=%d holdout=%d duplicate-slopes=%t\n",
		trainMonths, len(predicted), math.Abs(beta[0]-beta[1]) < 1e-9)
	fmt.Printf("holdout MAE: ridge=%.3f baseline=%.3f percentage-points\n", ridgeMAE, baselineMAE)
	fmt.Printf("stress: shock=%.2f recovery=%.2f percentage-points\n", stressResponse[0], stressResponse[1])
	// Output:
	// portfolio: train=36 holdout=12 duplicate-slopes=true
	// holdout MAE: ridge=0.048 baseline=0.232 percentage-points
	// stress: shock=-0.56 recovery=1.18 percentage-points
}

// ExampleFitWithLambda_spectroscopy solves 64 overlapping optical channel
// sensitivities from only 10 gas calibration mixtures. It scores heldout spectra
// against a concentration-mean baseline instead of claiming that p>n is invalid.
func ExampleFitWithLambda_spectroscopy() {
	const (
		channels     = 64
		calibrations = 10
		lambda       = 0.02
	)
	calibration := make([][]float64, calibrations)
	known := make([]float64, calibrations) // Volume-percent concentration.
	for i := range calibration {
		known[i] = 0.15 + 0.09*float64(i)
		calibration[i] = make([]float64, channels)
		for j := range calibration[i] {
			wavelength := float64(j) / channels
			band := math.Exp(-80 * (wavelength - 0.56) * (wavelength - 0.56))
			calibration[i][j] = 0.002 + 0.008*float64(j%3) + known[i]*band
		}
	}
	x, err := exampleDense(calibration)
	if err != nil {
		fmt.Println("calibration matrix failed")
		return
	}
	fit, err := ridge.FitWithLambda(context.Background(), x, known, lambda)
	if err != nil {
		fmt.Println("ridge calibration failed")
		return
	}

	actual := []float64{0.21, 0.47, 0.78, 0.92}
	measurement := make([][]float64, len(actual))
	for i, gas := range actual {
		measurement[i] = make([]float64, channels)
		for j := range measurement[i] {
			wavelength := float64(j) / channels
			band := math.Exp(-80 * (wavelength - 0.56) * (wavelength - 0.56))
			measurement[i][j] = 0.002 + 0.008*float64(j%3) + gas*band
		}
	}
	unseen, err := exampleDense(measurement)
	if err != nil {
		fmt.Println("measurement matrix failed")
		return
	}
	predictions, err := fit.Model.Predict(context.Background(), unseen)
	if err != nil {
		fmt.Println("measurement prediction failed")
		return
	}
	baseline := 0.0
	for _, gas := range known {
		baseline += gas / float64(len(known))
	}
	modelMAE, baselineMAE := 0.0, 0.0
	ordered := true
	for i, concentration := range predictions {
		modelMAE += math.Abs(concentration-actual[i]) / float64(len(actual))
		baselineMAE += math.Abs(baseline-actual[i]) / float64(len(actual))
		if i > 0 && concentration <= predictions[i-1] {
			ordered = false
		}
	}
	fmt.Printf("optical array: calibrations=%d channels=%d validation=%d\n", len(known), channels, len(actual))
	fmt.Printf("concentration MAE: ridge=%.3f baseline=%.3f vol-%%\n", modelMAE, baselineMAE)
	fmt.Printf("ascending unknown samples retained=%t\n", ordered)
	// Output:
	// optical array: calibrations=10 channels=64 validation=4
	// concentration MAE: ridge=0.008 baseline=0.255 vol-%
	// ascending unknown samples retained=true
}

// ExampleFitStandardized_seasonalAutoregression fits an AR(12) forecaster for
// monthly energy demand. All validation predictions use only observed previous
// months; recursive forecasts and future response leakage are deliberately absent.
func ExampleFitStandardized_seasonalAutoregression() {
	const (
		lags            = 12
		observations    = 108
		trainingWindows = 72
		lambda          = 0.25
	)
	usage := make([]float64, observations) // Synthetic monthly demand in MWh.
	for i := range usage {
		season := 12 * math.Sin(2*math.Pi*float64(i)/12)
		weeklyEffect := 1.4 * math.Sin(2*math.Pi*float64(i)/5)
		usage[i] = 100 + 0.06*float64(i) + season + weeklyEffect
	}
	windows := make([][]float64, observations-lags)
	targets := make([]float64, len(windows))
	for i := range windows {
		windows[i] = make([]float64, lags)
		for lag := 0; lag < lags; lag++ {
			windows[i][lag] = usage[i+lags-lag-1]
		}
		targets[i] = usage[i+lags]
	}
	train, err := exampleDense(windows[:trainingWindows])
	if err != nil {
		fmt.Println("training windows failed")
		return
	}
	validation, err := exampleDense(windows[trainingWindows:])
	if err != nil {
		fmt.Println("validation windows failed")
		return
	}
	fit, err := ridge.FitStandardized(context.Background(), train, targets[:trainingWindows], lambda)
	if err != nil {
		fmt.Println("autoregression failed")
		return
	}
	forecast, err := fit.Model.Predict(context.Background(), validation)
	if err != nil {
		fmt.Println("forecast failed")
		return
	}

	// The seasonal-naive baseline forecasts the observed value exactly 12
	// months earlier; neither method knows the current target during prediction.
	ridgeMAE, seasonalMAE := 0.0, 0.0
	for i, prediction := range forecast {
		actual := targets[trainingWindows+i]
		seasonalNaive := usage[trainingWindows+i]
		ridgeMAE += math.Abs(prediction-actual) / float64(len(forecast))
		seasonalMAE += math.Abs(seasonalNaive-actual) / float64(len(forecast))
	}
	fmt.Printf("AR(12): train=%d heldout=%d observed-one-step-ahead=true\n", trainingWindows, len(forecast))
	fmt.Printf("demand MAE: ridge=%.2f seasonal-naive=%.2f MWh\n", ridgeMAE, seasonalMAE)
	// Output:
	// AR(12): train=72 heldout=24 observed-one-step-ahead=true
	// demand MAE: ridge=1.58 seasonal-naive=1.89 MWh
}

// ExampleFitThroughOriginStandardized_sensorDose calibrates a physical monitor
// whose zero detector signals must yield a zero dose estimate. It scores an
// operational alarm candidate without treating regression as safety certification.
func ExampleFitThroughOriginStandardized_sensorDose() {
	const (
		calibrations   = 18
		lambda         = 0.04
		alarmThreshold = 1.5 // Synthetic radiation response in mSv/hour.
	)
	rows := make([][]float64, calibrations)
	dose := make([]float64, calibrations)
	for i := range rows {
		neutron := 4 + float64((i*5)%17)
		gamma := 10 + float64((i*7)%13)
		background := 2 + float64((i*3)%11)
		rows[i] = []float64{neutron, gamma, background}
		dose[i] = 0.07*neutron + 0.03*gamma + 0.01*background
	}
	training, err := exampleDense(rows)
	if err != nil {
		fmt.Println("detector calibration failed")
		return
	}
	fit, err := ridge.FitThroughOriginStandardized(context.Background(), training, dose, lambda)
	if err != nil {
		fmt.Println("dose fit failed")
		return
	}
	measurements, err := exampleDense([][]float64{{0, 0, 0}, {22, 15, 4}})
	if err != nil {
		fmt.Println("sensor batch failed")
		return
	}
	estimated, err := fit.Model.Predict(context.Background(), measurements)
	if err != nil {
		fmt.Println("dose inference failed")
		return
	}
	_, intercept := fit.Model.Coefficients()
	fmt.Printf("dosimeter: zero-intercept=%t zero-input-zero=%t\n", intercept == 0, estimated[0] == 0)
	fmt.Printf("field estimate=%.3f mSv/h above synthetic threshold=%t\n", estimated[1], estimated[1] >= alarmThreshold)
	// Output:
	// dosimeter: zero-intercept=true zero-input-zero=true
	// field estimate=2.000 mSv/h above synthetic threshold=true
}

// ExampleFit_singleCalibration is a calibration sufficiency warning: one
// concentration standard and 32 optical channels cannot identify channel
// sensitivities when an intercept is free. Ridge correctly shrinks every
// slope to zero and retains only the identifiable observed baseline.
func ExampleFit_singleCalibration() {
	const channels = 32
	standard := make([]float64, channels)
	for j := range standard {
		standard[j] = float64(100 + j*7)
	}
	training, err := exampleDense([][]float64{standard})
	if err != nil {
		fmt.Println("single standard failed")
		return
	}
	fit, err := ridge.Fit(context.Background(), training, []float64{17.5}, ridge.Options{
		Lambda:      1,
		Standardize: true,
	})
	if err != nil {
		fmt.Println("single-standard fit failed")
		return
	}
	future := make([]float64, channels)
	for j, value := range standard {
		future[j] = value * 1.2
	}
	unseen, err := exampleDense([][]float64{standard, future})
	if err != nil {
		fmt.Println("extra standards failed")
		return
	}
	prediction, err := fit.Model.Predict(context.Background(), unseen)
	if err != nil {
		fmt.Println("prediction failed")
		return
	}
	beta, intercept := fit.Model.Coefficients()
	largest := 0.0
	for _, slope := range beta {
		largest = math.Max(largest, math.Abs(slope))
	}
	fmt.Printf("single standard: samples=1 channels=%d baseline=%.2f\n", len(beta), intercept)
	fmt.Printf("channel sensitivity identifiable=%t heldout prediction invariant=%t\n", largest > 1e-9, prediction[0] == prediction[1])
	// Output:
	// single standard: samples=1 channels=32 baseline=17.50
	// channel sensitivity identifiable=false heldout prediction invariant=true
}

// ExampleFitThroughOrigin_extremeRegularization demonstrates a nonzero slope
// below 1e-90 without forming n*Lambda. The known scalar optimum is 5/(5+2λ),
// and the intercept remains exactly zero in physical units.
func ExampleFitThroughOrigin_extremeRegularization() {
	training, err := exampleDense([][]float64{{1}, {2}})
	if err != nil {
		fmt.Println("training failed")
		return
	}
	fit, err := ridge.FitThroughOrigin(context.Background(), training, []float64{1, 2}, 1e100)
	if err != nil {
		fmt.Println("extreme ridge failed")
		return
	}
	beta, intercept := fit.Model.Coefficients()
	fmt.Printf("extreme shrinkage: slope=%.3e intercept=%g\n", beta[0], intercept)
	fmt.Printf("penalty=%.3e SSE=%.1f\n", fit.Penalty, fit.ResidualSumSquares)
	// Output:
	// extreme shrinkage: slope=2.500e-100 intercept=0
	// penalty=3.125e-100 SSE=5.0
}
