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

// exampleDense creates an independent finite-only matrix from a sample ledger.
// It preserves feature order and refuses ragged records rather than padding
// missing measurements with zeros that would change the learned model.
//
// Implementation:
//   - Stage 1: require at least one observation with at least one feature.
//   - Stage 2: allocate a row-major owning Dense of the declared dimensions.
//   - Stage 3: check every subsequent row width and write features in order.
//
// Behavior highlights:
//   - A malformed record aborts construction; no partially filled matrix escapes.
//
// Inputs:
//   - rows: read-only n-by-p observation ledger; n,p>=1 and every row has width p.
//
// Returns:
//   - *matrix.Dense: detached owning feature matrix on success; otherwise nil.
//   - error: invalid fixture classification or the original matrix constructor/Set error.
//
// Errors:
//   - ridge.ErrEmptyInput for absent rows/features; ridge.ErrShape for ragged rows.
//   - Matrix constructor or Set errors are returned unchanged.
//
// Determinism:
//   - Rows ascend, then feature columns ascend; input slices are never mutated.
//
// Complexity:
//   - Time O(n*p), Space O(n*p) for the independent Dense allocation.
//
// Notes:
//   - This helper belongs solely to executable examples, not production ridge.
//   - It intentionally does not fit scales or derive target values.
//
// AI-Hints:
//   - Preserve every feature's physical unit; trained RIDGE owns scaling decisions.
//   - Check both matrix allocation and writes even for fixed demonstration data.
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

// ExampleFitStandardized_financialHedge evaluates a correlated macro-factor
// model before using its forecasts in a portfolio downside-risk review.
// A duplicate growth index reflects a common data-vendor overlap, while a
// chronological holdout prevents the fitted model from seeing future returns.
//
// Implementation:
//   - Stage 1: create 48 labeled monthly factor observations and reserve 12 months.
//   - Stage 2: fit standardized ridge on the first 36 months and score heldout MAE.
//   - Stage 3: compare a train-only baseline and score shock/recovery scenarios.
//
// Behavior highlights:
//   - Positive Lambda identifies a stable solution despite duplicated columns.
//   - Stress factors are expressed in original units and use frozen coefficients.
//   - A scenario forecast is a risk-screening signal, not a hedge transaction.
//
// Inputs:
//   - Synthetic monthly growth, rate, energy, and credit factors; returns in percentage points.
//   - Lambda=0.12, fixed before evaluating the chronological holdout.
//
// Returns:
//   - Reproducible holdout errors, duplicate-slope symmetry, and stress-review flags.
//
// Errors:
//   - Construction, fit, or prediction errors abort the scenario without partial results.
//
// Determinism:
//   - Fixed month order, fixed train/holdout boundary, and fixed stress rows.
//
// Complexity:
//   - Time O((n+p)*(p+1)^2), Space O((n+p)*(p+1)) for fitting; O(h*p) inference.
//
// Notes:
//   - This is reproducible synthetic data, not historical market observations.
//   - A lower MAE does not prove portfolio efficacy, causal effects, or hedge suitability.
//
// AI-Hints:
//   - Never fit scaling or choose Lambda from the heldout months in this example.
//   - Compare against a training-only baseline before interpreting a stress signal.
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
	// A review signal requires both a useful holdout fit and a downside shock.
	// It does not recommend a hedge size or initiate an execution.
	const materialShockPP = -0.50
	fmt.Printf("downside review: model-beats-baseline=%t material-shock=%t\n",
		ridgeMAE < baselineMAE, stressResponse[0] < materialShockPP)
	// Output:
	// portfolio: train=36 holdout=12 duplicate-slopes=true
	// holdout MAE: ridge=0.048 baseline=0.232 percentage-points
	// stress: shock=-0.56 recovery=1.18 percentage-points
	// downside review: model-beats-baseline=true material-shock=true
}

// ExampleFitWithLambda_spectroscopy calibrates a 64-channel optical gas array
// from ten reference mixtures and evaluates concentration estimates on heldout
// mixtures, a practical p>n problem with strongly overlapping sensor responses.
//
// Implementation:
//   - Stage 1: construct ten known-concentration spectra using a broad absorption band.
//   - Stage 2: fit affine ridge, retaining all 64 channels and their redundant information.
//   - Stage 3: score four withheld spectra against a calibration-only mean baseline.
//
// Behavior highlights:
//   - Augmented QR makes the wide calibration design solvable with positive Lambda.
//   - Ascending heldout concentrations should yield ascending predicted responses.
//   - Validation tolerances are engineering screening rules, not instrument certification.
//
// Inputs:
//   - 64 synthetic channels per sample; concentrations in volume percent.
//   - Lambda=0.02, chosen prior to scoring withheld calibration mixtures.
//
// Returns:
//   - Heldout MAE, ordering check, and a predeployment measurement-quality flag.
//
// Errors:
//   - Any construction, fitting, or inference failure stops this calibration run.
//
// Determinism:
//   - Channel order and all calibration concentrations are fixed and reproducible.
//
// Complexity:
//   - Time O((n+p)*(p+1)^2), Space O((n+p)*(p+1)) for augmented QR.
//
// Notes:
//   - Real spectroscopy needs blanks, reference drift checks, and independent standards.
//   - This example does not claim selective chemical identification or traceability.
//
// AI-Hints:
//   - Do not demand p<=n as with unregularized OLS; penalty rows restore rank.
//   - A low in-sample SSE alone never establishes a deployable optical calibration.
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
	// Reject a calibration that fails independent reference-mixture accuracy.
	// This illustrative tolerance is not a substitute for instrument standards.
	const maximumValidationMAE = 0.025 // Concentration in volume percent.
	fmt.Printf("predeployment check: reference-error-acceptable=%t\n",
		modelMAE < maximumValidationMAE && modelMAE < baselineMAE)
	// Output:
	// optical array: calibrations=10 channels=64 validation=4
	// concentration MAE: ridge=0.008 baseline=0.255 vol-%
	// ascending unknown samples retained=true
	// predeployment check: reference-error-acceptable=true
}

// ExampleFitStandardized_seasonalAutoregression trains a regularized AR(12)
// model for seasonal electricity demand and tests one-step-ahead forecasts on
// the final two years against a seasonal-naive operating baseline.
//
// Implementation:
//   - Stage 1: construct causal 12-lag windows from synthetic monthly meter readings.
//   - Stage 2: fit on the earliest 72 complete windows, freezing train-only scaling.
//   - Stage 3: forecast the next 24 observed-lag windows and compare MAE to last year.
//
// Behavior highlights:
//   - Overlapping lag predictors are legal under strictly positive Lambda.
//   - Validation uses prior observed months, never a future response to build a lag.
//   - The scenario is one-step-ahead; it is not a recursive 24-step forecast.
//
// Inputs:
//   - Synthetic monthly demand in MWh, twelve causally ordered lagged features.
//   - Lambda=0.25, selected before the chronological validation interval.
//
// Returns:
//   - Holdout MAE comparison and whether the model improves the scheduling baseline.
//
// Errors:
//   - Any matrix, Fit, or Predict error prevents publishing an evaluation.
//
// Determinism:
//   - Fixed demand curve, chronological split, and lag ordering.
//
// Complexity:
//   - Time O((n+p)*(p+1)^2), Space O((n+p)*(p+1)) to fit; O(h*p) scoring.
//
// Notes:
//   - Real deployments must validate the timestamp grid and segment missing months.
//   - Forecast-error thresholds must be calibrated to reserve and dispatch costs.
//
// AI-Hints:
//   - Never shuffle overlapping AR windows before splitting; that leaks time structure.
//   - Never interpret this observed-lag MAE as recursive multi-step performance.
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
	// Forecast skill is only established for observed-lag, one-step-ahead demand.
	fmt.Printf("scheduling review: improves-seasonal-baseline=%t\n", ridgeMAE < seasonalMAE)
	// Output:
	// AR(12): train=72 heldout=24 observed-one-step-ahead=true
	// demand MAE: ridge=1.58 seasonal-naive=1.89 MWh
	// scheduling review: improves-seasonal-baseline=true
}

// ExampleFitThroughOriginStandardized_sensorDose calibrates a three-channel
// radiation monitoring response where zero registered counts must map to zero
// estimated dose, then flags a synthetic high-response field observation.
//
// Implementation:
//   - Stage 1: construct eighteen cross-channel reference counts and target doses.
//   - Stage 2: fit strictly origin-constrained scale-only ridge calibration.
//   - Stage 3: check a zero-count control and compare a field estimate to a threshold.
//
// Behavior highlights:
//   - NoIntercept enforces an exact physical zero, without hidden centering.
//   - All calibration scaling is frozen before the two field measurements.
//   - The threshold is a synthetic review trigger, never a safety certification.
//
// Inputs:
//   - Neutron, gamma, and background channel counts; response in mSv/hour.
//   - Lambda=0.04 and an illustrative review threshold of 1.5 mSv/hour.
//
// Returns:
//   - Zero-origin invariants and a flagged field sample for human review.
//
// Errors:
//   - Fit and inference errors prevent an apparently successful field estimate.
//
// Determinism:
//   - Reference count patterns and field readings have fixed row/column order.
//
// Complexity:
//   - Time O((n+p)*p^2) for training; O(h*p) for the field batch.
//
// Notes:
//   - Real dosimetry requires metrological calibration and safety procedures.
//   - Positive Lambda biases slope estimates; monitor drift on independent standards.
//
// AI-Hints:
//   - Never center predictors under NoIntercept: doing so invents a physical offset.
//   - Model predictions must not replace an independent protection interlock.
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

// ExampleFit_singleCalibration illustrates a fundamental calibration limit:
// with one known response and 32 optical channels, an unpenalized intercept can
// explain the only sample, while all regularized slopes shrink to zero.
//
// Implementation:
//   - Stage 1: construct one 32-channel reference spectrum and its known response.
//   - Stage 2: fit positive-Lambda ridge with a free intercept and constant-safe scaling.
//   - Stage 3: compare predictions for unchanged and 20%-scaled new spectra.
//
// Behavior highlights:
//   - Full augmented rank does not imply enough data for empirical sensitivity.
//   - Constant training columns retain unit scales and incur positive slope penalties.
//   - Two distinct future spectra have identical intercept-only predictions.
//
// Inputs:
//   - One synthetic reference spectrum, observed concentration 17.5 units.
//   - Lambda=1 with training-only population standardization.
//
// Returns:
//   - An explicit non-identifiability warning encoded in reproducible output.
//
// Errors:
//   - Invalid spectra, construction failures, or fit errors stop the example.
//
// Determinism:
//   - The reference and shifted spectrum preserve original channel order.
//
// Complexity:
//   - Time O((1+p)*(p+1)^2), Space O((1+p)*(p+1)); p=32.
//
// Notes:
//   - Ridge gives a unique optimizer, not experimental information absent from data.
//   - More distinct reference standards are required to identify sensor sensitivities.
//
// AI-Hints:
//   - Never interpret a successful QR factorization as proof of calibration quality.
//   - Check response variability and independent references before deployment.
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

// ExampleFitThroughOrigin_extremeRegularization verifies a tiny but nonzero
// coefficient under extremely strong Tikhonov shrinkage, when naive binary64
// QR preparation can incorrectly erase the physically representable solution.
//
// Implementation:
//   - Stage 1: construct the scalar two-observation calibration and Lambda=1e100.
//   - Stage 2: solve via augmented, penalty-first, column-equilibrated QR.
//   - Stage 3: compare the coefficient and penalty to independent scalar formulas.
//
// Behavior highlights:
//   - The physical intercept remains positive zero under NoIntercept.
//   - The exact optimum is beta=5/(5+2*Lambda), not numerical zero.
//   - The penalty uses optimization slopes and still approaches zero as Lambda grows.
//
// Inputs:
//   - x=[1,2], y=[1,2], Lambda=1e100, no intercept.
//
// Returns:
//   - Stable coefficient, SSE, penalty, and analytic consistency flags.
//
// Errors:
//   - Any QR or final-diagnostic overflow suppresses the entire fit result.
//
// Determinism:
//   - One scalar oracle and a fixed augmented QR traversal order.
//
// Complexity:
//   - Time O((n+p)*p^2), Space O((n+p)*p); n=2, p=1.
//
// Notes:
//   - This fixture verifies numerical representability, not model usefulness.
//   - For even larger ratios, float64 rank/output limits may still require failure.
//
// AI-Hints:
//   - Never substitute zero for a finite analytic optimum because Lambda is large.
//   - Never form n*Lambda before taking a safe, factored square root.
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
	// The oracle is algebraic and independent of the fitted QR coefficients.
	const lambda = 1e100
	analyticSlope := 5 / (5 + 2*lambda)
	analyticPenalty := (lambda / 2) * analyticSlope * analyticSlope
	fmt.Printf("analytic oracle: slope=%t penalty=%t\n",
		math.Abs(beta[0]/analyticSlope-1) < 1e-8,
		math.Abs(fit.Penalty/analyticPenalty-1) < 1e-8)
	// Output:
	// extreme shrinkage: slope=2.500e-100 intercept=0
	// penalty=3.125e-100 SSE=5.0
	// analytic oracle: slope=true penalty=true
}

// ExampleFitStandardized_networkLatencyCapacity uses ridge as a predictive
// guardrail for service-latency capacity planning when CPU measurements from
// two telemetry agents are duplicated and the design is strongly collinear.
//
// Implementation:
//   - Stage 1: build time-ordered 15-minute request, CPU, cache, and queue records.
//   - Stage 2: fit on 30 windows and verify p99 latency against 12 withheld windows.
//   - Stage 3: compare a training-mean baseline, then evaluate two capacity plans.
//
// Behavior highlights:
//   - Duplicate CPU signals remain in the regression and receive symmetric slopes.
//   - The intervention changes projected CPU, cache misses, and queue depth while
//     holding the request rate constant: it tests a specific engineering hypothesis.
//   - Deployment review requires observed holdout improvement as well as SLO relief.
//
// Inputs:
//   - Five features: requests/s, two CPU percentage readings, cache misses (%), queue depth.
//   - The response is observed synthetic p99 latency in milliseconds; Lambda=0.08.
//
// Returns:
//   - A validation-gated before/after capacity-plan review, never a deployment command.
//
// Errors:
//   - Any matrix, Fit, or Predict error prevents a capacity decision.
//
// Determinism:
//   - Window times, train/holdout boundary, duplicate readings, and scenario order are fixed.
//
// Complexity:
//   - Time O((n+p)*(p+1)^2) for fitting, O(h*p) for scoring; O((n+p)*(p+1)) space.
//
// Notes:
//   - Measurements are synthetic; this is not a causal experiment or real incident record.
//   - A holdout gain does not establish a reliable SLO guarantee under distribution shift.
//
// AI-Hints:
//   - Do not fit feature scales on the 12 withheld windows or optimize Lambda using them.
//   - The counterfactual plan is a review candidate, not proof that cache tuning causes relief.
func ExampleFitStandardized_networkLatencyCapacity() {
	const (
		windows                       = 42
		trainingWindows               = 30
		lambda                        = 0.08
		latencyBudgetMillis           = 90.0
		duplicateCoefficientTolerance = 1e-9
	)

	// Each row is a 15-minute service telemetry window. The two CPU readings
	// deliberately agree because the agents expose the same underlying metric.
	telemetry := make([][]float64, windows)
	observedP99Millis := make([]float64, windows)
	for window := range telemetry {
		requestsPerSecond := 820 + float64((window*137)%480)
		cpuPercent := 45 + float64((window*17)%44)
		cacheMissPercent := 4 + float64((window*11)%21)
		queueDepth := 12 + float64((window*23)%65)
		telemetry[window] = []float64{
			requestsPerSecond, cpuPercent, cpuPercent, cacheMissPercent, queueDepth,
		}
		// A small repeatable perturbation avoids disguising a perfect linear law.
		observedP99Millis[window] = 18 + 0.013*requestsPerSecond +
			0.24*cpuPercent + 0.70*cacheMissPercent + 0.20*queueDepth +
			0.6*math.Sin(float64(window)*math.Pi/7)
	}

	train, err := exampleDense(telemetry[:trainingWindows])
	if err != nil {
		fmt.Println("telemetry training failed")
		return
	}
	holdout, err := exampleDense(telemetry[trainingWindows:])
	if err != nil {
		fmt.Println("telemetry holdout failed")
		return
	}
	fit, err := ridge.FitStandardized(context.Background(), train,
		observedP99Millis[:trainingWindows], lambda)
	if err != nil {
		fmt.Println("latency fit failed")
		return
	}
	forecast, err := fit.Model.Predict(context.Background(), holdout)
	if err != nil {
		fmt.Println("latency validation failed")
		return
	}

	// This baseline uses only the training responses. Validation observations
	// are read solely for error evaluation, never for choosing feature scales.
	trainingMean := 0.0
	for _, p99 := range observedP99Millis[:trainingWindows] {
		trainingMean += p99 / trainingWindows
	}
	modelMAE, baselineMAE := 0.0, 0.0
	for i, estimate := range forecast {
		actual := observedP99Millis[trainingWindows+i]
		modelMAE += math.Abs(estimate-actual) / float64(len(forecast))
		baselineMAE += math.Abs(trainingMean-actual) / float64(len(forecast))
	}

	// Model an intervention at the same arrival rate: fewer cache misses and
	// shallower queues after adding a cache shard and draining hot partitions.
	plans, err := exampleDense([][]float64{
		{1520, 92, 92, 31, 88}, // Before: overloaded hot partition.
		{1520, 67, 67, 11, 30}, // After: projected post-mitigation telemetry.
	})
	if err != nil {
		fmt.Println("capacity plan failed")
		return
	}
	projectedP99, err := fit.Model.Predict(context.Background(), plans)
	if err != nil {
		fmt.Println("capacity projection failed")
		return
	}
	beta, _ := fit.Model.Coefficients()
	validated := modelMAE < baselineMAE
	beforeBudget := projectedP99[0] >= latencyBudgetMillis
	afterBudget := projectedP99[1] >= latencyBudgetMillis

	fmt.Printf("latency calibration: train=%d heldout=%d features=%d cpu-weights-equal=%t\n",
		trainingWindows, len(forecast), len(beta), math.Abs(beta[1]-beta[2]) < duplicateCoefficientTolerance)
	fmt.Printf("holdout MAE: ridge=%.2fms training-mean=%.2fms\n", modelMAE, baselineMAE)
	fmt.Printf("capacity estimates: current=%.1fms mitigated=%.1fms SLO=%.0fms\n",
		projectedP99[0], projectedP99[1], latencyBudgetMillis)
	fmt.Printf("capacity review: original-exceeds-SLO=%t plan-exceeds-SLO=%t review-plan=%t\n",
		beforeBudget, afterBudget, validated && beforeBudget && !afterBudget)

	// Output:
	// latency calibration: train=30 heldout=12 features=5 cpu-weights-equal=true
	// holdout MAE: ridge=0.47ms training-mean=5.68ms
	// capacity estimates: current=97.2ms mitigated=67.5ms SLO=90ms
	// capacity review: original-exceeds-SLO=true plan-exceeds-SLO=false review-plan=true
}
