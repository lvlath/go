// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Executable examples demonstrate fitted PCA in financial risk, industrial
// condition monitoring, medical signal engineering, and energy operations.
// All synthetic fixtures have deterministic generation and chronological holdouts.
// A fitted projection is not a calibrated decision rule without an independent gate.
package pca_test

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/pca"
)

// exampleMatrix copies a complete rectangular observation ledger to owning Dense.
// Rows remain in their original order; no missing value is silently padded.
//
// Implementation:
//   - Stage 1: reject empty and ragged rectangular input.
//   - Stage 2: allocate a finite-only matrix with exact n-by-d shape.
//   - Stage 3: copy each value in row-major order through checked Set.
//
// Inputs: rows are read-only finite observations with equal feature widths.
// Returns: detached *matrix.Dense, or a descriptive sentinel-preserving error.
// Errors: pca.ErrEmptyInput, pca.ErrShape, or the original matrix error.
// Determinism: rows ascending, then features ascending.
// Complexity: Time O(n*d), space O(n*d).
// AI-Hints: this helper does not fit a model or normalize feature units.
func exampleMatrix(rows [][]float64) (*matrix.Dense, error) {
	if len(rows) == 0 || len(rows[0]) == 0 {
		return nil, pca.ErrEmptyInput
	}
	out, err := matrix.NewDense(len(rows), len(rows[0]))
	if err != nil {
		return nil, err
	}
	for i, row := range rows {
		if len(row) != len(rows[0]) {
			return nil, pca.ErrShape
		}
		for j, value := range row {
			if err = out.Set(i, j, value); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

// exampleSPE computes physical-unit squared prediction errors for equal-unit sensors.
// Each row is reconstructed with the frozen PCA model; thresholds are external.
//
// Implementation: project/reconstruct via Model.Reconstruct; sum squared deltas.
// Inputs: observations share one physical unit across all sensor channels.
// Returns: one nonnegative SPE per row, or nil on any read or reconstruction error.
// Determinism: row-major residual accumulation; no mutable global workspace.
// Complexity: O(n*d*k) time, O(n*d+n) memory including reconstructed matrix.
// AI-Hints: do not use physical-unit SPE across incomparable sensor units.
func exampleSPE(ctx context.Context, model *pca.Model, observations matrix.Matrix) ([]float64, error) {
	reconstructed, err := model.Reconstruct(ctx, observations)
	if err != nil {
		return nil, err
	}
	spe := make([]float64, observations.Rows())
	for row := range spe {
		for col := 0; col < observations.Cols(); col++ {
			measured, readErr := observations.At(row, col)
			if readErr != nil {
				return nil, readErr
			}
			estimated, readErr := reconstructed.At(row, col)
			if readErr != nil {
				return nil, readErr
			}
			delta := measured - estimated
			spe[row] = math.FMA(delta, delta, spe[row])
			if math.IsInf(spe[row], 0) || math.IsNaN(spe[row]) {
				return nil, pca.ErrOverflow
			}
		}
	}
	return spe, nil
}

// ExampleFit_yieldCurveFactorRisk compresses a synthetic sovereign zero-coupon
// rate surface into three data-derived factors and validates its heldout loss.
// A frozen training mean is the no-factor baseline; the result is not a trade signal.
//
// Implementation:
//   - Stage 1: generate 96 monthly curves for eight tenors from three changing
//     Nelson-Siegel-shaped drivers plus deterministic small measurement noise.
//   - Stage 2: fit three principal components on the first 72 months only.
//   - Stage 3: reconstruct the last 24 months and compare RMSE against the
//     historical training-mean curve in percentage-point yield units.
//
// Behavior highlights:
//   - PCA learns an orthogonal basis; economic level/slope/curvature labels require
//     separate inspection of the loading shapes and their temporal behavior.
//
// Inputs: generated monthly yields in percent for 0.5..30-year maturities.
// Returns: basis dimensions, holdout-versus-frozen-mean RMSE, retained variance,
// and the factor-score distance caused by a +45 bp parallel-rate stress.
// Errors: matrix, fit, or inference failures stop without publishing a conclusion.
// Determinism: fixed chronology, maturity order, and generation functions.
// Complexity: O(n*d*d+rotations*d*d+n*d*k); holdout scoring O(h*d*k).
// Notes: synthetic term structures are not observed market or trading data.
// AI-Hints: do not estimate a new mean on the validation months or claim yield forecasts.
func ExampleFit_yieldCurveFactorRisk() {
	ctx := context.Background()
	maturities := []float64{0.5, 1, 2, 3, 5, 10, 20, 30}
	const trainingMonths = 72
	const totalMonths = 96
	curves := make([][]float64, totalMonths)
	for month := 0; month < totalMonths; month++ {
		t := float64(month)
		level := 3.2 + 0.25*math.Sin(0.11*t) + 0.002*t
		slope := -0.65 + 0.21*math.Cos(0.17*t)
		curvature := 0.42 * math.Sin(0.29*t+0.3)
		curves[month] = make([]float64, len(maturities))
		for tenorIndex, years := range maturities {
			decay := years / 4.5
			loading := -math.Expm1(-decay) / decay
			curvatureLoading := loading - math.Exp(-decay)
			noise := 0.0009 * math.Sin(float64((month+3)*(tenorIndex+2))*0.47)
			curves[month][tenorIndex] = level + slope*loading + curvature*curvatureLoading + noise
		}
	}
	training, err := exampleMatrix(curves[:trainingMonths])
	if err != nil {
		fmt.Println("training matrix error")
		return
	}
	fit, err := pca.FitWithComponents(ctx, training, 3)
	if err != nil {
		fmt.Println("fit error")
		return
	}
	holdout, err := exampleMatrix(curves[trainingMonths:])
	if err != nil {
		fmt.Println("holdout matrix error")
		return
	}
	reconstructed, err := fit.Model.Reconstruct(ctx, holdout)
	if err != nil {
		fmt.Println("reconstruction error")
		return
	}
	trainMeans := make([]float64, training.Cols())
	for col := range trainMeans {
		for row := 0; row < training.Rows(); row++ {
			value, readErr := training.At(row, col)
			if readErr != nil {
				fmt.Println("training read error")
				return
			}
			trainMeans[col] += value / float64(training.Rows())
		}
	}
	var baselineSSE, modelSSE float64
	for row := 0; row < holdout.Rows(); row++ {
		for col := range trainMeans {
			actual, actualErr := holdout.At(row, col)
			predicted, predictedErr := reconstructed.At(row, col)
			if actualErr != nil || predictedErr != nil {
				fmt.Println("holdout read error")
				return
			}
			baselineSSE += math.Pow(actual-trainMeans[col], 2)
			modelSSE += math.Pow(actual-predicted, 2)
		}
	}
	retainedVariance := 0.0
	for _, ratio := range fit.ExplainedVarianceRatio {
		retainedVariance += ratio
	}
	fmt.Printf("yield curves: train=%d holdout=%d maturities=%d factors=%d\n",
		training.Rows(), holdout.Rows(), training.Cols(), len(fit.Eigenvalues))
	// Compare absolute reconstruction errors in percentage-point yield units;
	// the mean curve is frozen before the chronological validation interval.
	cells := float64(holdout.Rows() * holdout.Cols())
	modelRMSE := math.Sqrt(modelSSE / cells)
	baselineRMSE := math.Sqrt(baselineSSE / cells)

	// Stress the latest observed curve with an explicitly parallel +45 bp rate
	// shift. This is scenario analysis on a fitted basis, not a yield forecast.
	baseCurve := append([]float64(nil), curves[totalMonths-1]...)
	shockedCurve := append([]float64(nil), baseCurve...)
	const shockBasisPoints = 45.0
	for tenor := range shockedCurve {
		shockedCurve[tenor] += shockBasisPoints / 100 // Yields are in percentage points.
	}
	stressInput, err := exampleMatrix([][]float64{baseCurve, shockedCurve})
	if err != nil {
		fmt.Println("stress input error")
		return
	}
	stressScores, err := fit.Model.Transform(ctx, stressInput)
	if err != nil {
		fmt.Println("stress projection error")
		return
	}
	var factorDisplacement float64
	for component := range fit.Eigenvalues {
		before, beforeErr := stressScores.At(0, component)
		after, afterErr := stressScores.At(1, component)
		if beforeErr != nil || afterErr != nil {
			fmt.Println("stress score read error")
			return
		}
		factorDisplacement = math.Hypot(factorDisplacement, after-before)
	}
	fmt.Printf("holdout RMSE: PCA=%.5f baseline=%.5f pp; improves=%t\n",
		modelRMSE, baselineRMSE, modelRMSE < 0.02*baselineRMSE)
	fmt.Printf("retained training variance=%.6f; +45 bp stress score-distance=%.3f pp\n",
		retainedVariance, factorDisplacement)
	// Output:
	// yield curves: train=72 holdout=24 maturities=8 factors=3
	// holdout RMSE: PCA=0.00048 baseline=0.24225 pp; improves=true
	// retained training variance=0.999992; +45 bp stress score-distance=1.273 pp
}

// ExampleFit_turbineBearingSPE monitors six vibration sensors for bearing damage.
// It calibrates a nonparametric review threshold on normal validation data only,
// then checks a later anomaly without fitting, scaling, or tuning against the fault.
//
// Implementation:
//   - Stage 1: build 120 equal-unit vibration snapshots under normal operation.
//   - Stage 2: fit a two-factor normal manifold using the first 96 snapshots.
//   - Stage 3: calibrate an SPE threshold on the next 24 normal snapshots and
//     compare a withheld healthy observation with a simulated damaged bearing.
//
// Behavior highlights:
//   - All six features are vibration RMS values in mm/s, making raw SPE meaningful.
//   - Threshold is a screening review threshold, not a safety-certified alarm.
//
// Inputs: fixed synthetic operating vibration windows, with one bearing-channel shock.
// Returns: the calibrated physical-unit SPE, a maintenance review gate, and
// fault severity relative to the separately calibrated threshold.
// Errors: matrix and model errors abort without reporting a false all-clear.
// Determinism: chronological training/calibration split and fixed channel indexing.
// Complexity: O(n*d*d+rotations*d*d+n*d*k), with O(h*d*k) calibration.
// Notes: real plants need equipment-specific alarm thresholds and drift surveillance.
// AI-Hints: calibration rows must never be included in Fit; never calibrate on faults.
func ExampleFit_turbineBearingSPE() {
	ctx := context.Background()
	const trainingWindows = 96
	const totalNormalWindows = 120
	const vibrationChannels = 6
	normal := make([][]float64, totalNormalWindows)
	for row := range normal {
		load := math.Sin(float64(row) * 0.21)
		rotation := math.Cos(float64(row) * 0.17)
		normal[row] = make([]float64, vibrationChannels)
		for channel := 0; channel < vibrationChannels; channel++ {
			mix := float64(channel)
			baseline := 2.1 + 0.13*mix
			noise := 0.004 * math.Sin(float64((row+1)*(channel+2))*0.41)
			normal[row][channel] = baseline + (0.45+0.04*mix)*load + (0.23-0.025*mix)*rotation + noise
		}
	}
	training, err := exampleMatrix(normal[:trainingWindows])
	if err != nil {
		fmt.Println("training matrix error")
		return
	}
	fit, err := pca.FitWithComponents(ctx, training, 2)
	if err != nil {
		fmt.Println("fit error")
		return
	}
	calibration, err := exampleMatrix(normal[trainingWindows:])
	if err != nil {
		fmt.Println("calibration matrix error")
		return
	}
	calibrationSPE, err := exampleSPE(ctx, fit.Model, calibration)
	if err != nil {
		fmt.Println("calibration error")
		return
	}
	var maxNormalSPE float64
	for _, spe := range calibrationSPE {
		if spe > maxNormalSPE {
			maxNormalSPE = spe
		}
	}
	// A separate engineering policy applies a conservative margin to measured
	// calibration residuals. It is not an automatically learned package threshold.
	const reviewMargin = 4.0
	threshold := reviewMargin * maxNormalSPE
	newObservation := append([]float64(nil), normal[119]...)
	faultObservation := append([]float64(nil), normal[119]...)
	faultObservation[3] += 1.8 // Sudden bearing vibration increase (mm/s RMS).
	future, err := exampleMatrix([][]float64{newObservation, faultObservation})
	if err != nil {
		fmt.Println("future matrix error")
		return
	}
	futureSPE, err := exampleSPE(ctx, fit.Model, future)
	if err != nil {
		fmt.Println("future reconstruction error")
		return
	}
	fmt.Printf("turbine monitoring: train=%d calibration=%d channels=%d factors=%d\n",
		training.Rows(), calibration.Rows(), vibrationChannels, len(fit.Eigenvalues))
	// Express each outcome in calibration-threshold units. Operators can now
	// distinguish a marginal crossing from a severe deviation and route it
	// to independent vibration inspection rather than automatic shutdown.
	if threshold <= 0 {
		fmt.Println("invalid calibration threshold")
		return
	}
	// A compact order-of-magnitude display is useful operationally without
	// coupling executable examples to the final few floating-point ULPs.
	fmt.Printf("SPE: normal~%.1e threshold~%.1e fault~%.1f (mm/s)^2\n",
		futureSPE[0], threshold, futureSPE[1])
	fmt.Printf("maintenance review: normal=%t bearing=%t severity>1000x=%t\n",
		futureSPE[0] <= threshold, futureSPE[1] > threshold, futureSPE[1] > 1000*threshold)
	// Output:
	// turbine monitoring: train=96 calibration=24 channels=6 factors=2
	// SPE: normal~4.9e-05 threshold~2.6e-04 fault~2.7 (mm/s)^2
	// maintenance review: normal=true bearing=true severity>1000x=true
}

// ExampleFit_multichannelECGDenoising reconstructs 12 correlated leads in mV.
// Its independent clean synthetic reference quantifies noise removal on withheld
// samples; PCA is a cross-channel subspace filter, not a temporal low-pass filter.
//
// Implementation:
//   - Stage 1: generate 640 deterministic samples from three coherent waveforms
//     mixed into 12 leads, plus lead-dependent high-frequency instrumentation noise.
//   - Stage 2: fit three retained directions on the first 480 noisy samples.
//   - Stage 3: reconstruct the remaining 160 and compare errors against the
//     independently generated clean signals, not against the training input.
//
// Behavior highlights:
//   - Comparison against a clean oracle avoids calling small training residuals
//     evidence of denoising when the original measurements contain noise.
//
// Inputs: synthetic millivolt leads sampled at 200 Hz; independent clean reference.
// Returns: holdout reconstruction shape, original/restored clean-reference RMSE
// in millivolts, and a quantitative noise reduction check.
// Errors: fit or reconstruction failure aborts without a denoising claim.
// Determinism: fixed sampling grid, frequencies, mixing, and train/holdout split.
// Complexity: O(n*d*d+rotations*d*d+n*d*k), followed by O(h*d*k) inference.
// Notes: synthetic traces are not patient measurements or clinical diagnoses.
// AI-Hints: never evaluate denoising on training data alone or claim clinical efficacy.
func ExampleFit_multichannelECGDenoising() {
	ctx := context.Background()
	const trainingSamples = 480
	const totalSamples = 640
	const leads = 12
	noisy := make([][]float64, totalSamples)
	clean := make([][]float64, totalSamples)
	for sample := 0; sample < totalSamples; sample++ {
		t := float64(sample) / 200.0
		p := 0.19 * math.Sin(2*math.Pi*1.2*t)
		qrsPhase := math.Mod(t, 0.83) - 0.41
		qrs := 1.1 * math.Exp(-math.Pow(qrsPhase/0.035, 2))
		tWave := 0.31 * math.Sin(2*math.Pi*0.54*t+0.45)
		noisy[sample] = make([]float64, leads)
		clean[sample] = make([]float64, leads)
		for lead := 0; lead < leads; lead++ {
			mixP := 0.55 + float64(lead)*0.047
			mixQ := 0.82 - float64(lead)*0.051
			mixT := 0.28 + float64(lead%5)*0.1
			clean[sample][lead] = math.FMA(mixP, p, math.FMA(mixQ, qrs, mixT*tWave))
			interference := 0.032*math.Sin(2*math.Pi*float64(32+lead%4)*t+float64(lead)*0.47) +
				0.018*math.Cos(2*math.Pi*float64(43+lead%5)*t+float64(lead)*0.31)
			noisy[sample][lead] = clean[sample][lead] + interference
		}
	}
	training, err := exampleMatrix(noisy[:trainingSamples])
	if err != nil {
		fmt.Println("training matrix error")
		return
	}
	fit, err := pca.FitWithComponents(ctx, training, 3)
	if err != nil {
		fmt.Println("fit error")
		return
	}
	holdout, err := exampleMatrix(noisy[trainingSamples:])
	if err != nil {
		fmt.Println("holdout matrix error")
		return
	}
	reconstructed, err := fit.Model.Reconstruct(ctx, holdout)
	if err != nil {
		fmt.Println("reconstruction error")
		return
	}
	var noisySSE, restoredSSE float64
	for row := 0; row < holdout.Rows(); row++ {
		for lead := 0; lead < leads; lead++ {
			actual, readErr := holdout.At(row, lead)
			if readErr != nil {
				fmt.Println("holdout read error")
				return
			}
			restored, readErr := reconstructed.At(row, lead)
			if readErr != nil {
				fmt.Println("reconstruction read error")
				return
			}
			reference := clean[trainingSamples+row][lead]
			noisySSE = math.FMA(actual-reference, actual-reference, noisySSE)
			restoredSSE = math.FMA(restored-reference, restored-reference, restoredSSE)
		}
	}
	fmt.Printf("ECG leads: train=%d holdout=%d original=%d reconstructed=%d factors=%d\n",
		training.Rows(), holdout.Rows(), holdout.Cols(), reconstructed.Cols(), len(fit.Eigenvalues))
	// A synthetic clean waveform provides an independent evaluation oracle.
	// Reconstruction error against noisy input alone would not prove denoising.
	measurements := float64(holdout.Rows() * holdout.Cols())
	noisyRMSE := math.Sqrt(noisySSE / measurements)
	restoredRMSE := math.Sqrt(restoredSSE / measurements)
	fmt.Printf("clean-reference RMSE: noisy=%.4f PCA=%.4f mV; lower=%t\n",
		noisyRMSE, restoredRMSE, restoredRMSE < 0.85*noisyRMSE)
	// Output:
	// ECG leads: train=480 holdout=160 original=12 reconstructed=12 factors=3
	// clean-reference RMSE: noisy=0.0260 PCA=0.0145 mV; lower=true
}

// ExampleFit_substationLoadCompression compresses ten feeder-load trajectories.
// A chronological two-day holdout checks whether the low-dimensional basis can
// reconstruct operational telemetry better than a frozen feeder-mean baseline.
//
// Implementation:
//   - Stage 1: form 216 hourly feeder snapshots with common demand, industrial,
//     and weather-like factors, plus deterministic local meter noise.
//   - Stage 2: fit three components on the earliest 168 hours only.
//   - Stage 3: reconstruct the next 48 hours and compare squared MWh errors
//     against a constant per-feeder mean computed on training hours alone.
//
// Behavior highlights:
//   - The latent factors compress correlated loads; they are not forecasts of
//     future hourly demand or substitutes for chronological dispatch validation.
//
// Inputs: synthetic hourly MWh readings from ten substation feeders.
// Returns: retained feature dimensions, frozen-mean and PCA holdout RMSE in
// MWh, and a compression suitability review flag.
// Errors: malformed matrix, fit, or reconstruction aborts before publication.
// Determinism: fixed hour ordering and explicit 168/48 temporal separation.
// Complexity: O(n*d*d+rotations*d*d+n*d*k); holdout scoring O(h*d*k).
// Notes: no dispatch controls, load shedding, or automated grid actions are performed.
// AI-Hints: never shuffle time-split windows or claim PCA predicts unseen future rows.
func ExampleFit_substationLoadCompression() {
	ctx := context.Background()
	const trainingHours = 168
	const totalHours = 216
	const feeders = 10
	meterData := make([][]float64, totalHours)
	for hour := range meterData {
		t := float64(hour)
		dayPhase := float64(hour % 24)
		dailyLoad := 7.5 * math.Sin(2*math.Pi*(dayPhase-7)/24)
		industrial := 4.1 * math.Cos(2*math.Pi*t/168)
		weather := 3.2 * math.Sin(2*math.Pi*t/96)
		meterData[hour] = make([]float64, feeders)
		for feeder := 0; feeder < feeders; feeder++ {
			j := float64(feeder)
			noise := 0.055 * math.Sin(float64((hour+3)*(feeder+2))*0.23)
			meterData[hour][feeder] = 38 + 4.5*j + (1+0.08*j)*dailyLoad +
				(j/6-0.6)*industrial + float64(feeder%4-2)*weather + noise
		}
	}
	train, err := exampleMatrix(meterData[:trainingHours])
	if err != nil {
		fmt.Println("training matrix error")
		return
	}
	fit, err := pca.FitWithComponents(ctx, train, 3)
	if err != nil {
		fmt.Println("fit error")
		return
	}
	holdout, err := exampleMatrix(meterData[trainingHours:])
	if err != nil {
		fmt.Println("holdout matrix error")
		return
	}
	reconstructed, err := fit.Model.Reconstruct(ctx, holdout)
	if err != nil {
		fmt.Println("reconstruction error")
		return
	}
	trainMeans := make([]float64, feeders)
	for row := 0; row < train.Rows(); row++ {
		for feeder := range trainMeans {
			value, readErr := train.At(row, feeder)
			if readErr != nil {
				fmt.Println("training read error")
				return
			}
			trainMeans[feeder] += value / float64(train.Rows())
		}
	}
	var modelSSE, baselineSSE float64
	for row := 0; row < holdout.Rows(); row++ {
		for feeder := range trainMeans {
			actual, actualErr := holdout.At(row, feeder)
			restored, restoredErr := reconstructed.At(row, feeder)
			if actualErr != nil || restoredErr != nil {
				fmt.Println("holdout read error")
				return
			}
			baselineSSE += math.Pow(actual-trainMeans[feeder], 2)
			modelSSE += math.Pow(actual-restored, 2)
		}
	}
	fmt.Printf("grid loads: train=%d holdout=%d feeders=%d factors=%d\n",
		train.Rows(), holdout.Rows(), train.Cols(), len(fit.Eigenvalues))
	// These are *reconstructions of observed holdout rows*, not forecasts;
	// the operator may use the latent scores for transmission and monitoring,
	// but must independently validate any downstream dispatch decision.
	cells := float64(holdout.Rows() * holdout.Cols())
	modelRMSE := math.Sqrt(modelSSE / cells)
	baselineRMSE := math.Sqrt(baselineSSE / cells)
	fmt.Printf("heldout feeder RMSE: PCA=%.3f frozen-mean=%.3f MWh; better=%t\n",
		modelRMSE, baselineRMSE, modelRMSE < 0.1*baselineRMSE)
	// Output:
	// grid loads: train=168 holdout=48 feeders=10 factors=3
	// heldout feeder RMSE: PCA=0.033 frozen-mean=7.844 MWh; better=true
}

// ExampleFit_standardizedWaterQuality monitors a process with incompatible units.
// Population standardization balances six instruments before PCA; the alarm
// score is then evaluated on the conductivity channel in its original units.
//
// Implementation:
//   - Stage 1: generate 104 normal water-treatment observations from two common
//     process drivers across pH, temperature, turbidity, oxygen, flow, conductivity.
//   - Stage 2: fit two factors on 80 training observations and calibrate the
//     conductivity reconstruction tolerance on 24 normal validation observations.
//   - Stage 3: compare a normal holdout and a synthetic conductivity sensor shift.
//
// Behavior highlights:
//   - Original-unit residuals of differently dimensioned instruments are NOT
//     added into an unweighted SPE; only conductivity errors (uS/cm) are compared.
//   - PCA identifies unusual joint measurements, not the physical cause of a drift.
//
// Inputs: fixed synthetic heterogeneous sensor readings in original physical units.
// Returns: measured normal/fault conductivity deviations, calibration threshold
// in uS/cm, and the independent drift-review classifications.
// Errors: matrix/fit/reconstruction failure aborts the monitoring decision.
// Determinism: chronological fit/calibration boundary and named feature indexes.
// Complexity: O(n*d*d+rotations*d*d+n*d*k); validation O(h*d*k).
// Notes: alarm thresholds are illustrative and require site-specific certification.
// AI-Hints: never compare raw mixed-unit norms or fit scales on the fault batch.
func ExampleFit_standardizedWaterQuality() {
	ctx := context.Background()
	const trainingSamples = 80
	const totalNormalSamples = 104
	const conductivity = 4 // Conductivity in micro-Siemens per centimeter.
	base := []float64{7.2, 18, 2.5, 8.1, 420, 170}
	loadA := []float64{0.08, 1.5, 1.2, 0.4, 15, 7}
	loadB := []float64{0.03, 0.8, 0.7, 0.15, 8, 5}
	noiseScale := []float64{0.002, 0.04, 0.03, 0.012, 0.25, 0.2}
	normal := make([][]float64, totalNormalSamples)
	for row := range normal {
		a := math.Sin(float64(row) * 0.14)
		b := math.Cos(float64(row) * 0.19)
		normal[row] = make([]float64, len(base))
		for sensor := range base {
			noise := noiseScale[sensor] * math.Sin(float64((row+1)*(sensor+2))*0.37)
			normal[row][sensor] = base[sensor] + loadA[sensor]*a + loadB[sensor]*b + noise
		}
	}
	train, err := exampleMatrix(normal[:trainingSamples])
	if err != nil {
		fmt.Println("training matrix error")
		return
	}
	fit, err := pca.FitStandardizedWithComponents(ctx, train, 2)
	if err != nil {
		fmt.Println("fit error")
		return
	}
	validation, err := exampleMatrix(normal[trainingSamples:])
	if err != nil {
		fmt.Println("validation matrix error")
		return
	}
	baseline, err := fit.Model.Reconstruct(ctx, validation)
	if err != nil {
		fmt.Println("validation reconstruction error")
		return
	}
	var maxDeviation float64
	for row := 0; row < validation.Rows(); row++ {
		measured, measuredErr := validation.At(row, conductivity)
		estimated, estimatedErr := baseline.At(row, conductivity)
		if measuredErr != nil || estimatedErr != nil {
			fmt.Println("validation read error")
			return
		}
		if deviation := math.Abs(measured - estimated); deviation > maxDeviation {
			maxDeviation = deviation
		}
	}
	const calibrationMargin = 4.0
	thresholdUS := calibrationMargin * maxDeviation
	nominal := append([]float64(nil), normal[103]...)
	shifted := append([]float64(nil), normal[103]...)
	shifted[conductivity] += 80 // Synthetic probe drift in micro-Siemens/cm.
	observations, err := exampleMatrix([][]float64{nominal, shifted})
	if err != nil {
		fmt.Println("monitoring matrix error")
		return
	}
	predicted, err := fit.Model.Reconstruct(ctx, observations)
	if err != nil {
		fmt.Println("monitoring reconstruction error")
		return
	}
	deviation := make([]float64, observations.Rows())
	for row := range deviation {
		actual, actualErr := observations.At(row, conductivity)
		estimate, estimateErr := predicted.At(row, conductivity)
		if actualErr != nil || estimateErr != nil {
			fmt.Println("monitoring read error")
			return
		}
		deviation[row] = math.Abs(actual - estimate)
	}
	fmt.Printf("water quality: train=%d calibration=%d mixed-unit sensors=%d factors=%d\n",
		train.Rows(), validation.Rows(), train.Cols(), len(fit.Eigenvalues))
	// Present the calibrated single-channel decision in its physical units.
	// Mixing pH, temperature, and flow squared errors would have no valid
	// unweighted SPE interpretation even after training standardization.
	fmt.Printf("conductivity deviation: normal=%.2f drift=%.2f threshold=%.2f uS/cm\n",
		deviation[0], deviation[1], thresholdUS)
	fmt.Printf("probe review: normal=%t drift=%t\n",
		deviation[0] <= thresholdUS, deviation[1] > thresholdUS)
	// Output:
	// water quality: train=80 calibration=24 mixed-unit sensors=6 factors=2
	// conductivity deviation: normal=0.19 drift=66.02 threshold=1.26 uS/cm
	// probe review: normal=true drift=true
}

// ExampleModel_Transform proves that score drift is measured against training
// conditions rather than canceled by centering the future batch around itself.
// A production drift detector must distinguish such mean shifts from normal spread.
//
// Implementation:
//   - Stage 1: fit a one-feature model on a four-row calibration ledger.
//   - Stage 2: project two higher subsequent observations with the frozen mean.
//   - Stage 3: print the exact positive canonical scores, not test-batch anomalies.
//
// Inputs: raw training and future readings in the same instrument units.
// Returns: the two detached scores 7 and 9; errors abort the example.
// Determinism: single-feature positive loading fixes the only possible sign.
// Complexity: O(n) fit plus O(h) inference for d=k=1.
// AI-Hints: never call Fit a second time on the future readings.
func ExampleModel_Transform() {
	ctx := context.Background()
	train, err := exampleMatrix([][]float64{{10}, {12}, {14}, {16}})
	if err != nil {
		fmt.Println("training matrix error")
		return
	}
	fit, err := pca.FitDefault(ctx, train)
	if err != nil {
		fmt.Println("fit error")
		return
	}
	future, err := exampleMatrix([][]float64{{20}, {22}})
	if err != nil {
		fmt.Println("future matrix error")
		return
	}
	scores, err := fit.Model.Transform(ctx, future)
	if err != nil {
		fmt.Println("projection error")
		return
	}
	first, firstErr := scores.At(0, 0)
	second, secondErr := scores.At(1, 0)
	if firstErr != nil || secondErr != nil {
		fmt.Println("score read error")
		return
	}
	fmt.Printf("frozen training scores: %.0f %.0f\n", first, second)
	// Output:
	// frozen training scores: 7 9
}

// ExampleFit_unsupportedSolver demonstrates explicit solver admission failures.
// It never downgrades a requested direct SVD into covariance diagonalization.
//
// Implementation: request ThinSVD with nil data; classify the option failure.
// Inputs: the intentionally unavailable ThinSVD enum, no training matrix.
// Returns: an ErrUnsupportedSolver classification and a nil result.
// Determinism: validation of Solver precedes input presence and matrix access.
// Complexity: O(1) time and space because fitting never starts.
// AI-Hints: a failed solver must never publish an approximate fitted model.
func ExampleFit_unsupportedSolver() {
	result, err := pca.Fit(context.Background(), nil, pca.Options{Solver: pca.ThinSVD})
	fmt.Printf("unsupported=%t no-model=%t\n", errors.Is(err, pca.ErrUnsupportedSolver), result == nil)
	// Output:
	// unsupported=true no-model=true
}
