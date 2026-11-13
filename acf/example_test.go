// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package acf_test demonstrates end-to-end, reproducible diagnostics with the public ACF API.
// Data are explicitly synthetic engineering fixtures, not claims about observed live markets or machines.
package acf_test

import (
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/acf"
)

// ExampleComputeDefault_coldStorageLoad demonstrates daily and weekly periodicity
// discovery in four weeks of hourly industrial refrigeration electricity demand.
//
// Scenario:
//   - A cold-storage operator receives one power-meter sample per hour.
//   - Compressors cycle continuously, while weekday loading-dock traffic adds demand.
//   - Engineers compare the full-day and full-week lags before selecting forecasting
//     seasonality; the ACF alone does not validate a forecasting model.
//
// Process:
//   - Build 672 fixed hourly power samples with daily/weekly components.
//   - Use centered common-denominator ACF so future Toeplitz-based diagnostics
//     receive a compatible estimator convention.
//   - Report lag strengths and structural pair counts for comparable sample sizes.
//
// Complexity:
//   - Building the fixed series is O(n); the direct ACF is O(n*maxLag) time
//     with O(n+maxLag) memory, where n=672 and maxLag=168.
func ExampleComputeDefault_coldStorageLoad() {
	const hoursPerDay = 24
	const daysPerWeek = 7
	const weeks = 4
	const maxLag = hoursPerDay * daysPerWeek

	powerKW := make([]float64, weeks*maxLag)
	for i := range powerKW {
		hour := i % hoursPerDay
		weekday := (i / hoursPerDay) % daysPerWeek

		// Refrigeration is present every day; freight operations are reduced
		// on weekends, and the compressor envelope has a 24-hour cycle.
		compressor := 16.0 + 5.0*math.Cos(2*math.Pi*float64(hour)/hoursPerDay)
		dock := 0.0
		if weekday < 5 && hour >= 7 && hour < 19 {
			dock = 15
		}
		powerKW[i] = 42 + compressor + dock + 1.2*math.Sin(2*math.Pi*float64(i)/8)
	}

	result, err := acf.ComputeDefault(powerKW, maxLag)
	if err != nil {
		fmt.Println("calculation failed:", err)
		return
	}
	fmt.Printf("hourly samples=%d daily(r24)=%.3f weekly(r168)=%.3f\n", len(powerKW), result.Values[24], result.Values[168])
	fmt.Printf("available pairs: daily=%d weekly=%d\n", result.PairCounts[24], result.PairCounts[168])
	// Output:
	// hourly samples=672 daily(r24)=0.522 weekly(r168)=0.750
	// available pairs: daily=648 weekly=504
}

// ExampleComputeDefault_volatilityClustering compares serial dependence in
// signed daily returns with persistence in absolute return magnitudes.
//
// Scenario:
//   - A risk-service pipeline evaluates 160 consecutive daily log-return slots.
//   - Quiet, stressed, and recovery regimes change amplitude without assigning
//     predictable daily direction; this is a reproducible market-like fixture.
//   - Raw return ACF and absolute-return ACF answer different questions and
//     must be compared, not confused with prediction accuracy or a p-value.
//
// Process:
//   - Build returns using fixed volatility regimes and a deterministic sign source.
//   - Fit both ACFs on exactly the same 160-day observation window.
//   - Gate a volatility-memory review using a deliberately illustrative threshold.
//
// Complexity:
//   - O(n) input construction and O(n*maxLag) direct evaluation; O(n+maxLag) memory.
func ExampleComputeDefault_volatilityClustering() {
	const daysPerRegime = 20
	volatilityByRegime := []float64{0.003, 0.004, 0.018, 0.016, 0.004, 0.015, 0.003, 0.004}
	returns := make([]float64, len(volatilityByRegime)*daysPerRegime)
	absoluteReturns := make([]float64, len(returns))

	// Fixed pseudo-random variation avoids embedding a live network dependency
	// and never pretends these are historical trading results.
	seed := uint32(0x4d595df4)
	for day := range returns {
		seed = seed*1664525 + 1013904223
		volatility := volatilityByRegime[day/daysPerRegime]
		magnitude := volatility * (0.70 + 0.06*float64((seed>>12)%9))
		if (seed>>23)&1 == 0 {
			returns[day] = -magnitude
		} else {
			returns[day] = magnitude
		}
		absoluteReturns[day] = math.Abs(returns[day])
	}

	signed, err := acf.ComputeDefault(returns, 10)
	if err != nil {
		fmt.Println("signed ACF failed:", err)
		return
	}
	amplitude, err := acf.ComputeDefault(absoluteReturns, 10)
	if err != nil {
		fmt.Println("amplitude ACF failed:", err)
		return
	}

	// This threshold is an operational illustrative gate, not a statistical
	// significance test; decisions need an explicit model and risk policy.
	const reviewThreshold = 0.50
	fmt.Printf("samples=%d signed-r1=%+.3f |returns|-r1=%+.3f\n", len(returns), signed.Values[1], amplitude.Values[1])
	fmt.Printf("volatility review=%t pairs=%d\n", amplitude.Values[1] > reviewThreshold, amplitude.PairCounts[1])
	// Output:
	// samples=160 signed-r1=+0.106 |returns|-r1=+0.893
	// volatility review=true pairs=159
}

// ExampleComputeUncentered_pitchDetection estimates a known voice-like tone
// period in an AC-coupled microphone buffer with harmonic distortion and noise.
//
// Scenario:
//   - An embedded audio front end provides 8 kHz samples with an established
//     physical zero and a limited speech fundamental-frequency search window.
//   - The tone has a fundamental plus a second harmonic and high-frequency noise;
//     reading only the strongest FFT bin could incorrectly select a harmonic.
//   - Search the expected lag interval explicitly and convert the winning lag
//     into a candidate pitch; this is not a complete voiced/unvoiced classifier.
//
// Process:
//   - Generate 0.2 seconds of deterministic AC-coupled test audio.
//   - Compute uncentered common-denominator ACF through the public shortcut.
//   - Find the maximum correlation in the supported 150-250 Hz period interval.
//
// Complexity:
//   - O(n) signal construction and O(n*maxLag) ACF work; O(n+maxLag) memory.
func ExampleComputeUncentered_pitchDetection() {
	const sampleRateHz = 8000
	const referencePeriod = 40
	const sampleCount = 1600
	const shortestPeriod = 32
	const longestPeriod = 53

	samples := make([]float64, sampleCount)
	for i := range samples {
		phase := 2 * math.Pi * float64(i) / referencePeriod
		samples[i] = math.Sin(phase) + 0.38*math.Sin(2*phase+0.15) + 0.08*math.Sin(2*math.Pi*float64(i)/7)
	}

	result, err := acf.ComputeUncentered(samples, longestPeriod)
	if err != nil {
		fmt.Println("pitch analysis failed:", err)
		return
	}
	candidatePeriod := shortestPeriod
	for lag := shortestPeriod + 1; lag <= longestPeriod; lag++ {
		if result.Values[lag] > result.Values[candidatePeriod] {
			candidatePeriod = lag
		}
	}

	fmt.Printf("candidate pitch=%d Hz (lag=%d, score=%.3f)\n", sampleRateHz/candidatePeriod, candidatePeriod, result.Values[candidatePeriod])
	fmt.Printf("samples=%d pairs-at-pitch=%d\n", len(samples), result.PairCounts[candidatePeriod])
	// Output:
	// candidate pitch=200 Hz (lag=40, score=0.968)
	// samples=1600 pairs-at-pitch=1560
}

// ExampleComputeDefault_bearingFaultPeriodicity compares healthy and impacted
// vibration windows for a candidate periodic bearing-impact signature.
//
// Scenario:
//   - A maintenance system samples an industrial spindle at 12.8 kHz.
//   - A harmonic motor carrier and deterministic broadband interference exist
//     in healthy operation, while the suspect channel also includes periodic
//     damped impacts at a known inspection-test frequency.
//   - An elevated ACF peak supports a targeted vibration review, not a conclusive
//     diagnosis of a particular bearing defect or fault severity.
//
// Process:
//   - Build independent baseline and suspect channels with matched carrier noise.
//   - Compute centered ACF on both fixed acquisition windows.
//   - Search a physically justified lag band and compare the peak to baseline.
//
// Complexity:
//   - O(n) fixture construction and O(n*maxLag) per ACF; O(n+maxLag) memory.
func ExampleComputeDefault_bearingFaultPeriodicity() {
	const sampleRateHz = 12800
	const sampleCount = 2048
	const impulseSpacing = 64
	const searchFrom = 48
	const searchThrough = 80

	healthy := make([]float64, sampleCount)
	suspect := make([]float64, sampleCount)
	for i := range healthy {
		motor := 0.50*math.Sin(2*math.Pi*float64(i)/29) + 0.13*math.Sin(2*math.Pi*float64(i)/17)
		interference := 0.04 * math.Sin(2*math.Pi*float64(i*i%113)/113)
		healthy[i] = motor + interference

		// A damped impact excites a short resonant waveform once every
		// 64 samples, modeling a repeatable defect rather than random noise.
		impactPhase := i % impulseSpacing
		impact := 3.5 * math.Exp(-float64(impactPhase)/2.3) * math.Cos(2*math.Pi*float64(impactPhase)/5)
		suspect[i] = healthy[i] + impact
	}

	baseline, err := acf.ComputeDefault(healthy, searchThrough)
	if err != nil {
		fmt.Println("baseline failed:", err)
		return
	}
	fault, err := acf.ComputeDefault(suspect, searchThrough)
	if err != nil {
		fmt.Println("suspect failed:", err)
		return
	}

	candidate := searchFrom
	for lag := searchFrom + 1; lag <= searchThrough; lag++ {
		if fault.Values[lag] > fault.Values[candidate] {
			candidate = lag
		}
	}

	fmt.Printf("candidate repeat rate=%d Hz, lag=%d\n", sampleRateHz/candidate, candidate)
	fmt.Printf("baseline-r=%+.3f suspect-r=%+.3f pairs=%d\n", baseline.Values[candidate], fault.Values[candidate], fault.PairCounts[candidate])
	// Output:
	// candidate repeat rate=200 Hz, lag=64
	// baseline-r=+0.250 suspect-r=+0.706 pairs=1984
}

// ExampleComputeLagAdjusted_calibrationDrift demonstrates why a long-lag
// adjusted covariance-ratio coefficient is not a bounded Pearson correlation.
//
// Scenario:
//   - A sensor calibrator records a 12-step thermal settling ramp at equal intervals.
//   - The last lag is supported by exactly one product, while lag zero uses 12.
//   - Pair-count normalization can therefore amplify the long-lag estimate
//     beyond one in magnitude without indicating an arithmetic failure.
//
// Process:
//   - Fit both conventions to the identical short observation window.
//   - Compare late-lag estimates and their actual available pairs.
//   - Keep the common estimator for a later PSD-dependent PACF computation.
//
// Complexity:
//   - O(n*maxLag) direct scans, O(n+maxLag) memory for each result.
func ExampleComputeLagAdjusted_calibrationDrift() {
	calibrationCelsius := []float64{41.0, 41.1, 41.3, 41.4, 41.6, 41.8, 42.0, 42.3, 42.5, 42.8, 43.0, 43.3}
	lastLag := len(calibrationCelsius) - 1

	common, err := acf.ComputeDefault(calibrationCelsius, lastLag)
	if err != nil {
		fmt.Println("common estimator failed:", err)
		return
	}
	adjusted, err := acf.ComputeLagAdjusted(calibrationCelsius, lastLag)
	if err != nil {
		fmt.Println("adjusted estimator failed:", err)
		return
	}

	fmt.Printf("lag=%d common=%+.3f adjusted=%+.3f\n", lastLag, common.Values[lastLag], adjusted.Values[lastLag])
	fmt.Printf("available pairs=%d adjusted-outside-bounds=%t\n", adjusted.PairCounts[lastLag], math.Abs(adjusted.Values[lastLag]) > 1)
	// Output:
	// lag=11 common=-0.199 adjusted=-2.394
	// available pairs=1 adjusted-outside-bounds=true
}

// ExampleComputeUncenteredLagAdjusted_standbyCurrent distinguishes a physically
// nonzero constant current from a zero-variance centered observation window.
//
// Scenario:
//   - A UPS controller collects a 48-sample stable standby current trace.
//   - The current is truly nonzero, so uncentered second-moment correlation is
//     defined even though centered serial covariance has zero energy.
//   - Treating raw second-moment persistence as varying-current evidence would
//     produce false alarms; the two meanings must remain separate.
//
// Process:
//   - Build a flat, zero-referenced current trace.
//   - Compute raw pair-adjusted ACF and independently classify centered failure.
//   - Read estimator metadata, lag coefficients, and sample-pair counts.
//
// Complexity:
//   - O(n*maxLag) time and O(n+maxLag) space for the valid raw estimator.
func ExampleComputeUncenteredLagAdjusted_standbyCurrent() {
	const samplesInWindow = 48
	const lagInSamples = 12

	standbyAmperes := make([]float64, samplesInWindow)
	for i := range standbyAmperes {
		standbyAmperes[i] = 2.4
	}
	raw, err := acf.ComputeUncenteredLagAdjusted(standbyAmperes, lagInSamples)
	if err != nil {
		fmt.Println("raw ACF failed:", err)
		return
	}
	centered, centeredErr := acf.ComputeDefault(standbyAmperes, lagInSamples)

	fmt.Printf("raw-r12=%.3f mean-metadata=%.0f pairs=%d\n", raw.Values[lagInSamples], raw.Mean, raw.PairCounts[lagInSamples])
	fmt.Printf("centered-undefined=%t nil-result=%t\n", errors.Is(centeredErr, acf.ErrZeroVariance), centered == nil)
	// Output:
	// raw-r12=1.000 mean-metadata=0 pairs=36
	// centered-undefined=true nil-result=true
}
