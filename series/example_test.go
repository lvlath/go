// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Executable examples show reproducible, realistic engineering workloads for
// grid validation and causal array transformations. All input measurements are
// illustrative fixed fixtures, not claims about real operational incidents.
// Each example validates, computes, and consumes an observable public result.
package series_test

import (
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/series"
)

// ExampleValidate audits the completeness of a water-utility pressure feed.
// The operator identifies missing five-second telemetry slots before running
// any downstream incident detection, without manufacturing replacement samples.
//
// Implementation:
//   - Stage 1: Build 30 fixed pressure readings with two missing grid slots.
//   - Stage 2: Validate exact five-second cadence with gap reporting enabled.
//   - Stage 3: Publish coverage and verify that strict admission rejects the gaps.
//
// Behavior highlights:
//   - Gap.AfterIndex refers to the received array, not the expected grid.
//   - AllowGaps never interpolates a missing pressure measurement.
//
// Inputs:
//   - Fixed UTC-referenced int64 timestamps in nanoseconds, pressure in kPa.
//
// Returns:
//   - Standard output: sample count, missing-slot locations, and coverage.
//
// Errors:
//   - ErrGap is classified with errors.Is under strict admission policy.
//
// Determinism:
//   - The same ordered samples always produce the same two gap witnesses.
//
// Complexity:
//   - Time O(n), extra Space O(g), n=30 samples and g=2 gaps.
//
// Notes:
//   - The chosen date documents the fixture; series itself is timezone-agnostic.
//
// AI-Hints:
//   - Do not treat an allowed gap as a complete or trustworthy measurement interval.
func ExampleValidate() {
	const cadence = int64(5 * time.Second)
	base := time.Date(2026, time.October, 12, 4, 0, 0, 0, time.UTC).UnixNano()

	// Slot numbers expose packet losses at expected positions 18 and 23.
	slots := []int64{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31,
	}
	pressureKPa := []float64{
		412, 411, 411, 410, 411, 412, 411, 410, 409, 410,
		411, 412, 411, 409, 408, 407, 405, 403, 400, 401,
		400, 398, 396, 394, 393, 392, 392, 391, 389, 389,
	}
	timesNS := make([]int64, len(slots))
	for i, slot := range slots {
		timesNS[i] = base + slot*cadence
	}

	grid, err := series.Validate(timesNS, pressureKPa, series.GridOptions{
		CadenceNS: cadence,
		AllowGaps: true,
	})
	if err != nil {
		fmt.Println("grid validation failed:", err)
		return
	}

	// Coverage uses the expected grid count, not just the received sample count.
	var missing int64
	for _, gap := range grid.Gaps {
		missing += gap.MissingCount
		fmt.Printf("missing after received[%d]: %d slot(s)\n", gap.AfterIndex, gap.MissingCount)
	}
	expected := int64(grid.Samples) + missing
	fmt.Printf("received=%d expected=%d coverage=%.2f%%\n",
		grid.Samples, expected, 100*float64(grid.Samples)/float64(expected))

	strictResult, strictErr := series.Validate(timesNS, pressureKPa, series.GridOptions{CadenceNS: cadence})
	fmt.Printf("strict admission rejected=%t result_nil=%t\n",
		errors.Is(strictErr, series.ErrGap), strictResult.IsNil())

	// Output:
	// missing after received[17]: 1 slot(s)
	// missing after received[21]: 1 slot(s)
	// received=30 expected=32 coverage=93.75%
	// strict admission rejected=true result_nil=true
}

// ExampleLag compares feeder demand against its two-hour historical baseline.
// A grid-verified load monitor can identify rapid ramps without using future
// readings or treating the initial warmup as actual zero consumption.
//
// Implementation:
//   - Stage 1: Supply 32 half-hourly feeder-power observations in kW.
//   - Stage 2: Obtain a four-sample causal lag and traverse valid comparisons.
//   - Stage 3: Report high-ramp events and the largest observed rise.
//
// Behavior highlights:
//   - Lag measures sample positions; uniform timing is an upstream assumption.
//
// Inputs:
//   - Fixed feeder load in kW and a four-sample/two-hour horizon.
//
// Returns:
//   - Standard output: valid comparison count, alarms, maximum rise and index.
//
// Errors:
//   - Any invalid lag or nonfinite sample aborts before output publication.
//
// Determinism:
//   - Candidate positions are inspected in ascending sample-index order.
//
// Complexity:
//   - Time O(n), Space O(n), including detached lag and validity arrays.
//
// Notes:
//   - A ramp alarm requests capacity investigation; it does not predict failures.
//
// AI-Hints:
//   - Never compare against lag warmup cells or equate four positions with
//     two hours unless the source cadence has been validated separately.
func ExampleLag() {
	const (
		lookbackSamples = 4
		rampAlarmKW     = 300.0
	)
	feederKW := []float64{
		820, 810, 800, 790, 780, 775, 770, 760,
		755, 750, 760, 770, 805, 850, 900, 970,
		1060, 1150, 1240, 1350, 1450, 1540, 1630, 1750,
		1735, 1680, 1580, 1490, 1390, 1320, 1240, 1180,
	}

	baseline, err := series.Lag(feederKW, lookbackSamples)
	if err != nil {
		fmt.Println("baseline failed:", err)
		return
	}

	validCount, alarmCount, peakIndex := 0, 0, -1
	peakRiseKW := 0.0
	for i, valid := range baseline.Valid {
		if !valid {
			continue
		}
		validCount++
		riseKW := feederKW[i] - baseline.Values[i]
		if riseKW >= rampAlarmKW {
			alarmCount++
		}
		if riseKW > peakRiseKW {
			peakRiseKW, peakIndex = riseKW, i
		}
	}

	fmt.Printf("two-hour comparisons=%d rapid ramps=%d\n", validCount, alarmCount)
	fmt.Printf("largest rise=%.0f kW at sample=%d\n", peakRiseKW, peakIndex)

	// Output:
	// two-hour comparisons=28 rapid ramps=7
	// largest rise=400 kW at sample=23
}

// ExampleDifference locates a fast pressure decline in a pumping subsystem.
// A site reliability engineer compares readings 30 minutes apart to identify
// an investigation candidate, without assuming that the decline proves a leak.
//
// Implementation:
//   - Stage 1: Supply 32 five-minute pressure readings in kPa.
//   - Stage 2: Compute a six-sample finite difference with an explicit mask.
//   - Stage 3: Report the first severe decline and the worst decline observed.
//
// Behavior highlights:
//   - Difference returns signed changes; negative values indicate pressure loss.
//
// Inputs:
//   - Representative pressure in kPa; six samples represent 30 minutes.
//
// Returns:
//   - Standard output: evaluated count and severity statistics.
//
// Errors:
//   - ErrOverflow rejects an unrepresentable subtraction rather than +Inf.
//
// Determinism:
//   - Each eligible pair is evaluated once in increasing index order.
//
// Complexity:
//   - Time O(n), Space O(n) for validation and detached result arrays.
//
// Notes:
//   - Validate timestamp cadence before treating an index lag as a duration.
//
// AI-Hints:
//   - A fall in pressure warrants inspection; it is not a causal diagnosis.
func ExampleDifference() {
	const (
		horizonSamples = 6
		alarmDropKPa   = -55.0
	)
	pressureKPa := []float64{
		480, 479, 480, 481, 480, 479, 480, 479,
		478, 477, 478, 476, 475, 476, 474, 473,
		472, 471, 468, 445, 414, 402, 397, 394,
		392, 391, 390, 390, 392, 395, 401, 405,
	}

	changes, err := series.Difference(pressureKPa, horizonSamples)
	if err != nil {
		fmt.Println("difference failed:", err)
		return
	}

	validCount, alarmCount, firstAlarm, worstIndex := 0, 0, -1, -1
	worstDrop := 0.0
	for i, valid := range changes.Valid {
		if !valid {
			continue
		}
		validCount++
		change := changes.Values[i]
		if change <= alarmDropKPa {
			alarmCount++
			if firstAlarm < 0 {
				firstAlarm = i
			}
		}
		if change < worstDrop {
			worstDrop, worstIndex = change, i
		}
	}

	fmt.Printf("half-hour comparisons=%d severe declines=%d first=%d\n", validCount, alarmCount, firstAlarm)
	fmt.Printf("worst change=%.0f kPa at sample=%d\n", worstDrop, worstIndex)

	// Output:
	// half-hour comparisons=26 severe declines=5 first=20
	// worst change=-77 kPa at sample=23
}

// ExampleSimpleReturns applies a one-observation loss gate to collateral marks.
// A treasury monitor inspects sequential recorded valuations for abrupt negative
// relative changes without claiming that the samples follow a trading calendar.
//
// Implementation:
//   - Stage 1: Provide 30 representative positive valuation marks.
//   - Stage 2: Compute causal adjacent simple returns and ignore warmup.
//   - Stage 3: Report threshold breaches and the largest observed loss.
//
// Behavior highlights:
//   - A return of zero is valid when Valid[i] is true; warmup is zero/false.
//
// Inputs:
//   - Fixed USD-per-unit mark series and a -1.2% screening threshold.
//
// Returns:
//   - Standard output: breaches, largest percentage loss, and its sample index.
//
// Errors:
//   - ErrDomain rejects only zero values used as previous denominators.
//
// Determinism:
//   - Returns and threshold checks are performed in input order.
//
// Complexity:
//   - Time O(n), Space O(n), including full input validation and masks.
//
// Notes:
//   - This is a monitoring criterion, not a trading recommendation.
//
// AI-Hints:
//   - Do not assign financial meaning to negative-base relative changes without
//     a separate domain model; series accepts them mathematically.
func ExampleSimpleReturns() {
	const lossScreen = -0.012
	marks := []float64{
		100.0, 100.4, 100.8, 100.3, 99.9, 100.7,
		101.2, 100.1, 99.4, 98.8, 99.1, 99.7,
		100.2, 100.9, 101.7, 102.3, 102.0, 101.1,
		100.6, 99.3, 98.0, 97.1, 95.8, 96.3,
		97.0, 97.5, 98.2, 99.1, 99.8, 100.4,
	}

	changes, err := series.SimpleReturns(marks)
	if err != nil {
		fmt.Println("relative change failed:", err)
		return
	}

	breaches, worstIndex := 0, -1
	worstReturn := 0.0
	for i, valid := range changes.Valid {
		if !valid {
			continue
		}
		v := changes.Values[i]
		if v <= lossScreen {
			breaches++
		}
		if v < worstReturn {
			worstReturn, worstIndex = v, i
		}
	}

	fmt.Printf("marks=%d one-step loss breaches=%d\n", len(marks), breaches)
	fmt.Printf("worst return=%.2f%% at sample=%d\n", 100*worstReturn, worstIndex)

	// Output:
	// marks=30 one-step loss breaches=3
	// worst return=-1.34% at sample=22
}

// ExampleLogReturns detects multiplicative surges in CDN request intensity.
// An operations analyst compares strictly positive one-minute request rates
// using natural-log changes, then converts the largest change to a ratio.
//
// Implementation:
//   - Stage 1: Provide 32 positive, representative request-rate observations.
//   - Stage 2: Compute stable adjacent log returns and classify >60% increases.
//   - Stage 3: Report burst count and the largest one-step multiplicative jump.
//
// Behavior highlights:
//   - Only positive finite samples are permitted; output is a log change.
//
// Inputs:
//   - Measured-like request rates per minute (illustrative synthetic fixture).
//
// Returns:
//   - Standard output: burst count, maximum log change, ratio and index.
//
// Errors:
//   - ErrDomain rejects nonpositive source values before arithmetic.
//
// Determinism:
//   - The complete input domain is checked before ordered pair evaluation.
//
// Complexity:
//   - Time O(n), Space O(n) with detached values and masks.
//
// Notes:
//   - A demand surge can suggest scaling; series does not perform forecasting.
//
// AI-Hints:
//   - Compare log changes with a log threshold; do not compare them directly
//     with a percentage increase or assume regular timestamps automatically.
func ExampleLogReturns() {
	const burstMultiplier = 1.6
	requestsPerMinute := []float64{
		1200, 1220, 1250, 1240, 1260, 1280, 1310, 1300,
		1330, 1380, 1400, 1420, 1460, 1500, 1550, 1700,
		2800, 3150, 3210, 3180, 3100, 3000, 2950, 3020,
		4900, 5200, 5100, 4700, 4550, 4300, 4150, 4050,
	}

	changes, err := series.LogReturns(requestsPerMinute)
	if err != nil {
		fmt.Println("log changes failed:", err)
		return
	}

	threshold := math.Log(burstMultiplier)
	burstCount, peakIndex := 0, -1
	peakLogChange := 0.0
	for i, valid := range changes.Valid {
		if !valid {
			continue
		}
		v := changes.Values[i]
		if v >= threshold {
			burstCount++
		}
		if v > peakLogChange {
			peakLogChange, peakIndex = v, i
		}
	}

	fmt.Printf("minute samples=%d bursts=%d\n", len(requestsPerMinute), burstCount)
	fmt.Printf("largest jump=%.3f log units (%.2fx) at sample=%d\n",
		peakLogChange, math.Exp(peakLogChange), peakIndex)

	// Output:
	// minute samples=32 bursts=2
	// largest jump=0.499 log units (1.65x) at sample=16
}

// ExampleWindowRanges scans complete overlapping windows of API error counts.
// The on-call engineer checks 12-minute incident windows every six minutes and
// deliberately excludes a two-minute tail that cannot form a complete window.
//
// Implementation:
//   - Stage 1: Build 62 fixed one-minute counts with one elevated incident band.
//   - Stage 2: Enumerate complete half-open windows with width 12 and stride 6.
//   - Stage 3: Aggregate errors in each complete window and report alarms.
//
// Behavior highlights:
//   - WindowRanges only provides indices; aggregation belongs to this caller.
//
// Inputs:
//   - Representative per-minute error counts and explicit index-window policy.
//
// Returns:
//   - Standard output: qualifying window summaries and uncovered tail count.
//
// Errors:
//   - ErrInvalidOption rejects invalid dimensions; ErrOverflow guards size math.
//
// Determinism:
//   - Windows advance by exactly six indices in increasing start order.
//
// Complexity:
//   - Window enumeration O(w) time/space; this caller sums O(w*width) counts.
//
// Notes:
//   - A window is not a calendar interval; samples must be prevalidated if needed.
//
// AI-Hints:
//   - Never extend End to cover the tail: an incomplete window changes the metric.
func ExampleWindowRanges() {
	const (
		width         = 12
		stride        = 6
		incidentLimit = 120
	)

	// The elevated period simulates a reproducible API incident, not random traffic.
	errorsPerMinute := make([]int, 62)
	for i := range errorsPerMinute {
		errorsPerMinute[i] = 2 + i%4
		if i >= 24 && i < 40 {
			errorsPerMinute[i] = 16 + i%5
		}
	}

	windows, err := series.WindowRanges(len(errorsPerMinute), width, stride)
	if err != nil {
		fmt.Println("window construction failed:", err)
		return
	}

	alerts := 0
	for _, window := range windows {
		// Half-open boundaries ensure every complete window has exactly width cells.
		totalErrors := 0
		for i := window.Start; i < window.End; i++ {
			totalErrors += errorsPerMinute[i]
		}
		if totalErrors >= incidentLimit {
			alerts++
			fmt.Printf("incident window=[%d,%d) errors=%d\n", window.Start, window.End, totalErrors)
		}
	}

	fmt.Printf("complete windows=%d alarms=%d uncovered tail=%d\n",
		len(windows), alerts, len(errorsPerMinute)-windows[len(windows)-1].End)

	// Output:
	// incident window=[18,30) errors=133
	// incident window=[24,36) errors=216
	// incident window=[30,42) errors=185
	// complete windows=9 alarms=3 uncovered tail=2
}

// ExampleLogReturns_extremeDynamicRange protects a numeric boundary condition.
// Finite positive values spanning 600 decimal orders and one-ULP changes must
// not be confused with an overflowing ratio or an exact zero log change.
//
// Implementation:
//   - Stage 1: Supply an extreme positive pair and a one-ULP successor.
//   - Stage 2: Compute the same public LogReturns contract.
//   - Stage 3: Assert qualitative finite/positive properties in stable output.
//
// Behavior highlights:
//   - The kernel chooses log differences and log1p according to numeric distance.
//
// Inputs:
//   - Fixed IEEE-754 float64 stress values, not measured telemetry.
//
// Returns:
//   - Standard output: boolean finiteness and positivity certificates.
//
// Errors:
//   - ErrOverflow would reject an unrepresentable computed return.
//
// Determinism:
//   - Fixed operations with no data-dependent iteration order.
//
// Complexity:
//   - Time O(n), Space O(n), n=3.
//
// Notes:
//   - This is a special-law demonstration alongside the operational examples.
//
// AI-Hints:
//   - Never replace this with math.Log(curr/prev) or naive subtraction of
//     rounded logarithms for adjacent large floating-point values.
func ExampleLogReturns_extremeDynamicRange() {
	values := []float64{1e-300, 1e300, math.Nextafter(1e300, math.Inf(1))}
	result, err := series.LogReturns(values)
	if err != nil {
		fmt.Println("numeric stability check failed:", err)
		return
	}
	fmt.Printf("wide dynamic range finite=%t, single-ULP return positive=%t\n",
		!math.IsInf(result.Values[1], 0), result.Values[2] > 0)

	// Output:
	// wide dynamic range finite=true, single-ULP return positive=true
}

// ExampleTransformResult_IsNil illustrates typed-nil result handling in adapters.
// A generic consumer can distinguish a nil result pointer from a valid
// length-preserving artifact even when both are stored as core.Nilable.
//
// Implementation:
//   - Stage 1: Store a typed nil pointer behind core.Nilable.
//   - Stage 2: Produce a non-nil result through the regular Lag facade.
//   - Stage 3: Compare nilability without reading result buffers or reflecting.
//
// Behavior highlights:
//   - A typed-nil interface is non-nil while its dynamic result pointer is nil.
//
// Inputs:
//   - Fixed two-observation series; pointer-only interface adapter.
//
// Returns:
//   - Standard output: typed-nil and successful result classifications.
//
// Errors:
//   - Lag retains its usual ErrNilInput/ErrEmptyInput/ErrNonFinite protocol.
//
// Determinism:
//   - Pointer checks are deterministic and side-effect free.
//
// Complexity:
//   - IsNil O(1) time/space; Lag O(n) time/space.
//
// Notes:
//   - This example imports core only in external example code, not in series.
//
// AI-Hints:
//   - Do not use interfaceValue == nil for typed-nil result detection.
func ExampleTransformResult_IsNil() {
	var missing *series.TransformResult
	var optional core.Nilable = missing
	present, err := series.Lag([]float64{10, 12}, 1)
	if err != nil {
		fmt.Println("lag failed:", err)
		return
	}
	fmt.Printf("interface_non_nil=%t missing=%t published=%t\n",
		optional != nil, optional.IsNil(), present.IsNil())

	// Output:
	// interface_non_nil=true missing=true published=false
}
