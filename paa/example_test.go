// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package paa examples demonstrate decision-oriented, deterministic reductions
// of uniform telemetry. This file is external-package executable documentation:
// every scenario builds a realistic observation stream, runs a public facade,
// and uses the resulting exact geometry or numeric evidence for one decision.
//
// AI-Hints:
//   - PAA stores a segment AVERAGE, not a peak detector or lossless event log.
//   - Sample indices describe equal sampling intervals, not absolute times;
//     external timestamps must first be checked for cadence and gaps.
//   - Do not infer a DTW bound from a Euclidean PAA projection.
package paa_test

import (
	"fmt"
	"math"

	"github.com/lvlath/go/paa"
)

// ExampleTransform_sensorFleet summarizes rack inlet temperatures in a sampled
// cooling-monitor trace and locates an elevated average interval for inspection.
// It demonstrates fractional groups without inventing samples or using a
// single instantaneous spike as evidence of a sustained condition.
func ExampleTransform_sensorFleet() {
	// A data-center inlet probe reports one finite Celsius reading every five
	// minutes. Nineteen readings do not split into six integer-sized intervals.
	celsius := []float64{
		28.0, 28.2, 28.5, 29.0, 29.2, 29.1, 29.3,
		31.0, 33.6, 35.7, 38.4, 37.9, 36.0, 34.2,
		32.8, 31.2, 30.1, 29.6, 29.2,
	}
	const groups = 6
	const reviewThresholdC = 35.0

	// The canonical result records both the means and exact source intervals.
	result, err := paa.Transform(celsius, groups)
	if err != nil {
		fmt.Println("PAA:", err)
		return
	}

	// Rank the aggregated intervals for operator triage. A segment mean above
	// threshold flags an elevated interval, but cannot certify how long each
	// individual input exceeded the threshold or replace raw peak alarms.
	hottest := 0
	for j := 1; j < len(result.Values); j++ {
		if result.Values[j] > result.Values[hottest] {
			hottest = j
		}
	}
	interval := result.Segments[hottest]
	last := result.Segments[len(result.Segments)-1]
	fmt.Printf("samples=%d groups=%d hot=%d average=%.1fC review=%t\n",
		result.OriginalLength, len(result.Values), hottest,
		result.Values[hottest], result.Values[hottest] >= reviewThresholdC)
	fmt.Printf("hot-window=[%d/%d,%d/%d) final-end=%d/%d\n",
		interval.LeftNumerator, interval.Denominator,
		interval.RightNumerator, interval.Denominator,
		last.RightNumerator, last.Denominator)
	// Output:
	// samples=19 groups=6 hot=3 average=37.3C review=true
	// hot-window=[57/6,76/6) final-end=114/6
}

// ExampleTransform_loadTelemetry protects the final samples in a sampled
// service-latency stream. Dropping a short remainder block would hide an
// end-of-shift degradation from the compressed operational summary.
func ExampleTransform_loadTelemetry() {
	// One end-to-end request-latency measurement (milliseconds) per minute.
	// A late incident raises the final sample without changing sample cadence.
	latencyMS := []float64{
		90, 95, 87, 88, 92, 100, 104, 98,
		103, 99, 96, 102, 99, 104, 110, 112,
		111, 106, 101, 108, 109, 125, 950,
	}
	const groups = 7
	const meanLatencyReviewMS = 250.0

	result, err := paa.Transform(latencyMS, groups)
	if err != nil {
		fmt.Println("PAA:", err)
		return
	}

	// The final group covers the real interval ending at sample 23 even
	// though 23/7 is nonintegral. It contains the actual late incident.
	lastIndex := len(result.Values) - 1
	lastMean := result.Values[lastIndex]
	lastCell := result.Segments[lastIndex]
	fmt.Printf("last-group=%d mean=%.1fms investigate=%t\n",
		lastIndex, lastMean, lastMean > meanLatencyReviewMS)
	fmt.Printf("interval=[%d/%d,%d/%d) raw-tail=%.0fms\n",
		lastCell.LeftNumerator, lastCell.Denominator,
		lastCell.RightNumerator, lastCell.Denominator,
		latencyMS[len(latencyMS)-1])
	// Output:
	// last-group=6 mean=369.7ms investigate=true
	// interval=[138/7,161/7) raw-tail=950ms
}

// ExampleTransform_energyBudget aggregates quarter-hour pump demand into six
// equal-duration accounting windows and compares energy before and after PAA.
// An average power level is not an energy total: elapsed time must be applied
// to each compressed segment to recover the integrated quantity.
func ExampleTransform_energyBudget() {
	// Each sample is the average kilowatt demand over one 15-minute interval.
	// The 25th interval must contribute in full to the compressed total.
	powerKW := []float64{
		4.1, 4.3, 4.4, 4.8, 5.2, 5.3, 5.6,
		5.9, 6.2, 6.7, 7.1, 7.3, 7.0, 6.8,
		6.5, 6.1, 5.8, 5.4, 5.1, 4.9, 4.8,
		4.6, 4.4, 4.2, 7.2,
	}
	const minutesPerSample = 15
	const minutesPerHour = 60
	const groups = 6

	result, err := paa.Transform(powerKW, groups)
	if err != nil {
		fmt.Println("PAA:", err)
		return
	}

	// Integrate the original step function in physical kilowatt-hours.
	// Apply the same total-duration law to the reduced segment averages.
	sampleHours := float64(minutesPerSample) / minutesPerHour
	segmentHours := float64(result.OriginalLength) * sampleHours / float64(groups)
	var originalKWh, reducedKWh float64
	for _, power := range powerKW {
		originalKWh += power * sampleHours
	}
	for _, averagePower := range result.Values {
		reducedKWh += averagePower * segmentHours
	}

	// This scale is deliberately ordinary; a small tolerance accounts for the
	// two independent float64 summations, not any invented energy samples.
	const energyToleranceKWh = 1e-10
	end := result.Segments[len(result.Segments)-1]
	fmt.Printf("raw=%.3fkWh reduced=%.3fkWh conserved=%t\n",
		originalKWh, reducedKWh, math.Abs(originalKWh-reducedKWh) < energyToleranceKWh)
	fmt.Printf("source-end=%d/%d samples groups=%d\n",
		end.RightNumerator, end.Denominator, len(result.Values))
	// Output:
	// raw=34.925kWh reduced=34.925kWh conserved=true
	// source-end=150/6 samples groups=6
}

// ExampleTransform_extremeValues checks signed cancellation at IEEE-754 limits.
// It is an arithmetic-safety diagnostic for scientific pipelines whose raw
// finite values can differ by thousands of binary orders of magnitude.
func ExampleTransform_extremeValues() {
	// A constant maximum-magnitude signal has a finite and representable mean.
	large, err := paa.Transform([]float64{math.MaxFloat64, math.MaxFloat64}, 1)
	if err != nil {
		fmt.Println("constant:", err)
		return
	}

	// Three strong positive/negative pairs cancel exactly. The remaining
	// subnormal contributions still form a nonzero exactly representable mean.
	// A naive floating sum or an unnormalized x*overlap could lose this residue.
	minimal := math.SmallestNonzeroFloat64
	signal := []float64{
		math.MaxFloat64, -math.MaxFloat64, 4 * minimal,
		math.MaxFloat64, -math.MaxFloat64, 4 * minimal,
		math.MaxFloat64, -math.MaxFloat64, 4 * minimal,
	}
	balanced, err := paa.Transform(signal, 1)
	if err != nil {
		fmt.Println("balanced:", err)
		return
	}

	fmt.Printf("large-finite=%t exact-subnormal-residue=%t bits=%d\n",
		!math.IsInf(large.Values[0], 0) && large.Values[0] == math.MaxFloat64,
		balanced.Values[0] == minimal, math.Float64bits(balanced.Values[0]))
	// Output:
	// large-finite=true exact-subnormal-residue=true bits=1
}

// ExampleTransform_detachedIdentity turns an unchanged series into a detached
// input-aligned Result. A caller can annotate the published analytics snapshot
// without modifying the upstream audit buffer or losing signed-zero identity.
func ExampleTransform_detachedIdentity() {
	// An immutable upstream baseline contains 15 signed risk deltas, including
	// an intentionally signed zero produced by a measurement process.
	baseline := []float64{
		math.Copysign(0, -1), 0.01, -0.03, 0.05, 0.12,
		0.06, -0.02, 0.01, 0.00, -0.04,
		0.07, 0.03, 0.11, -0.01, 0.02,
	}

	// w=n is an exact detached identity; segment geometry remains available
	// to a downstream consumer using a uniform Result-based interface.
	snapshot, err := paa.Transform(baseline, len(baseline))
	if err != nil {
		fmt.Println("PAA:", err)
		return
	}
	negativeZeroKept := math.Signbit(snapshot.Values[0])

	// Deliberately annotate only caller-owned snapshot storage after return.
	// Neither this mutation nor changing segment metadata can affect baseline.
	snapshot.Values[1] = 0.99
	snapshot.Segments[1].LeftNumerator = -1
	fmt.Printf("baseline=%.2f snapshot=%.2f negative-zero=%t detached=%t\n",
		baseline[1], snapshot.Values[1], negativeZeroKept,
		baseline[1] != snapshot.Values[1])
	// Output:
	// baseline=0.01 snapshot=0.99 negative-zero=true detached=true
}

// ExampleTransformRange_workShift compresses only a validated operating shift
// inside a larger meter trace. Readings outside this half-open interval may
// belong to maintenance periods and are intentionally not inspected by PAA.
func ExampleTransformRange_workShift() {
	// Nonfinite sentinels represent unavailable *out-of-shift* data, not
	// missing values inside a supposedly valid sampled production window.
	flowLitersPerMin := []float64{
		math.NaN(), math.NaN(), 0, 0, 0,
		52, 54, 55, 56, 56, 57, 59,
		60, 66, 70, 82, 88, 84, 79,
		72, 68, 65, 63, 60, 58, 57,
		0, 0, math.NaN(), math.NaN(),
	}
	const shiftStart = 5
	const shiftEnd = 26
	const groups = 5

	// The selected view contains 21 finite source intervals and no copied
	// observations. The returned coordinates deliberately restart at zero.
	result, err := paa.TransformRange(flowLitersPerMin, shiftStart, shiftEnd, groups)
	if err != nil {
		fmt.Println("shift:", err)
		return
	}

	// Locate the busiest averaged shift interval. For global display, the
	// caller retains shiftStart as a separate absolute sample-index offset.
	peak := 0
	for j := 1; j < len(result.Values); j++ {
		if result.Values[j] > result.Values[peak] {
			peak = j
		}
	}
	cell := result.Segments[peak]
	fmt.Printf("shift=[%d,%d) samples=%d peak-bin=%d avg=%.1fL/min\n",
		shiftStart, shiftEnd, result.OriginalLength, peak, result.Values[peak])
	fmt.Printf("local-peak=[%d/%d,%d/%d)\n",
		cell.LeftNumerator, cell.Denominator,
		cell.RightNumerator, cell.Denominator)
	// Output:
	// shift=[5,26) samples=21 peak-bin=2 avg=78.6L/min
	// local-peak=[42/5,63/5)
}

// ExampleMeans_compactFeatures compares two equal-length, equally sampled
// vibration signatures using a PAA-derived Euclidean lower bound. The value
// projection is suitable because the consumer already knows n and w; a
// geometry-dependent symbolic index should retain full paa.Result instead.
func ExampleMeans_compactFeatures() {
	// Seventeen synchronized 10-ms samples from a healthy bearing and a
	// current bearing. The current trace has a repeatable mid-window deviation.
	baseline := []float64{
		0.2, 0.3, 0.5, 0.8, 1.0, 1.1, 1.0, 0.8, 0.5,
		0.3, 0.2, 0.4, 0.7, 0.9, 1.0, 0.8, 0.5,
	}
	current := []float64{
		0.2, 0.3, 0.5, 0.8, 1.0, 1.1, 1.0, 0.8, 1.9,
		2.3, 2.2, 2.4, 2.2, 0.9, 1.0, 0.8, 0.5,
	}
	const groups = 5
	const earlyReviewLimit = 2.0

	// Means projects only the aggregated numeric values of the canonical
	// result; both traces share the same explicit n and w by construction.
	baselinePAA, err := paa.Means(baseline, groups)
	if err != nil {
		fmt.Println("baseline:", err)
		return
	}
	currentPAA, err := paa.Means(current, groups)
	if err != nil {
		fmt.Println("current:", err)
		return
	}

	// Jensen's inequality on common equal-duration intervals yields
	// sqrt(n/w * sum_j (meanA_j-meanB_j)^2) <= ||A-B||_2, in real arithmetic.
	// This is a Euclidean bound in exact arithmetic, NOT an unrestricted DTW
	// bound or a certified float64 pruning test (which needs rounding guards).
	var reducedSquared, directSquared float64
	for j := range baselinePAA {
		delta := currentPAA[j] - baselinePAA[j]
		reducedSquared += delta * delta
	}
	for i := range baseline {
		delta := current[i] - baseline[i]
		directSquared += delta * delta
	}
	bound := math.Sqrt(float64(len(baseline)) / groups * reducedSquared)
	direct := math.Sqrt(directSquared)
	fmt.Printf("euclidean=%.2f paa-bound=%.2f observed-order=%t review=%t\n",
		direct, bound, bound <= direct, bound >= earlyReviewLimit)
	// Output:
	// euclidean=4.03 paa-bound=3.45 observed-order=true review=true
}
