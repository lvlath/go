// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats_test compares scaled central moments with independent rationals.
// The tests include huge multiplicities and deterministic adversarial inputs.
package stats_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/stats"
)

const (
	// varianceRelativeErrorBudget bounds the observed relative discrepancy of
	// ordinary finite rounded variances against exact binary rational results.
	// This is an oracle acceptance tolerance, NOT a zero-variance threshold.
	varianceRelativeErrorBudget = 2e-12

	// varianceSubnormalULPBudget permits at most two float64 representable steps
	// when the true rounded variance is too small for a relative metric.
	varianceSubnormalULPBudget = uint64(2)

	// randomPrecisionSeed fixes the adversarial input generator across test runs.
	randomPrecisionSeed = int64(20261109)
)

// varianceErrorMetrics compares two nonnegative finite float64 variances.
// It separates relative error from ULP distance so subnormal rounding is
// evaluated without dividing by a zero or nearly zero expected variance.
//
// Implementation:
//   - Stage 1: calculate the absolute difference at the public float64 boundary.
//   - Stage 2: calculate relative error only for a positive reference variance.
//   - Stage 3: compare monotonic positive IEEE-754 bit patterns for ULP distance.
//
// Behavior highlights:
//   - Reference zero has zero-relative error only when observed zero is exact.
//
// Inputs:
//   - actual, expected: finite nonnegative variances with no caller aliasing.
//
// Returns:
//   - relative: |actual-expected|/expected when expected>0.
//   - ulps: difference between adjacent representable nonnegative float64 values.
//
// Errors:
//   - None; the caller has checked the output domain and finiteness.
//
// Determinism:
//   - Exact bit-pattern comparison for ULP distance.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Relative error is meaningful for normal nonzero results; ULPs for subnormals.
//
// AI-Hints:
//   - Do not use an arbitrary zero threshold for constant-column classification.
//   - This helper measures error only and is not called by production kernels.
func varianceErrorMetrics(actual, expected float64) (float64, uint64) {
	var ulps uint64
	actualBits := math.Float64bits(actual)
	expectedBits := math.Float64bits(expected)
	if actualBits > expectedBits {
		ulps = actualBits - expectedBits
	} else {
		ulps = expectedBits - actualBits
	}

	if expected == 0 {
		if actual == 0 {
			return 0, ulps
		}
		return math.Inf(1), ulps
	}
	return math.Abs(actual-expected) / expected, ulps
}

// requireVarianceOracle verifies a finite variance using two numeric regimes.
// Normal results use a scale-relative budget; subnormal and zero results use
// a fixed ULP distance, preventing an erroneous forced-zero shortcut.
//
// Implementation:
//   - Stage 1: reject negative and nonfinite production variance immediately.
//   - Stage 2: compute relative and ULP discrepancies independently.
//   - Stage 3: select subnormal-ULP or normal-relative acceptance rules.
//
// Behavior highlights:
//   - Checks remain anchored in exact rational oracle outputs.
//
// Inputs:
//   - t: owning test; actual, expected: floating variances; caseName: diagnostic.
//
// Returns:
//   - relative: observed finite relative discrepancy for aggregate reporting.
//   - ulps: observed representable float64 distance.
//
// Errors:
//   - Fails the owning test when the numerical error violates the declared budget.
//
// Determinism:
//   - Exact fixed threshold selection and IEEE-754 comparison order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Relative and ULP budgets quantify floating arithmetic, not domain constants.
//
// AI-Hints:
//   - Do not relax the budget to make a defective recurrence pass.
//   - Failures must be investigated against the rational oracle first.
func requireVarianceOracle(t *testing.T, actual, expected float64, caseName string) (float64, uint64) {
	t.Helper()
	if math.IsNaN(actual) || math.IsInf(actual, 0) || actual < 0 {
		t.Fatalf("%s: invalid production variance %g", caseName, actual)
	}
	if math.IsNaN(expected) || math.IsInf(expected, 0) || expected < 0 {
		t.Fatalf("%s: invalid oracle fixture variance %g", caseName, expected)
	}

	relative, ulps := varianceErrorMetrics(actual, expected)
	if expected < math.SmallestNonzeroFloat64*float64(1<<52) {
		if ulps > varianceSubnormalULPBudget {
			t.Fatalf("%s: subnormal discrepancy: actual=%g expected=%g ulps=%d", caseName, actual, expected, ulps)
		}
		return relative, ulps
	}
	if relative > varianceRelativeErrorBudget {
		t.Fatalf("%s: variance discrepancy: actual=%g expected=%g relative=%g ulps=%d", caseName, actual, expected, relative, ulps)
	}
	return relative, ulps
}

// TestMoments_M2AdversarialRationalOracle validates the one-pass Welford law.
// It tests adjacent float64 values, opposing exponents, cancellation, and
// subnormal underflow using exact binary-input rational mean and variance.
//
// Implementation:
//   - Stage 1: generate fixed adversarial families and reproducible random inputs.
//   - Stage 2: compute the rational oracle independently of production algorithms.
//   - Stage 3: verify errors, exact mean, finite M2-derived variance, and extrema.
//
// Behavior highlights:
//   - A mathematically unrepresentable variance must yield ErrOverflow, not Inf.
//   - A small positive variance may round to zero without being constant.
//
// Inputs:
//   - t: owning test; all float inputs are generated deterministically.
//
// Returns:
//   - No value; logs the worst observed finite relative variance discrepancy.
//
// Errors:
//   - ErrOverflow must classify a non-finite oracle result at materialization.
//
// Determinism:
//   - Fixed random seed, cases in increasing index order and fixed ddof choice.
//
// Complexity:
//   - O(total observations) production work and oracle big.Rat operations.
//
// Notes:
//   - The reported maximum is empirical and is not a universal proof of accuracy.
//
// AI-Hints:
//   - Never compare independent grouping results bitwise; Welford is not associative.
//   - The oracle must not call production Moments to compute expected moments.
func TestMoments_M2AdversarialRationalOracle(t *testing.T) {
	cases := []struct {
		name   string
		values []float64
		ddof   int
	}{
		{"large_offset", []float64{1e12 + 1, 1e12 + 2, 1e12 + 3}, 1},
		{"wide_exponent_cancellation", []float64{1e154, 1, -1e154}, 0},
		{"subnormal_at_zero", []float64{0, math.SmallestNonzeroFloat64}, 0},
		{"huge_identical", []float64{1e308, 1e308}, 1},
		{"opposite_float64_extremes", []float64{math.MaxFloat64, -math.MaxFloat64}, 0},
		{"subnormal_variance_edge", []float64{-math.Ldexp(1, -537), 0, math.Ldexp(1, -537)}, 1},
	}

	// Stage 1: neighboring binary float64 inputs systematically expose
	// cancellation that would be invisible to decimal-value test fixtures.
	for exponent := -1000; exponent < 1020; exponent += 23 {
		base := math.Ldexp(1, exponent)
		next := math.Nextafter(base, math.Inf(1))
		cases = append(cases, struct {
			name   string
			values []float64
			ddof   int
		}{fmt.Sprintf("adjacent_exponent_%d", exponent), []float64{base, next, base, next, base}, 1})
	}

	// A fixed PRNG explores independent large offsets and wide-ranging signs.
	rng := rand.New(rand.NewSource(randomPrecisionSeed))
	for caseID := 0; caseID < 1000; caseID++ {
		count := 2 + rng.Intn(48)
		values := make([]float64, count)
		exponent := rng.Intn(1950) - 1000
		base := math.Ldexp(1, exponent)
		for index := range values {
			switch caseID % 4 {
			case 0:
				values[index] = math.Float64frombits(math.Float64bits(base) + uint64(rng.Intn(40)))
			case 1:
				values[index] = math.Ldexp(float64(rng.Intn(41)-20), rng.Intn(1900)-950)
			case 2:
				values[index] = base
				if index%3 == 1 {
					values[index] = -base
				}
			case 3:
				values[index] = math.Ldexp(0.5+rng.Float64(), rng.Intn(1900)-950)
				if index%2 == 0 {
					values[index] = -values[index]
				}
			}
		}
		cases = append(cases, struct {
			name   string
			values []float64
			ddof   int
		}{fmt.Sprintf("random_%d", caseID), values, caseID % 2})
	}

	// Stage 2 and 3: exact means must match after a single rounding; centered
	// variances may differ within the explicit compensated-arithmetic budget.
	maxRelative := 0.0
	worstCase := ""
	countOverflow := 0
	for _, testCase := range cases {
		wantMean, wantVariance := oracleRatSummary(testCase.values, testCase.ddof)
		got, err := stats.Summarize(testCase.values, testCase.ddof)
		if math.IsInf(wantVariance, 0) {
			mustErrorIs(t, err, stats.ErrOverflow)
			mustNilState(t, got, true, testCase.name)
			countOverflow++
			continue
		}
		if err != nil {
			t.Fatalf("%s: unexpected error %v", testCase.name, err)
		}
		if got.Mean != wantMean {
			t.Fatalf("%s: incorrectly rounded mean: got %.17g want %.17g", testCase.name, got.Mean, wantMean)
		}
		if got.Count != uint64(len(testCase.values)) {
			t.Fatalf("%s: incorrect Count %d", testCase.name, got.Count)
		}
		// Extrema refer to original observations and are exact order-invariant
		// values, unlike floating central-moment arithmetic.
		expectedMin, expectedMax := testCase.values[0], testCase.values[0]
		for _, value := range testCase.values[1:] {
			if value < expectedMin {
				expectedMin = value
			}
			if value > expectedMax {
				expectedMax = value
			}
		}
		if got.Min != expectedMin || got.Max != expectedMax {
			t.Fatalf("%s: extrema got [%g,%g], want [%g,%g]", testCase.name, got.Min, got.Max, expectedMin, expectedMax)
		}
		relative, _ := requireVarianceOracle(t, got.Variance, wantVariance, testCase.name)
		if relative > maxRelative && !math.IsInf(relative, 0) {
			maxRelative = relative
			worstCase = testCase.name
		}
	}
	t.Logf("Welford oracle: cases=%d, overflow classifications=%d, max relative variance error=%g (%s)", len(cases), countOverflow, maxRelative, worstCase)
}

// TestMoments_M2WeightedChanOracle exercises enormous unnormalized moments.
// It constructs exact multiplicities with Merge, while an independent weighted
// rational oracle evaluates M2/(Count-ddof) without enumerating samples.
//
// Implementation:
//   - Stage 1: define extreme exact-count and reproducible partition fixtures.
//   - Stage 2: merge independently constructed populations in fixed order.
//   - Stage 3: compare final moments and error classifications to the oracle.
//
// Behavior highlights:
//   - M2 may exceed 1e308 by many orders while finite variance remains available.
//   - MaxUint64 and non-power-of-two counts are handled exactly before rounding.
//
// Inputs:
//   - t: owning test; all group values are finite and multiplicities positive.
//
// Returns:
//   - No value; logs the maximum relative discrepancy across finite partitions.
//
// Errors:
//   - ErrOverflow is expected only when final variance is unrepresentable.
//
// Determinism:
//   - Fixed group order, fixed random seed and binary population construction.
//
// Complexity:
//   - Time O(cases * groups * log MaxUint64), bounded oracle-big.Rat workspace.
//
// Notes:
//   - The test checks a real high-count numeric regime without private fixtures.
//
// AI-Hints:
//   - Do not calculate nA*nB as uint64 or materialize scaled M2 too early.
//   - Never interpret ErrOverflow for a representable final variance as success.
func TestMoments_M2WeightedChanOracle(t *testing.T) {
	const halfCount = uint64(1) << 63
	cases := []struct {
		name   string
		groups []weightedObservation
		ddof   int
	}{
		{"unrepresentable_m2_representable_variance", []weightedObservation{{-1e154, halfCount}, {1e154, halfCount - 1}}, 1},
		{"unequal_high_count_offset", []weightedObservation{{1e12 + 1, halfCount}, {1e12 + 3, halfCount - 1}}, 0},
		{"maxcount_identical_maxfloat", []weightedObservation{{math.MaxFloat64, halfCount}, {math.MaxFloat64, halfCount - 1}}, 0},
		{"huge_cancellation", []weightedObservation{{-math.MaxFloat64, halfCount}, {math.MaxFloat64, halfCount - 1}}, 0},
		{"subnormal_count", []weightedObservation{{-math.SmallestNonzeroFloat64, halfCount}, {math.SmallestNonzeroFloat64, halfCount - 1}}, 1},
	}

	// Stage 1: bounded multiplicities produce enormous Count without overflow.
	rng := rand.New(rand.NewSource(randomPrecisionSeed + 1))
	counts := []uint64{1, 3, 17, 1 << 31, 1 << 52, (1 << 62) - 1}
	for caseID := 0; caseID < 500; caseID++ {
		exponent := rng.Intn(1800) - 900
		base := math.Ldexp(1, exponent)
		value := base
		sibling := math.Nextafter(base, math.Inf(1))
		switch caseID % 4 {
		case 1:
			sibling = -base
		case 2:
			sibling = 0
		case 3:
			value = -base
		}
		cases = append(cases, struct {
			name   string
			groups []weightedObservation
			ddof   int
		}{
			name: fmt.Sprintf("weighted_%d", caseID),
			groups: []weightedObservation{
				{value, counts[rng.Intn(len(counts))]},
				{sibling, counts[rng.Intn(len(counts))]},
				{0, counts[rng.Intn(len(counts))]},
			},
			ddof: caseID % 2,
		})
	}

	// Stage 2: every group uses a valid public Update/Merge stream. None
	// exposes private compensated state through a test-only production API.
	maxRelative := 0.0
	worstCase := ""
	countOverflow := 0
	for _, testCase := range cases {
		merged := stats.NewMoments()
		var expectedCount uint64
		for _, group := range testCase.groups {
			partition := repeatedMoments(t, group.value, group.count)
			if err := merged.Merge(partition); err != nil {
				t.Fatalf("%s: unexpected Merge error: %v", testCase.name, err)
			}
			expectedCount += group.count
		}

		// Stage 3: the oracle materializes variance only after summing exact
		// weighted central squares; a large M2 is not an overflow by itself.
		wantMean, wantVariance := oracleWeightedRatSummary(testCase.groups, testCase.ddof)
		got, err := merged.Summary(testCase.ddof)
		if math.IsInf(wantVariance, 0) {
			mustErrorIs(t, err, stats.ErrOverflow)
			mustNilState(t, got, true, testCase.name)
			countOverflow++
			continue
		}
		if err != nil {
			t.Fatalf("%s: unexpected Summary error: %v", testCase.name, err)
		}
		if got.Count != expectedCount || got.Mean != wantMean {
			t.Fatalf("%s: Count/Mean mismatch: got %+v, exact mean %g count %d", testCase.name, got, wantMean, expectedCount)
		}
		expectedMin, expectedMax := testCase.groups[0].value, testCase.groups[0].value
		for _, group := range testCase.groups[1:] {
			if group.value < expectedMin {
				expectedMin = group.value
			}
			if group.value > expectedMax {
				expectedMax = group.value
			}
		}
		if got.Min != expectedMin || got.Max != expectedMax {
			t.Fatalf("%s: merged extrema got [%g,%g], want [%g,%g]", testCase.name, got.Min, got.Max, expectedMin, expectedMax)
		}
		relative, _ := requireVarianceOracle(t, got.Variance, wantVariance, testCase.name)
		if relative > maxRelative && !math.IsInf(relative, 0) {
			maxRelative = relative
			worstCase = testCase.name
		}
	}
	t.Logf("Chan weighted oracle: cases=%d, overflow classifications=%d, max relative variance error=%g (%s)", len(cases), countOverflow, maxRelative, worstCase)
}
