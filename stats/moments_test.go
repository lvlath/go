// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats_test validates Welford/Chan behavior through the public API.
// The rational oracle, state copies, and counter-overflow fixtures are external.
// AI-Hints: comparison tolerances must not mask zeroed offset-scale variance.
package stats_test

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/stats"
)

// TestMoments_LargeOffsetSmallVariance verifies the exact canonical offset case.
// A subtraction-of-squares implementation returns zero here, whereas a
// properly centered algorithm returns the sample variance of exactly one.
func TestMoments_LargeOffsetSmallVariance(t *testing.T) {
	values := []float64{1e12 + 1, 1e12 + 2, 1e12 + 3}
	summary, err := stats.Summarize(values, 1)
	if err != nil {
		t.Fatal(err)
	}
	if summary.Count != 3 || summary.Mean != 1e12+2 || summary.Variance != 1 ||
		summary.Min != values[0] || summary.Max != values[2] {
		t.Fatalf("unstable large-offset result: %+v", summary)
	}

	// A second offset tests sub-ULP mean state when the mean is only
	// representable as a compensated pair of float64 values.
	values = []float64{1e12 + 1, 1e12 + 2}
	got, err := stats.Summarize(values, 0)
	if err != nil {
		t.Fatal(err)
	}
	mean, variance := oracleRatSummary(values, 0)
	requireNear(t, got.Mean, mean)
	requireNear(t, got.Variance, variance)
}

// TestMoments_ExtremeFloat64Ranges tests full-range scaled M2 retention.
// Both same-sign huge observations and opposite-sign huge observations must
// finish without premature overflow when the requested result is representable.
func TestMoments_ExtremeFloat64Ranges(t *testing.T) {
	cases := []struct {
		name   string
		values []float64
		ddof   int
	}{
		{name: "identical_huge", values: []float64{1e308, 1e308}, ddof: 0},
		{name: "sample_extreme", values: []float64{-1e154, 0, 1e154}, ddof: 1},
		{name: "population_extreme", values: []float64{-1e154, 0, 1e154}, ddof: 0},
		{name: "zero_and_tiny", values: []float64{0, math.SmallestNonzeroFloat64}, ddof: 0},
		{name: "alternating_signs", values: []float64{-1e100, 1e100, -1e100, 1e100}, ddof: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			summary, err := stats.Summarize(tc.values, tc.ddof)
			if err != nil {
				t.Fatal(err)
			}
			wantMean, wantVariance := oracleRatSummary(tc.values, tc.ddof)
			requireNear(t, summary.Mean, wantMean)
			requireNear(t, summary.Variance, wantVariance)
			if summary.Count != uint64(len(tc.values)) {
				t.Fatalf("Count %d, want %d", summary.Count, len(tc.values))
			}
		})
	}
}

// TestMoments_MergeAtomicOverflow constructs enormous counts using only Merge.
// Doubling external accumulators 63 times reaches Count=2^63, so combining
// two such states must reject uint64 overflow without any private fixture.
func TestMoments_MergeAtomicOverflow(t *testing.T) {
	power := stats.NewMoments()
	if err := power.Update(7); err != nil {
		t.Fatal(err)
	}

	// This loop scales count exponentially while remaining O(log MaxUint64).
	for i := 0; i < 63; i++ {
		snapshot := cloneMoments(t, power)
		if err := power.Merge(snapshot); err != nil {
			t.Fatal(err)
		}
	}
	if summary, err := power.Summary(0); err != nil || summary.Count != (uint64(1)<<63) {
		t.Fatalf("failed to construct 2^63 count: %+v, %v", summary, err)
	}

	right := cloneMoments(t, power)
	leftBefore, _ := power.Summary(0)
	rightBefore, _ := right.Summary(0)
	mustErrorIs(t, power.Merge(right), stats.ErrOverflow)
	leftAfter, _ := power.Summary(0)
	rightAfter, _ := right.Summary(0)
	requireSummaryEqual(t, leftAfter, leftBefore)
	requireSummaryEqual(t, rightAfter, rightBefore)

	// Follow-up valid updates show a failed merge did not poison a private
	// compensation field that happens to be invisible in the snapshot.
	if err := power.Update(9); err != nil {
		t.Fatal(err)
	}
	continued, err := power.Summary(0)
	if err != nil || continued.Count != (uint64(1)<<63)+1 || continued.Min != 7 || continued.Max != 9 {
		t.Fatalf("bad continuation after overflow: %+v, %v", continued, err)
	}
}

// TestMoments_UpdateCountOverflow constructs MaxUint64 without test-only APIs.
// The binary-decomposition merge sequence sums 1+2+...+2^63, then proves
// that a rejected Update leaves the entire observable accumulator unchanged.
func TestMoments_UpdateCountOverflow(t *testing.T) {
	power := stats.NewMoments()
	if err := power.Update(11); err != nil {
		t.Fatal(err)
	}
	full := stats.NewMoments()
	for bit := 0; bit < 64; bit++ {
		if err := full.Merge(power); err != nil {
			t.Fatal(err)
		}
		if bit != 63 {
			if err := power.Merge(cloneMoments(t, power)); err != nil {
				t.Fatal(err)
			}
		}
	}
	before, err := full.Summary(0)
	if err != nil || before.Count != math.MaxUint64 {
		t.Fatalf("failed to construct MaxUint64 count: %+v, %v", before, err)
	}
	mustErrorIs(t, full.Update(13), stats.ErrOverflow)
	after, err := full.Summary(0)
	if err != nil {
		t.Fatal(err)
	}
	requireSummaryEqual(t, after, before)
}

// TestMoments_MergeEmptyAndOwnership protects both merge identity laws.
// Taking a nonempty source into an empty destination must not alias its state;
// a subsequent source update cannot change the destination's results.
func TestMoments_MergeEmptyAndOwnership(t *testing.T) {
	source := stats.NewMoments()
	for _, value := range []float64{3, 7, 11} {
		if err := source.Update(value); err != nil {
			t.Fatal(err)
		}
	}
	blank := stats.NewMoments()
	before, _ := source.Summary(1)
	if err := source.Merge(blank); err != nil {
		t.Fatal(err)
	}
	after, _ := source.Summary(1)
	requireSummaryEqual(t, after, before)

	if err := blank.Merge(source); err != nil {
		t.Fatal(err)
	}
	adopted, _ := blank.Summary(1)
	requireSummaryEqual(t, adopted, before)
	if err := source.Update(100); err != nil {
		t.Fatal(err)
	}
	stayed, _ := blank.Summary(1)
	requireSummaryEqual(t, stayed, before)
}

// TestSummary_DDOFAndSingleton checks denominator and zero-value semantics.
// Population singleton variance is zero; sample singleton and empty state
// have no positive denominator and must return ErrInsufficientSamples.
func TestSummary_DDOFAndSingleton(t *testing.T) {
	values := []float64{1, 2, 3}
	pop, err := stats.SummarizePopulation(values)
	if err != nil {
		t.Fatal(err)
	}
	sample, err := stats.SummarizeSample(values)
	if err != nil {
		t.Fatal(err)
	}
	requireNear(t, pop.Variance, 2.0/3.0)
	if sample.Variance != 1 || sample.Mean != 2 || sample.Count != 3 {
		t.Fatalf("invalid sample summary: %+v", sample)
	}

	single, err := stats.Summarize([]float64{4}, 0)
	if err != nil || single.Variance != 0 || single.Mean != 4 {
		t.Fatalf("bad singleton: %+v, %v", single, err)
	}
	_, err = stats.Summarize([]float64{4}, 1)
	mustErrorIs(t, err, stats.ErrInsufficientSamples)
	var empty stats.Moments
	_, err = empty.Summary(0)
	mustErrorIs(t, err, stats.ErrInsufficientSamples)
}

// TestSummary_FinalOverflowAndUnderflow distinguishes materialization domains.
// An extreme M2 may persist successfully even when the requested variance
// cannot be returned in finite float64; tiny positive M2 may round to zero.
func TestSummary_FinalOverflowAndUnderflow(t *testing.T) {
	m := stats.NewMoments()
	for _, value := range []float64{-math.MaxFloat64, math.MaxFloat64} {
		if err := m.Update(value); err != nil {
			t.Fatal(err)
		}
	}
	for _, ddof := range []int{0, 1} {
		result, err := m.Summary(ddof)
		if result != nil {
			t.Fatalf("overflow exposed partial summary: %+v", result)
		}
		mustErrorIs(t, err, stats.ErrOverflow)
	}
	minSub := math.SmallestNonzeroFloat64
	underflow, err := stats.Summarize([]float64{-minSub, minSub}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if underflow.Variance != 0 || underflow.Min != -minSub || underflow.Max != minSub {
		t.Fatalf("incorrect rounded tiny variance: %+v", underflow)
	}
}

// TestSummarize_WrappersAndErrorPrecedence verifies the facade's error law.
// Invalid options outrank nil slices; nil slices differ from non-nil empty
// inputs, and neither path may produce a partial summary object.
func TestSummarize_WrappersAndErrorPrecedence(t *testing.T) {
	cases := []struct {
		name   string
		values []float64
		ddof   int
		want   error
	}{
		{"invalid_option_before_nil", nil, -1, stats.ErrInvalidOption},
		{"invalid_option_large", []float64{1}, 2, stats.ErrInvalidOption},
		{"nil_input", nil, 0, stats.ErrNilInput},
		{"empty_non_nil", []float64{}, 0, stats.ErrEmptyInput},
		{"nan", []float64{1, math.NaN()}, 0, stats.ErrNonFinite},
		{"positive_inf", []float64{math.Inf(1)}, 0, stats.ErrNonFinite},
		{"negative_inf", []float64{math.Inf(-1)}, 0, stats.ErrNonFinite},
		{"insufficient", []float64{1}, 1, stats.ErrInsufficientSamples},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			summary, err := stats.Summarize(tc.values, tc.ddof)
			mustErrorIs(t, err, tc.want)
			mustNilState(t, summary, true, tc.name)
		})
	}

	// Wrappers must be semantically equivalent to their canonical call.
	values := []float64{2, 3, 5}
	p, err := stats.SummarizePopulation(values)
	if err != nil {
		t.Fatal(err)
	}
	pCanonical, _ := stats.Summarize(values, 0)
	requireSummaryEqual(t, p, pCanonical)
	s, err := stats.SummarizeSample(values)
	if err != nil {
		t.Fatal(err)
	}
	sCanonical, _ := stats.Summarize(values, 1)
	requireSummaryEqual(t, s, sCanonical)
}

// TestMoments_MergeOracleAndGrouping compares independent partition trees.
// This validates the Chan cross term against a rational two-pass reference
// without assuming bitwise association for different floating merge orders.
func TestMoments_MergeOracleAndGrouping(t *testing.T) {
	values := []float64{1e12 + 1, 1e12 + 4, 1e12 + 2, 1e12 + 5, 1e12 + 6, 1e12 + 3}
	wantMean, wantVariance := oracleRatSummary(values, 1)

	// Left-deep merge is a fixed partition sequence.
	left := stats.NewMoments()
	for _, value := range values {
		part := stats.NewMoments()
		if err := part.Update(value); err != nil {
			t.Fatal(err)
		}
		if err := left.Merge(part); err != nil {
			t.Fatal(err)
		}
	}

	// Balanced grouping merges three independently reduced pairs.
	parts := make([]*stats.Moments, 3)
	for i := range parts {
		parts[i] = stats.NewMoments()
		for _, value := range values[2*i : 2*i+2] {
			if err := parts[i].Update(value); err != nil {
				t.Fatal(err)
			}
		}
	}
	balanced := cloneMoments(t, parts[0])
	if err := balanced.Merge(parts[1]); err != nil {
		t.Fatal(err)
	}
	if err := balanced.Merge(parts[2]); err != nil {
		t.Fatal(err)
	}

	// Reverse grouping must still agree within the numerical budget.
	reverse := stats.NewMoments()
	for i := len(values) - 1; i >= 0; i-- {
		if err := reverse.Update(values[i]); err != nil {
			t.Fatal(err)
		}
	}
	for _, accumulator := range []*stats.Moments{left, balanced, reverse} {
		got, err := accumulator.Summary(1)
		if err != nil {
			t.Fatal(err)
		}
		requireNear(t, got.Mean, wantMean)
		requireNear(t, got.Variance, wantVariance)
		if got.Count != uint64(len(values)) || got.Min != values[0] || got.Max != values[4] {
			t.Fatalf("partition metadata mismatch: %+v", got)
		}
	}
}

// TestMoments_DeterministicRationalStress covers scaled offsets and input orders.
// Each case is built from finite binary float64 values, then checked against
// the mathematically exact big.Rat mean/variance computed independently.
func TestMoments_DeterministicRationalStress(t *testing.T) {
	rng := rand.New(rand.NewSource(20261109))
	for caseID := 0; caseID < 150; caseID++ {
		length := 3 + rng.Intn(45)
		power := rng.Intn(1950) - 1000
		base := math.Ldexp(1, power)
		step := math.Nextafter(base, math.Inf(1)) - base
		values := make([]float64, length)
		for i := range values {
			jitter := float64(rng.Intn(11) - 5)
			values[i] = base + jitter*step
		}
		for _, ddof := range []int{0, 1} {
			wantMean, wantVariance := oracleRatSummary(values, ddof)
			got, err := stats.Summarize(values, ddof)
			if math.IsInf(wantVariance, 0) || math.IsInf(wantMean, 0) {
				// The independent oracle proves that the requested float64 result
				// cannot be materialized; the only correct outcome is ErrOverflow.
				mustErrorIs(t, err, stats.ErrOverflow)
				if got != nil {
					t.Fatalf("case %d: overflow returned partial %+v", caseID, got)
				}
				continue
			}
			if err != nil {
				t.Fatalf("case %d ddof %d: %v", caseID, ddof, err)
			}
			requireNear(t, got.Mean, wantMean)
			requireNear(t, got.Variance, wantVariance)
		}
	}
}

// TestMoments_NilReceiverAndSelfMerge makes error priorities explicit.
// A nil receiver never dereferences, and even a zero-valued accumulator is
// forbidden to merge with itself; neither rejection mutates state.
func TestMoments_NilReceiverAndSelfMerge(t *testing.T) {
	var nilMoments *stats.Moments
	mustErrorIs(t, nilMoments.Update(math.NaN()), stats.ErrNilInput)
	mustErrorIs(t, nilMoments.Merge(nil), stats.ErrNilInput)
	_, err := nilMoments.Summary(-1)
	mustErrorIs(t, err, stats.ErrInvalidOption)
	_, err = nilMoments.Summary(0)
	mustErrorIs(t, err, stats.ErrNilInput)

	var empty stats.Moments
	mustErrorIs(t, empty.Merge(&empty), stats.ErrInvalidOption)
	mustErrorIs(t, empty.Merge(nil), stats.ErrNilInput)
	if err = empty.Merge(stats.NewMoments()); err != nil {
		t.Fatal(err)
	}
	_, err = empty.Summary(0)
	mustErrorIs(t, err, stats.ErrInsufficientSamples)
}

// TestMoments_SourceInvalidInputDoesNotMutate snapshots a live accumulator.
// NaN and both infinities must always return ErrNonFinite and leave the
// complete externally observable state intact before future valid updates.
func TestMoments_SourceInvalidInputDoesNotMutate(t *testing.T) {
	m := stats.NewMoments()
	for _, value := range []float64{5, 7} {
		if err := m.Update(value); err != nil {
			t.Fatal(err)
		}
	}
	before, _ := m.Summary(0)
	for _, invalid := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		mustErrorIs(t, m.Update(invalid), stats.ErrNonFinite)
		after, err := m.Summary(0)
		if err != nil {
			t.Fatal(err)
		}
		requireSummaryEqual(t, after, before)
	}
	mustErrorIs(t, m.Merge(m), stats.ErrInvalidOption)
	mustErrorIs(t, m.Merge(nil), stats.ErrNilInput)
	after, _ := m.Summary(0)
	requireSummaryEqual(t, after, before)
	if err := m.Update(9); err != nil {
		t.Fatal(err)
	}
	continued, _ := m.Summary(1)
	if continued.Count != 3 || continued.Mean != 7 || continued.Variance != 4 {
		t.Fatalf("bad continuation after errors: %+v", continued)
	}
}

// TestMoments_ExternalSentinels checks errors.Is independently of message text.
// This prevents accidentally comparing error strings or requiring internal
// implementation-specific wrapping formats in consumer code.
func TestMoments_ExternalSentinels(t *testing.T) {
	m := stats.NewMoments()
	if err := m.Update(math.NaN()); !errors.Is(err, stats.ErrNonFinite) {
		t.Fatalf("nonfinite sentinel identity: %v", err)
	}
}

// TestMoments_MillionPointOffsetCompensation checks cumulative roundoff drift.
// One million alternating observations around a huge offset have exactly
// mean=offset+2 and population variance=1, independent of sample count.
func TestMoments_MillionPointOffsetCompensation(t *testing.T) {
	const pairs = 500_000
	const baseline = 1e12
	m := stats.NewMoments()
	for pair := 0; pair < pairs; pair++ {
		if err := m.Update(baseline + 1); err != nil {
			t.Fatal(err)
		}
		if err := m.Update(baseline + 3); err != nil {
			t.Fatal(err)
		}
	}
	summary, err := m.Summary(0)
	if err != nil {
		t.Fatal(err)
	}
	if summary.Count != pairs*2 || summary.Mean != baseline+2 {
		t.Fatalf("long-stream count/center drifted: %+v", summary)
	}
	requireNear(t, summary.Variance, 1)
}

// TestMoments_LargeCountMergePreservesSmallSpread checks count precision.
// Repeated partition duplication grows a two-point population to 2^63
// observations without changing its mean or population variance.
func TestMoments_LargeCountMergePreservesSmallSpread(t *testing.T) {
	m := stats.NewMoments()
	for _, value := range []float64{1e12 + 1, 1e12 + 3} {
		if err := m.Update(value); err != nil {
			t.Fatal(err)
		}
	}
	for bit := 1; bit < 63; bit++ {
		if err := m.Merge(cloneMoments(t, m)); err != nil {
			t.Fatal(err)
		}
	}
	summary, err := m.Summary(0)
	if err != nil {
		t.Fatal(err)
	}
	if summary.Count != uint64(1)<<63 || summary.Mean != 1e12+2 {
		t.Fatalf("huge-count mean/count mismatch: %+v", summary)
	}
	requireNear(t, summary.Variance, 1)
}

// TestMoments_ExponentGapCancellationRetainsMean is a fixed regression anchor.
// The intermediate double-double mean cannot represent a contribution of 1
// between two 1e154 terms, but the final exact binary sum must preserve it.
func TestMoments_ExponentGapCancellationRetainsMean(t *testing.T) {
	values := []float64{1e154, 1, -1e154}
	wantMean, wantVariance := oracleRatSummary(values, 1)

	sequential, err := stats.Summarize(values, 1)
	if err != nil {
		t.Fatal(err)
	}
	if sequential.Mean != wantMean {
		t.Fatalf("exponent cancellation lost the representable mean: got %.17g, want %.17g", sequential.Mean, wantMean)
	}
	requireNear(t, sequential.Variance, wantVariance)

	// The same exact sum must survive a Chan merge of different partitions.
	partA, partB := stats.NewMoments(), stats.NewMoments()
	for _, value := range values[:2] {
		if err = partA.Update(value); err != nil {
			t.Fatal(err)
		}
	}
	if err = partB.Update(values[2]); err != nil {
		t.Fatal(err)
	}
	if err = partA.Merge(partB); err != nil {
		t.Fatal(err)
	}
	merged, err := partA.Summary(1)
	if err != nil {
		t.Fatal(err)
	}
	if merged.Mean != wantMean {
		t.Fatalf("merged exponent cancellation lost mean: %+v", merged)
	}
	requireNear(t, merged.Variance, wantVariance)
}

// TestMoments_ExponentGapCancellationRetainsSubnormals covers the smallest input.
// With two huge opposite observations, the remaining sum of subnormals is
// required to round to exactly one minSub; no exponent-gap cutoff is allowed.
func TestMoments_ExponentGapCancellationRetainsSubnormals(t *testing.T) {
	tiny := math.SmallestNonzeroFloat64
	values := []float64{1e154, tiny, -1e154, 2 * tiny}
	wantMean, wantVariance := oracleRatSummary(values, 0)
	got, err := stats.Summarize(values, 0)
	if err != nil {
		t.Fatal(err)
	}
	if wantMean != tiny || got.Mean != wantMean {
		t.Fatalf("subnormal mean lost: got %.17g, exact oracle %.17g", got.Mean, wantMean)
	}
	requireNear(t, got.Variance, wantVariance)
}

// TestMoments_MaxCountExactSumWidth validates the fixed 2176-bit bound.
// A count of MaxUint64 observations all equal to MaxFloat64 has a huge
// unnormalized exact sum but must still publish the original finite mean.
func TestMoments_MaxCountExactSumWidth(t *testing.T) {
	for _, value := range []float64{math.MaxFloat64, -math.MaxFloat64} {
		t.Run(fmt.Sprintf("sign_%.0f", math.Copysign(1, value)), func(t *testing.T) {
			power := stats.NewMoments()
			if err := power.Update(value); err != nil {
				t.Fatal(err)
			}
			full := stats.NewMoments()
			for bit := 0; bit < 64; bit++ {
				if err := full.Merge(power); err != nil {
					t.Fatal(err)
				}
				if bit != 63 {
					if err := power.Merge(cloneMoments(t, power)); err != nil {
						t.Fatal(err)
					}
				}
			}
			got, err := full.Summary(0)
			if err != nil {
				t.Fatal(err)
			}
			if got.Count != math.MaxUint64 || got.Mean != value || got.Variance != 0 ||
				got.Min != value || got.Max != value {
				t.Fatalf("fixed-point overflow at extreme count: %+v", got)
			}
		})
	}
}

// TestMoments_FullFloatExponentRationalOracle validates arbitrary binary inputs.
// Randomized but fixed IEEE-754 exponent/mantissa patterns exercise mixed signs,
// subnormals, huge dynamic range, and final overflow against an exact oracle.
func TestMoments_FullFloatExponentRationalOracle(t *testing.T) {
	const exponentMask = uint64(0x7ff)
	const fractionMask = uint64(1<<52) - 1
	const exponentShift = 52
	const signShift = 63

	rng := rand.New(rand.NewSource(7001979))
	for caseID := 0; caseID < 250; caseID++ {
		length := 2 + rng.Intn(8)
		values := make([]float64, length)
		for index := range values {
			// Exclude the all-ones exponent to generate only finite inputs;
			// exponent zero remains valid and deliberately includes subnormals.
			exponent := rng.Uint64() % exponentMask
			fraction := rng.Uint64() & fractionMask
			sign := rng.Uint64() & 1
			values[index] = math.Float64frombits((sign << signShift) | (exponent << exponentShift) | fraction)
		}

		for _, ddof := range []int{0, 1} {
			wantMean, wantVariance := oracleRatSummary(values, ddof)
			got, err := stats.Summarize(values, ddof)
			if math.IsInf(wantVariance, 0) {
				mustErrorIs(t, err, stats.ErrOverflow)
				mustNilState(t, got, true, "full-range overflow")
				continue
			}
			if err != nil {
				t.Fatalf("case %d ddof %d: unexpected error %v", caseID, ddof, err)
			}
			// The exact binary numerator must round identically to the independent
			// rational oracle, not merely satisfy an approximate tolerance.
			if got.Mean != wantMean {
				t.Fatalf("case %d ddof %d: mean %.17g, exact oracle %.17g", caseID, ddof, got.Mean, wantMean)
			}
			requireNear(t, got.Variance, wantVariance)
		}
	}
}
