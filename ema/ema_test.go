// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema_test verifies EMA mathematical laws and batch boundary behavior.
// All numerical comparisons are exact binary64 comparisons against independent data.
package ema_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/ema"
)

// TestEMA_SMASortedGolden anchors the exact SMA readiness boundary on an increasing history.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Compare all four payloads and masks with independently specified integer means.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - P3, four observations; first two entries unavailable.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Never accept a valid value before the Pth observation.
func TestEMA_SMASortedGolden(t *testing.T) {
	result, err := ema.Compute(context.Background(), []float64{1, 2, 3, 4}, ema.Options{Period: 3, Seed: ema.SeedSMA})
	if err != nil {
		t.Fatal(err)
	}
	assertResult(t, result, []ema.Value{{}, {}, {Value: 2, Valid: true}, {Value: 3, Valid: true}})
}

// TestEMA_FirstSeedGolden anchors immediate first seeding and the subsequent half-weight recurrence.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Compare every output with an independently specified dyadic sequence.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - P3 First, four observations; all entries ready.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Do not substitute SMA warmup in the First wrapper.
func TestEMA_FirstSeedGolden(t *testing.T) {
	result, err := ema.Compute(context.Background(), []float64{1, 2, 3, 4}, ema.Options{Period: 3, Seed: ema.SeedFirst})
	if err != nil {
		t.Fatal(err)
	}
	assertResult(t, result, []ema.Value{{Value: 1, Valid: true}, {Value: 1.5, Valid: true}, {Value: 2.25, Valid: true}, {Value: 3.125, Valid: true}})
}

// TestEMA_PeriodOneIdentity protects direct input selection across huge, tiny, and signed-zero values.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Check both seed policies against an exact independently normalized identity.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - P1, extremes, smallest subnormals, both zero signs.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - The ABI normalizes negative zero; it does not erase nonzero subnormals.
func TestEMA_PeriodOneIdentity(t *testing.T) {
	x := []float64{1e308, 1, -1e308, 0, math.Copysign(0, -1), math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64}
	for _, seed := range []ema.Seed{ema.SeedSMA, ema.SeedFirst} {
		assertOracle(t, x, ema.Options{Period: 1, Seed: seed})
	}
}

// TestEMA_SeedCancellation protects a small residual between two enormous canceling seed terms.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Check the golden one-third value and an independent rational continuation.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - P3, canceling extreme prefixes.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Do not weaken the exact bit comparison to an offset-based tolerance.
func TestEMA_SeedCancellation(t *testing.T) {
	x := []float64{1e308, 1, -1e308}
	result, err := ema.Compute(context.Background(), x, ema.Options{Period: 3})
	if err != nil {
		t.Fatal(err)
	}
	assertResult(t, result, []ema.Value{{}, {}, {Value: 1.0 / 3, Valid: true}})
	assertOracle(t, []float64{math.MaxFloat64, 1, -math.MaxFloat64, 7, 0}, ema.Options{Period: 3})
}

// TestEMA_OppositeExtremes rejects false difference-overflow behavior at opposite finite extremes.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Require a valid positive zero as the second half-weight output.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - P3 First, MaxFloat64 and its negative.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - The represented final mean is zero even though a floating-point difference overflows.
func TestEMA_OppositeExtremes(t *testing.T) {
	x := []float64{math.MaxFloat64, -math.MaxFloat64}
	result, err := ema.Compute(context.Background(), x, ema.Options{Period: 3, Seed: ema.SeedFirst})
	if err != nil {
		t.Fatal(err)
	}
	assertResult(t, result, []ema.Value{{Value: math.MaxFloat64, Valid: true}, {Valid: true}})
}

// TestEMA_EmptyAndShortHistory distinguishes successful empty and unfinished histories from errors.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Check non-nil channels, their lengths, and each readiness bit.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Nil, allocated empty, and prefixes shorter than P3.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Empty input is not ErrNilInput in this API.
func TestEMA_EmptyAndShortHistory(t *testing.T) {
	for _, x := range [][]float64{nil, {}, {1}, {1, 2}} {
		for _, seed := range []ema.Seed{ema.SeedSMA, ema.SeedFirst} {
			opts := ema.Options{Period: 3, Seed: seed}
			result, err := ema.Compute(context.Background(), x, opts)
			if err != nil {
				t.Fatal(err)
			}
			assertResult(t, result, oracleHistory(x, opts))
		}
	}
}

// TestEMA_ValidationAndPrecedence anchors context, option, and finite-input error ordering.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Reject invalid constructors and batch calls using errors.Is and nil results.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Invalid periods/seeds, nil/canceled context, NaN and both infinities.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Do not infer error categories by parsing diagnostic text.
func TestEMA_ValidationAndPrecedence(t *testing.T) {
	ctx := context.Background()
	badOptions := []ema.Options{{}, {Period: -1}, {Period: math.MaxInt}, {Period: 3, Seed: ema.Seed(2)}, {Period: 3, Seed: ema.Seed(255)}}
	for _, opts := range badOptions {
		stream, err := ema.New(opts)
		if stream != nil || !errors.Is(err, ema.ErrInvalidOption) {
			t.Fatalf("New: %v %v", stream, err)
		}
		result, err := ema.Compute(ctx, []float64{math.NaN()}, opts)
		if result != nil || !errors.Is(err, ema.ErrInvalidOption) {
			t.Fatalf("options precedence: %v", err)
		}
	}
	result, err := ema.Compute(nil, []float64{math.NaN()}, ema.Options{})
	if result != nil || !errors.Is(err, ema.ErrNilContext) {
		t.Fatalf("nil context precedence: %v", err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	result, err = ema.Compute(canceled, nil, ema.Options{})
	if result != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation precedence: %v", err)
	}
	for _, x := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		result, err = ema.Compute(ctx, []float64{1, x, 3}, ema.Options{Period: 3})
		if result != nil || !errors.Is(err, ema.ErrNonFinite) {
			t.Fatalf("nonfinite: %v", err)
		}
	}
}

// TestResult_IsNilInterface protects pointer-presence semantics behind a typed interface.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Compare typed nil, manually allocated empty, and computed empty results.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - The structural IsNil interface and empty batch.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Readiness and slice lengths are not pointer nilability.
func TestResult_IsNilInterface(t *testing.T) {
	var missing *ema.Result
	var view nilable = missing
	if !view.IsNil() {
		t.Fatal("typed nil must report true")
	}
	view = &ema.Result{}
	if view.IsNil() {
		t.Fatal("allocated empty result is not nil")
	}
	result, err := ema.Compute(context.Background(), nil, ema.Options{Period: 14})
	if err != nil || result.IsNil() {
		t.Fatalf("computed empty: %v", err)
	}
}

// TestEMA_RationalOracle compares sequential rounding against an independent exact rational history.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Compare each batch and stream step bitwise in both seed modes.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - 640 deterministic finite samples and twelve periods, including MaxInt-1.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - The oracle must round after each transition, not just at the end.
func TestEMA_RationalOracle(t *testing.T) {
	x := finiteFixture(640)
	periods := []int{1, 2, 3, 4, 7, 14, 31, 64, 65, 127, 512, math.MaxInt - 1}
	for _, period := range periods {
		for _, seed := range []ema.Seed{ema.SeedSMA, ema.SeedFirst} {
			assertOracle(t, x, ema.Options{Period: period, Seed: seed})
		}
	}
}

// TestEMA_RoundingBoundaries exercises halfway, subnormal, exponent, and significand-carry boundaries.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Compare every constructed history against the exact rational oracle.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Boundary fixtures and adjacent values across all finite exponent fields.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Guard and sticky bits cannot be replaced by an approximate epsilon.
func TestEMA_RoundingBoundaries(t *testing.T) {
	smallest := math.SmallestNonzeroFloat64
	largestSubnormal := math.Float64frombits(0x000fffffffffffff)
	firstNormal := math.Float64frombits(0x0010000000000000)
	fixtures := [][]float64{
		{smallest, 0}, {3 * smallest, 0}, {-smallest, 0}, {-3 * smallest, 0},
		{smallest, 2 * smallest}, {largestSubnormal, firstNormal},
		{1, math.Nextafter(1, 2)}, {math.Nextafter(1, 2), math.Nextafter(math.Nextafter(1, 2), 2)},
		{math.MaxFloat64, math.MaxFloat64, math.MaxFloat64},
		{-math.MaxFloat64, -math.MaxFloat64, -math.MaxFloat64},
		{math.MaxFloat64, smallest, -math.MaxFloat64, smallest},
		{math.Copysign(0, -1), 0, smallest},
	}
	for _, x := range fixtures {
		for _, period := range []int{2, 3, 4, 5, 14, 64, math.MaxInt - 1} {
			for _, seed := range []ema.Seed{ema.SeedSMA, ema.SeedFirst} {
				assertOracle(t, x, ema.Options{Period: period, Seed: seed})
			}
		}
	}
	// Exercise guard/sticky and even ties at every binary exponent and limb offset.
	for exponent := 0; exponent < 2047; exponent++ {
		a := math.Float64frombits(uint64(exponent)<<52 | 0x0005555555555555)
		b := math.Nextafter(a, math.Inf(1))
		assertOracle(t, []float64{a, b, -a, -b}, ema.Options{Period: 3, Seed: ema.SeedFirst})
	}
}

// TestEMA_SubnormalIntegerGrid checks rational rounding over a signed subnormal lattice.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Enumerate all input pairs on the small integer-unit grid.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Periods 2..17 and signed unit counts -9..9.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Underflow to canonical zero is successful rounding, not an error.
func TestEMA_SubnormalIntegerGrid(t *testing.T) {
	for period := 2; period <= 17; period++ {
		for left := -9; left <= 9; left++ {
			for right := -9; right <= 9; right++ {
				x := []float64{float64(left) * math.SmallestNonzeroFloat64, float64(right) * math.SmallestNonzeroFloat64}
				assertOracle(t, x, ema.Options{Period: period, Seed: ema.SeedFirst})
			}
		}
	}
}

// TestEMA_OwnershipAndPrefixInvariance protects causal prefixes and independently owned output arrays.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Compare every prefix, then mutate inputs and a separate returned result.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - A fixed eight-observation price-like history.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - A shallow copy of Result is not a deep copy; independent Compute calls are.
func TestEMA_OwnershipAndPrefixInvariance(t *testing.T) {
	ctx := context.Background()
	x := []float64{62001, 61999, 62004, 62002, 62006, 62003, 62009, 62011}
	opts := ema.Options{Period: 3}
	expected := oracleHistory(x, opts)
	result, err := ema.Compute(ctx, x, opts)
	if err != nil {
		t.Fatal(err)
	}
	for n := 0; n <= len(x); n++ {
		prefix, callErr := ema.Compute(ctx, x[:n], opts)
		if callErr != nil {
			t.Fatal(callErr)
		}
		assertResult(t, prefix, expected[:n])
	}
	duplicate, err := ema.Compute(ctx, x, opts)
	if err != nil {
		t.Fatal(err)
	}
	x[2] = -1
	assertResult(t, result, expected)
	result.Values[2], result.Valid[2] = -123, false
	assertResult(t, duplicate, expected)
}

// TestEMA_TransformedRepresentedInputs checks transformed inputs without assuming impossible translation commutation.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Recompute the rational oracle from each transformed binary64 history.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Multiple positive scales and offsets, including very large shifts.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Do not compare to a transformed rounded result as though rounding were linear.
func TestEMA_TransformedRepresentedInputs(t *testing.T) {
	base := []float64{-4.5, 1.25, 2, -0.75, 8.5, 3.125, 4, -2, 7, 9, 3}
	for _, scale := range []float64{0.125, 1, 3, 1e100} {
		for _, offset := range []float64{0, 1000, 1e120} {
			x := make([]float64, len(base))
			for i, value := range base {
				x[i] = value*scale + offset
			}
			assertOracle(t, x, ema.Options{Period: 3})
			assertOracle(t, x, ema.Options{Period: 3, Seed: ema.SeedFirst})
		}
	}
}

// TestEMA_CancellationAtEveryBatchCheckpoint checks both scan phases and final publication cancellation.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Inject cancellation at each deterministic Err call and require a nil result.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - 2050 observations crossing three polling blocks, plus empty input.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Use logical cancellation rather than timing sleeps.
func TestEMA_CancellationAtEveryBatchCheckpoint(t *testing.T) {
	x := benchmarkHistory(2050)
	for check := int32(1); check <= 9; check++ {
		ctx, cancel := cancelAfter(check)
		result, err := ema.Compute(ctx, x, ema.Options{Period: 14})
		cancel()
		if result != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("check=%d result=%v err=%v", check, result, err)
		}
	}
	ctx, cancel := cancelAfter(3)
	defer cancel()
	result, err := ema.Compute(ctx, nil, ema.Options{Period: 14})
	if result != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("empty final boundary: %v", err)
	}
}

// TestEMA_IndependentConcurrentCalls checks independent computations under concurrent execution.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Run independent fixtures and oracles through subtest-owned goroutines.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Four parallel subtests with distinct state.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Do not call Fatal from an unmanaged worker goroutine.
func TestEMA_IndependentConcurrentCalls(t *testing.T) {
	for worker := 0; worker < 4; worker++ {
		t.Run("independent", func(t *testing.T) {
			t.Parallel()
			assertOracle(t, finiteFixture(48), ema.Options{Period: 14, Seed: ema.SeedFirst})
		})
	}
}

// TestEMA_SeedPermutationInvariance protects order independence of the exact SMA seed sum.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Check the first ready value for all permutations of a canceling triple.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - All six permutations of [1e308,1,-1e308].
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Only seed summation is permutation invariant; the mature recurrence is ordered.
func TestEMA_SeedPermutationInvariance(t *testing.T) {
	inputs := [][]float64{
		{1e308, 1, -1e308}, {1e308, -1e308, 1},
		{1, 1e308, -1e308}, {1, -1e308, 1e308},
		{-1e308, 1, 1e308}, {-1e308, 1e308, 1},
	}
	for _, x := range inputs {
		result, err := ema.Compute(context.Background(), x, ema.Options{Period: 3})
		if err != nil {
			t.Fatal(err)
		}
		assertValue(t, ema.Value{Value: result.Values[2], Valid: result.Valid[2]}, ema.Value{Value: 1.0 / 3, Valid: true})
	}
}

// TestEMA_ConstantFixedPointAndConvexHull protects exact fixed points and finite convex-result bounds.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Repeat constant extrema, then bound each arbitrary update by its two operands.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Five periods, signed extreme constants, and a deterministic mixed-exponent history.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Convexity does not justify approximating the single-rounding contract.
func TestEMA_ConstantFixedPointAndConvexHull(t *testing.T) {
	ctx := context.Background()
	for _, period := range []int{2, 3, 14, 64, math.MaxInt - 1} {
		for _, constant := range []float64{math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, 1, math.Nextafter(1, 2)} {
			stream := newStream(t, ema.Options{Period: period, Seed: ema.SeedFirst})
			for i := 0; i < 16; i++ {
				assertValue(t, pushValue(t, stream, constant), ema.Value{Value: constant, Valid: true})
			}
		}
		stream := newStream(t, ema.Options{Period: period, Seed: ema.SeedFirst})
		x := finiteFixture(128)
		previous := x[0]
		for _, sample := range x {
			value, err := stream.Push(ctx, sample)
			if err != nil {
				t.Fatal(err)
			}
			if !value.Valid || value.Value < math.Min(previous, sample) || value.Value > math.Max(previous, sample) {
				t.Fatalf("convex hull: previous=%g sample=%g got=%+v", previous, sample, value)
			}
			previous = value.Value
		}
	}
}
