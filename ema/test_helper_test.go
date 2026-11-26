// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema_test checks public contracts using independent exact rational arithmetic.
// Shared fixtures, controlled contexts, assertions, and oracles live in this file only.
package ema_test

import (
	"context"
	"math"
	"math/big"
	"sync/atomic"
	"testing"

	"github.com/lvlath/go/ema"
)

var (
	benchmarkResult *ema.Result
	benchmarkValue  ema.Value
)

// newStream constructs a stream or fails in the owning test context.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Validate Options through ema.New and return its owning stream.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - testing.TB and explicit ema.Options.
//
// Returns:
//   - *ema.Stream with an initialized empty history.
//
// Errors:
//   - Unexpected operation failures are reported by the owning test when applicable.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time/space O(W).
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Do not create test-only production constructors.
func newStream(t testing.TB, opts ema.Options) *ema.Stream {
	t.Helper()
	stream, err := ema.New(opts)
	if err != nil {
		t.Fatalf("New(%+v): %v", opts, err)
	}
	return stream
}

// pushValue accepts one public observation or fails with its original context.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Call Push with Background and return the detached accepted Value.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - testing.TB, initialized stream, and finite fixture input.
//
// Returns:
//   - ema.Value from the public transition.
//
// Errors:
//   - Unexpected operation failures are reported by the owning test when applicable.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time/space O(W).
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Use explicit Push calls in error-path tests instead of hiding an expected failure here.
func pushValue(t testing.TB, stream *ema.Stream, x float64) ema.Value {
	t.Helper()
	value, err := stream.Push(context.Background(), x)
	if err != nil {
		t.Fatalf("Push(%g): %v", x, err)
	}
	return value
}

// sameValue compares readiness and every output bit without reflection.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Compare Valid, then compare Float64bits of both payloads.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - Two ema.Value records.
//
// Returns:
//   - bool: exact value-and-mask equality.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time/space O(1).
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - A tolerance must not merge positive and negative zero or hide a missing readiness flag.
func sameValue(a, b ema.Value) bool {
	return a.Valid == b.Valid && math.Float64bits(a.Value) == math.Float64bits(b.Value)
}

// sameSnapshot compares all comparable fields and the exact Current bits.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Use field equality plus a zero-sign-sensitive float bit comparison.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - Two detached ema.Snapshot records.
//
// Returns:
//   - bool: exact canonical state equality.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time O(D), auxiliary Space O(1), D<=660.
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Count-only rollback checks can miss an altered seed or current value.
func sameSnapshot(a, b ema.Snapshot) bool {
	return a == b && math.Float64bits(a.Current) == math.Float64bits(b.Current)
}

// assertValue reports exact measurement mismatches with hexadecimal payload evidence.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Mark the helper frame and fail on a bit or readiness difference.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - testing.TB plus actual and expected measurements.
//
// Returns:
//   - No value; owning-test failure on mismatch.
//
// Errors:
//   - Unexpected operation failures are reported by the owning test when applicable.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time/space O(1) outside diagnostic formatting.
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Preserve the expected mask instead of inferring it from the numeric payload.
func assertValue(t testing.TB, got, want ema.Value) {
	t.Helper()
	if !sameValue(got, want) {
		t.Fatalf("got=%+v (%016x), want=%+v (%016x)", got, math.Float64bits(got.Value), want, math.Float64bits(want.Value))
	}
}

// assertResult checks owning batch shape before accessing any output element.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Require non-nil channels and exact lengths, then compare each aligned Value.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - testing.TB, result pointer, and independent expected measurements.
//
// Returns:
//   - No value; failures identify the exact mismatch index.
//
// Errors:
//   - Unexpected operation failures are reported by the owning test when applicable.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time O(n), auxiliary Space O(1).
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Check result nilness before dereferencing its slice fields.
func assertResult(t testing.TB, got *ema.Result, want []ema.Value) {
	t.Helper()
	if got == nil || got.Values == nil || got.Valid == nil {
		t.Fatal("nil result or channel")
	}
	if len(got.Values) != len(want) || len(got.Valid) != len(want) {
		t.Fatalf("shape: %d/%d, expected %d", len(got.Values), len(got.Valid), len(want))
	}
	for i, expected := range want {
		actual := ema.Value{Value: got.Values[i], Valid: got.Valid[i]}
		if !sameValue(actual, expected) {
			t.Fatalf("index=%d got=%+v bits=%016x want=%+v bits=%016x", i, actual,
				math.Float64bits(actual.Value), expected, math.Float64bits(expected.Value))
		}
	}
}

// ratRound rounds an independent exact rational once and canonicalizes zero.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Use math/big.Rat.Float64 and remove only a negative-zero sign.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - A non-nil exact rational.
//
// Returns:
//   - float64 nearest-even measurement.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Big-rational conversion cost depends on operand bit length.
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Never call the production rounding helper from the oracle.
func ratRound(value *big.Rat) float64 {
	result, _ := value.Float64()
	if result == 0 {
		return 0
	}
	return result
}

// oracleHistory models the sequentially rounded seed and affine history independently.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Use exact rational seed sums and round each recurrence before reusing previous.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - Finite history and valid explicit ema.Options.
//
// Returns:
//   - Owning []ema.Value with exact readiness and canonical zeros.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time O(n) rational operations, Space O(n) output plus bounded-size operands.
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Expanding the entire history and rounding only once computes a different algorithm.
func oracleHistory(x []float64, opts ema.Options) []ema.Value {
	result := make([]ema.Value, len(x))
	sum := new(big.Rat)
	period := new(big.Rat).SetInt64(int64(opts.Period))
	oldWeight := new(big.Rat).SetInt64(int64(opts.Period) - 1)
	denominator := new(big.Rat).SetInt64(int64(opts.Period) + 1)
	two := big.NewRat(2, 1)
	var previous float64
	for i, value := range x {
		input := new(big.Rat).SetFloat64(value)
		switch {
		case opts.Period == 1:
			previous = value
			if previous == 0 {
				previous = 0
			}
		case opts.Seed == ema.SeedSMA && i < opts.Period:
			sum.Add(sum, input)
			if i+1 < opts.Period {
				continue
			}
			previous = ratRound(new(big.Rat).Quo(sum, period))
		case i == 0:
			previous = value
			if previous == 0 {
				previous = 0
			}
		default:
			old := new(big.Rat).SetFloat64(previous)
			numerator := new(big.Rat).Add(new(big.Rat).Mul(old, oldWeight), new(big.Rat).Mul(input, two))
			previous = ratRound(new(big.Rat).Quo(numerator, denominator))
		}
		result[i] = ema.Value{Value: previous, Valid: true}
	}
	return result
}

// assertOracle compares batch and stream with an independent mathematical history.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Build oracle values, compare the public batch, and compare every public Push.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - testing.TB, finite fixture, and valid options.
//
// Returns:
//   - No value; complete aligned comparisons on success.
//
// Errors:
//   - Unexpected operation failures are reported by the owning test when applicable.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time O(n) rational operations plus O(n*W) production work; Space O(n+W).
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Checking batch against stream alone cannot detect a shared kernel defect.
func assertOracle(t testing.TB, x []float64, opts ema.Options) {
	t.Helper()
	want := oracleHistory(x, opts)
	got, err := ema.Compute(context.Background(), x, opts)
	if err != nil {
		t.Fatalf("Compute(%+v): %v", opts, err)
	}
	assertResult(t, got, want)
	stream := newStream(t, opts)
	for i, value := range x {
		actual := pushValue(t, stream, value)
		if !sameValue(actual, want[i]) {
			t.Fatalf("stream index=%d opts=%+v x=%g got=%+v want=%+v", i, opts, value, actual, want[i])
		}
	}
}

// exactSeedText computes an independent canonical decimal seed representation.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Sum exact rationals, scale by 2^1074, and format the integer numerator.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - Finite binary64 prefix.
//
// Returns:
//   - Canonical decimal units of its exact sum.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - O(n) rational additions plus bounded integer formatting.
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Do not use float64 accumulation before constructing the rational.
func exactSeedText(values []float64) string {
	sum := new(big.Rat)
	for _, value := range values {
		sum.Add(sum, new(big.Rat).SetFloat64(value))
	}
	scale := new(big.Int).Lsh(big.NewInt(1), 1074)
	sum.Mul(sum, new(big.Rat).SetInt(scale))
	return sum.Num().String()
}

// maxUnits builds the largest finite binary64 magnitude in exact integer units.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Construct (2^53-1) and shift it by 2045 bits.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - No runtime input; binary64 representation constants.
//
// Returns:
//   - New mutable big.Int equal to MaxBinary64Units.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Fixed bounded bigint work and storage.
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Return a new integer so one test cannot mutate another test's bound.
func maxUnits() *big.Int {
	mantissa := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 53), big.NewInt(1))
	return mantissa.Lsh(mantissa, 2045)
}

// fixtureBits advances a local reproducible integer fixture generator.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Mix caller-owned state with fixed unsigned SplitMix-style operations.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - Non-nil pointer to local uint64 generator state.
//
// Returns:
//   - Deterministic uint64 bit pattern.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time/space O(1).
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - This is reproducible test data generation, not cryptographic randomness.
func fixtureBits(state *uint64) uint64 {
	*state += 0x9e3779b97f4a7c15
	value := *state
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
	value = (value ^ (value >> 27)) * 0x94d049bb133111eb
	return value ^ (value >> 31)
}

// finiteFixture builds deterministic finite values across the binary64 exponent range.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Generate local bits and remap only the nonfinite exponent pattern.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - n: nonnegative fixture length.
//
// Returns:
//   - Owning finite []float64 of length n.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Do not silently map all exceptional patterns to zero and lose exponent coverage.
func finiteFixture(n int) []float64 {
	result := make([]float64, n)
	state := uint64(0x31a01)
	for i := range result {
		raw := fixtureBits(&state)
		if raw>>52&0x7ff == 0x7ff {
			raw ^= uint64(1) << 52
		}
		result[i] = math.Float64frombits(raw)
	}
	return result
}

// benchmarkHistory builds a bounded price-like workload before benchmark timing.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Populate a reproducible eighth-unit oscillation around a fixed level.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - n: requested observation count.
//
// Returns:
//   - Owning []float64 with deterministic finite values.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time O(n), Space O(n), excluded from benchmark timing.
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Input construction is not part of the EMA throughput measurement.
func benchmarkHistory(n int) []float64 {
	values := make([]float64, n)
	for i := range values {
		values[i] = 62000 + float64((i*37)%211)/8
	}
	return values
}

// cancelAfterContext implements logical cancellation at a selected public Err checkpoint.
// It embeds an ordinary cancelable context and drives cancellation without sleeping.
//
// Implementation:
//   - Stage 1: create the documented fixture fields explicitly.
//   - Stage 2: consume the fields through the corresponding external test or example boundary.
//
// Behavior highlights:
//   - The carrier introduces no production API or private implementation access.
//
// Inputs:
//   - Initialized embedded context, cancel function, and atomic remaining-check counter.
//
// Returns:
//   - A context usable at deterministic cancellation boundaries.
//
// Errors:
//   - The carrier itself creates no errors; consuming public operations report their own failures.
//
// Determinism:
//   - Explicit fixture order and field values determine behavior.
//
// Complexity:
//   - O(1) carrier size apart from detached snapshot text and referenced fixture data.
//
// Notes:
//   - Shared helper types live here rather than in production or example-only package internals.
//
// AI-Hints:
//   - Do not infer guarantees about racing real-world cancellation from a controlled test context.
type cancelAfterContext struct {
	// Context supplies standard Done and cancellation state.
	context.Context
	// cancel closes Done when the configured Err boundary is reached.
	cancel context.CancelFunc
	// remaining counts Err observations rather than elapsed time.
	remaining atomic.Int32
}

// cancelAfter creates deterministic logical cancellation after a chosen number of Err checks.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Construct a cancelable base context and initialize an atomic countdown.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - checks: positive countdown for the observed Err boundary.
//
// Returns:
//   - Controlled context and its cleanup CancelFunc.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time/space O(1) setup, outside measured Push loops.
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Do not replace this with a sleep that can race the computation.
func cancelAfter(checks int32) (*cancelAfterContext, context.CancelFunc) {
	base, cancel := context.WithCancel(context.Background())
	result := &cancelAfterContext{Context: base, cancel: cancel}
	result.remaining.Store(checks)
	return result, cancel
}

// Err advances the logical cancellation countdown and reports the base context state.
// Closing the base Done channel precedes returning a cancellation error.
//
// Implementation:
//   - Stage 1: atomically decrement the remaining-check counter.
//   - Stage 2: cancel the embedded context when the configured boundary is reached.
//   - Stage 3: return the embedded context error.
//
// Behavior highlights:
//   - No sleep or scheduler-dependent timeout is involved.
//
// Inputs:
//   - ctx: initialized controlled context from cancelAfter.
//
// Returns:
//   - error: nil before the threshold, context.Canceled at and after it.
//
// Errors:
//   - The base cancellation sentinel is preserved.
//
// Determinism:
//   - The configured number of observed Err calls determines cancellation.
//
// Complexity:
//   - Time/space O(1), excluding standard context internals.
//
// Notes:
//   - A test chooses the threshold for the public operation it is exercising.
//
// AI-Hints:
//   - Never use this test context to assert a deterministic order between genuinely concurrent events.
func (ctx *cancelAfterContext) Err() error {
	if ctx.remaining.Add(-1) <= 0 {
		ctx.cancel()
	}
	return ctx.Context.Err()
}

// nilable is the structural pointer-presence protocol for optional results.
// Its one method permits safe typed-nil handling without reflection.
//
// Implementation:
//   - Stage 1: create the documented fixture fields explicitly.
//   - Stage 2: consume the fields through the corresponding external test or example boundary.
//
// Behavior highlights:
//   - The carrier introduces no production API or private implementation access.
//
// Inputs:
//   - A *ema.Result stored directly or inside this interface.
//
// Returns:
//   - bool from IsNil, independent of channel length or readiness.
//
// Errors:
//   - The carrier itself creates no errors; consuming public operations report their own failures.
//
// Determinism:
//   - Explicit fixture order and field values determine behavior.
//
// Complexity:
//   - O(1) carrier size apart from detached snapshot text and referenced fixture data.
//
// Notes:
//   - Shared helper types live here rather than in production or example-only package internals.
//
// AI-Hints:
//   - Do not equate an empty successful result with a missing result.
type nilable interface {
	IsNil() bool
}

// Compile-time satisfaction complements runtime typed-nil and empty-result checks.
var _ nilable = (*ema.Result)(nil)

// namedSeedFacade groups the two public entry points for one explicitly named seed policy.
// Tests can verify wrapper configuration, errors, and ownership without sharing numeric code.
//
// Implementation:
//   - Stage 1: create the documented fixture fields explicitly.
//   - Stage 2: consume the fields through the corresponding external test or example boundary.
//
// Behavior highlights:
//   - The carrier introduces no production API or private implementation access.
//
// Inputs:
//   - A name, expected Seed enum, and exported Compute/New function values.
//
// Returns:
//   - One immutable-by-convention external test fixture.
//
// Errors:
//   - The carrier itself creates no errors; consuming public operations report their own failures.
//
// Determinism:
//   - Explicit fixture order and field values determine behavior.
//
// Complexity:
//   - O(1) carrier size apart from detached snapshot text and referenced fixture data.
//
// Notes:
//   - Shared helper types live here rather than in production or example-only package internals.
//
// AI-Hints:
//   - Never replace a function value with a local numerical imitation of the facade.
type namedSeedFacade struct {
	// name labels the policy in subtest diagnostics.
	name string
	// seed is the exact policy that the wrappers must select.
	seed ema.Seed
	// compute points to an exported named batch facade.
	compute func(context.Context, []float64, int) (*ema.Result, error)
	// newStream points to its matching exported stream constructor.
	newStream func(int) (*ema.Stream, error)
}

// namedFacades enumerates the two explicit public seed-policy boundaries.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Return constructor/batch function pairs with their exact seed enum.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - No input; only exported ema functions are referenced.
//
// Returns:
//   - Detached small table of named seed facades.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time/space O(1).
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Do not implement a second algorithm inside a wrapper fixture.
func namedFacades() []namedSeedFacade {
	return []namedSeedFacade{
		{"SMA", ema.SeedSMA, ema.ComputeSMA, ema.NewSMA},
		{"First", ema.SeedFirst, ema.ComputeFirst, ema.NewFirst},
	}
}

// snapshotAdmitted independently models structural snapshot admission with big integers.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Check schema and phase, then require canonical bigint round-trip and the exact count bound.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - Untrusted snapshot and valid receiving options.
//
// Returns:
//   - bool: expected admission independent of production parsing.
//
// Errors:
//   - None; admission failures are represented by false, not an error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Time O(D) bounded-bigint digit steps, Space O(D), D<=660.
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - A fuzz test must check acceptance correctness, not just stability after an accepted state.
func snapshotAdmitted(state ema.Snapshot, opts ema.Options) bool {
	// Schema and exact options select the mathematical continuation contract.
	if state.Version != 1 || state.Algorithm != "ema/affine-rn/v1" || state.Options != opts {
		return false
	}
	// Scalar admission is independent of seed parsing and accepts signed nonzero values.
	if math.IsNaN(state.Current) || math.IsInf(state.Current, 0) || (state.Current == 0 && math.Signbit(state.Current)) {
		return false
	}
	// First/mature phases admit no residual seed; an empty stream has no measurement.
	if opts.Seed == ema.SeedFirst || state.Count >= uint64(opts.Period) {
		return state.SeedSum == "0" && (state.Count != 0 || state.Current == 0)
	}
	// Warmup is zero-valued and bounded before any allocating oracle parser is invoked.
	if state.Current != 0 || len(state.SeedSum) == 0 || len(state.SeedSum) > 660 {
		return false
	}
	// Canonical round-trip rejects plus signs, whitespace, leading zeros, and negative zero.
	integer, ok := new(big.Int).SetString(state.SeedSum, 10)
	if !ok || integer.String() != state.SeedSum {
		return false
	}
	// The independent bigint bound has no shared fixed-limb implementation.
	bound := new(big.Int).Mul(maxUnits(), new(big.Int).SetUint64(state.Count))
	return integer.Abs(integer).Cmp(bound) <= 0
}

// completedSeedValue independently rounds a restored seed followed by its final observation.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Rescale exact units, add the represented sample, divide by P, and round once.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - Signed big.Int units, finite sample, and positive period.
//
// Returns:
//   - Ready ema.Value with the exact rounded mean.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Bounded rational work and storage.
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Copy the supplied bigint before performing mutable rational operations.
func completedSeedValue(sum *big.Int, sample float64, period int) ema.Value {
	units := new(big.Int).Lsh(big.NewInt(1), 1074)
	rational := new(big.Rat).SetFrac(new(big.Int).Set(sum), units)
	rational.Add(rational, new(big.Rat).SetFloat64(sample))
	rational.Quo(rational, new(big.Rat).SetInt64(int64(period)))
	return ema.Value{Value: ratRound(rational), Valid: true}
}

// quoteTick represents one upstream bid/ask event in a deterministic replay fixture.
// Source ordering and quote validation belong to the example consumer, not EMA.
//
// Implementation:
//   - Stage 1: create the documented fixture fields explicitly.
//   - Stage 2: consume the fields through the corresponding external test or example boundary.
//
// Behavior highlights:
//   - The carrier introduces no production API or private implementation access.
//
// Inputs:
//   - Consecutive sequence IDs and finite positive prices, with explicit adversarial events.
//
// Returns:
//   - A by-value event with no pointers or retained history.
//
// Errors:
//   - The carrier itself creates no errors; consuming public operations report their own failures.
//
// Determinism:
//   - Explicit fixture order and field values determine behavior.
//
// Complexity:
//   - O(1) carrier size apart from detached snapshot text and referenced fixture data.
//
// Notes:
//   - Shared helper types live here rather than in production or example-only package internals.
//
// AI-Hints:
//   - Do not acknowledge a sequence whose quote was rejected.
type quoteTick struct {
	// sequence is the application event identity, independent of EMA Count.
	sequence uint64
	// bid is the quoted buy-side price in fixture currency units.
	bid float64
	// ask is the quoted sell-side price; crossed quotes are explicitly rejected upstream.
	ask float64
}

// quoteReplay provides an auditable upstream quote replay with deliberate admission faults.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Return explicit ordered bid/ask records including one duplicate and one crossed quote.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - No input; prices are synthetic application fixtures.
//
// Returns:
//   - 18 records representing 16 valid consecutive events.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Fixed O(n) fixture storage, outside Push timing.
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - Do not attribute these illustrative prices to a real exchange or historical date.
func quoteReplay() []quoteTick {
	return []quoteTick{
		{9001, 62104.25, 62104.75}, {9002, 62104.00, 62104.50},
		{9002, 62104.00, 62104.50}, {9003, 62104.50, 62105.00},
		{9004, 62105.00, 62104.50}, {9004, 62104.25, 62104.75},
		{9005, 62104.75, 62105.25}, {9006, 62104.50, 62105.00},
		{9007, 62105.00, 62105.50}, {9008, 62104.75, 62105.25},
		{9009, 62105.25, 62105.75}, {9010, 62105.00, 62105.50},
		{9011, 62105.50, 62106.00}, {9012, 62105.25, 62105.75},
		{9013, 62105.75, 62106.25}, {9014, 62105.50, 62106.00},
		{9015, 62106.00, 62106.50}, {9016, 62105.75, 62106.25},
	}
}

// checkpointEnvelope binds application replay identity to a detached numerical snapshot.
// Typed persistence protects uint64 precision while the storage owner supplies authentication.
//
// Implementation:
//   - Stage 1: create the documented fixture fields explicitly.
//   - Stage 2: consume the fields through the corresponding external test or example boundary.
//
// Behavior highlights:
//   - The carrier introduces no production API or private implementation access.
//
// Inputs:
//   - Instrument identity, last accepted source sequence, and same-options EMA snapshot.
//
// Returns:
//   - A detached JSON-serializable continuation record.
//
// Errors:
//   - The carrier itself creates no errors; consuming public operations report their own failures.
//
// Determinism:
//   - Explicit fixture order and field values determine behavior.
//
// Complexity:
//   - O(1) carrier size apart from detached snapshot text and referenced fixture data.
//
// Notes:
//   - Shared helper types live here rather than in production or example-only package internals.
//
// AI-Hints:
//   - JSON encoding is not authentication or proof of the source history.
type checkpointEnvelope struct {
	// Instrument identifies which upstream series owns this state.
	Instrument string
	// LastSequence advances only together with a committed numeric state.
	LastSequence uint64
	// EMA is the complete detached continuation state, including unfinished seed units.
	EMA ema.Snapshot
}

// failoverHistory provides a reproducible close history spanning incomplete and completed warmup.
// Its scope is limited to deterministic external-package verification or example input preparation.
//
// Implementation:
//   - Stage 1: Return twenty explicit eighth-dollar observations in fixed order.
//   - Stage 2: return or compare the documented artifact without accessing production internals.
//
// Behavior highlights:
//   - No reflection, private test hooks, or implicit global randomness is used.
//
// Inputs:
//   - No input; values are synthetic application fixtures.
//
// Returns:
//   - Owning []float64 used by replay and continuation checks.
//
// Errors:
//   - None for the documented fixture/input domain; this helper returns no error object.
//
// Determinism:
//   - All local fixture traversal and explicit comparisons have fixed order.
//
// Complexity:
//   - Fixed O(n) fixture allocation.
//
// Notes:
//   - Helper allocations are not included in the production hot-path allocation claim.
//
// AI-Hints:
//   - A checkpoint must persist accepted source identity as well as these numeric state fields.
func failoverHistory() []float64 {
	return []float64{
		201.125, 201.250, 201.000, 200.875, 201.375, 201.500, 201.250,
		201.875, 202.000, 201.750, 202.125, 202.375, 202.000, 202.500,
		202.875, 202.625, 203.000, 203.125, 202.750, 203.375,
	}
}
