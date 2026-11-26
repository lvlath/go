// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema_test exercises atomic streaming acceptance, reset, and allocation laws.
// Tests use only the exported ema surface, with no production test hooks.
package ema_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/ema"
)

// TestStream_NilAndZeroReceivers distinguishes nil receivers from unconstructed zero-value streams.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Check Push, Restore, Snapshot, Reset, and nil-context precedence.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Nil *Stream and an allocated zero Stream.
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
//   - Reset must not silently initialize a zero receiver.
func TestStream_NilAndZeroReceivers(t *testing.T) {
	var missing *ema.Stream
	var zero ema.Stream
	for _, stream := range []*ema.Stream{missing, &zero} {
		want := ema.ErrStateVersion
		if stream == nil {
			want = ema.ErrNilInput
		}
		value, err := stream.Push(context.Background(), 1)
		assertValue(t, value, ema.Value{})
		if !errors.Is(err, want) {
			t.Fatalf("Push: %v", err)
		}
		if err = stream.Restore(ema.Snapshot{}); !errors.Is(err, want) {
			t.Fatalf("Restore: %v", err)
		}
		stream.Reset()
		if !sameSnapshot(stream.Snapshot(), ema.Snapshot{}) {
			t.Fatal("empty snapshot required")
		}
	}
	value, err := missing.Push(nil, math.NaN())
	assertValue(t, value, ema.Value{})
	if !errors.Is(err, ema.ErrNilContext) {
		t.Fatalf("context precedence: %v", err)
	}
}

// TestStream_AtomicRollback protects complete rollback at every small-history phase.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Reject each nonfinite class and cancel before commit, comparing complete snapshots.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - P1, SMA and First across empty, partial-seed, completion, and mature phases.
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
//   - Never test rollback by checking Count alone; seed and Current also matter.
func TestStream_AtomicRollback(t *testing.T) {
	for _, opts := range []ema.Options{{Period: 1}, {Period: 3}, {Period: 3, Seed: ema.SeedFirst}} {
		stream := newStream(t, opts)
		for phase := 0; phase < 5; phase++ {
			before := stream.Snapshot()
			for _, invalid := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
				value, err := stream.Push(context.Background(), invalid)
				assertValue(t, value, ema.Value{})
				if !errors.Is(err, ema.ErrNonFinite) || !sameSnapshot(before, stream.Snapshot()) {
					t.Fatalf("nonfinite rollback: %v", err)
				}
			}
			value, err := stream.Push(nil, 1)
			assertValue(t, value, ema.Value{})
			if !errors.Is(err, ema.ErrNilContext) || !sameSnapshot(before, stream.Snapshot()) {
				t.Fatal("nil-context rollback")
			}
			// Cancellation occurs after the complete private candidate was calculated.
			ctx, cancel := cancelAfter(2)
			value, err = stream.Push(ctx, math.MaxFloat64)
			cancel()
			assertValue(t, value, ema.Value{})
			if !errors.Is(err, context.Canceled) || !sameSnapshot(before, stream.Snapshot()) {
				t.Fatal("final-check rollback")
			}
			pushValue(t, stream, float64(phase+1))
		}
	}
}

// TestStream_CountOverflowAndReset checks the final uint64 accept, overflow refusal, and explicit reset.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Restore MaxUint64-1, accept once, reject overflow, and replay after Reset.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Identity, SMA, and First mature states.
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
//   - Nonfinite input precedes count exhaustion; Reset must clear both.
func TestStream_CountOverflowAndReset(t *testing.T) {
	for _, opts := range []ema.Options{{Period: 1}, {Period: 3}, {Period: 3, Seed: ema.SeedFirst}} {
		stream := newStream(t, opts)
		state := stream.Snapshot()
		state.Count, state.Current = math.MaxUint64-1, 10
		if err := stream.Restore(state); err != nil {
			t.Fatal(err)
		}
		pushValue(t, stream, 20)
		before := stream.Snapshot()
		if before.Count != math.MaxUint64 {
			t.Fatal("last count was not accepted")
		}
		value, err := stream.Push(context.Background(), 30)
		assertValue(t, value, ema.Value{})
		if !errors.Is(err, ema.ErrOverflow) || !sameSnapshot(before, stream.Snapshot()) {
			t.Fatalf("overflow rollback: %v", err)
		}
		value, err = stream.Push(context.Background(), math.Inf(1))
		assertValue(t, value, ema.Value{})
		if !errors.Is(err, ema.ErrNonFinite) {
			t.Fatalf("input precedes count: %v", err)
		}
		stream.Reset()
		fresh := newStream(t, opts)
		if !sameSnapshot(stream.Snapshot(), fresh.Snapshot()) {
			t.Fatal("Reset differs from New")
		}
		for _, x := range []float64{1, 2, 3, 4} {
			assertValue(t, pushValue(t, stream, x), pushValue(t, fresh, x))
		}
	}
}

// TestStream_ResetClearsPartialSeed checks removal of a partially accumulated extreme seed.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Reset an extreme partial sum and compare the entire state with a new stream.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - P14 before warmup completes.
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
//   - Reset cannot leave low seed limbs hidden behind a zero Count.
func TestStream_ResetClearsPartialSeed(t *testing.T) {
	stream := newStream(t, ema.Options{Period: 14})
	pushValue(t, stream, math.MaxFloat64)
	pushValue(t, stream, 1)
	stream.Reset()
	fresh := newStream(t, ema.Options{Period: 14})
	if !sameSnapshot(stream.Snapshot(), fresh.Snapshot()) {
		t.Fatal("seed retained after Reset")
	}
	for _, value := range benchmarkHistory(32) {
		assertValue(t, pushValue(t, stream, value), pushValue(t, fresh, value))
	}
}

// TestStream_ZeroAllocsPush measures zero allocations in every streaming arithmetic regime.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Use AllocsPerRun outside construction and measure seed completion separately.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Identity, First, mature SMA, long warmup, and the exact completion step.
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
//   - Snapshot formatting and fixture setup do not belong inside a Push measurement.
func TestStream_ZeroAllocsPush(t *testing.T) {
	fixtures := []struct {
		name string
		opts ema.Options
		warm int
	}{
		{"identity", ema.Options{Period: 1}, 0},
		{"first", ema.Options{Period: 20, Seed: ema.SeedFirst}, 1},
		{"sma-mature", ema.Options{Period: 20}, 20},
		{"sma-warmup", ema.Options{Period: math.MaxInt - 1}, 0},
	}
	ctx := context.Background()
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			stream := newStream(t, fixture.opts)
			for i := 0; i < fixture.warm; i++ {
				pushValue(t, stream, 62000.125)
			}
			var err error
			var value ema.Value
			allocations := testing.AllocsPerRun(1000, func() { value, err = stream.Push(ctx, 62001.375) })
			if err != nil || allocations != 0 {
				t.Fatalf("allocs=%g err=%v value=%+v", allocations, err, value)
			}
		})
	}
	// Allocation-free even at the exact SMA seed-to-mature transition.
	stream := newStream(t, ema.Options{Period: 3})
	pushValue(t, stream, math.MaxFloat64)
	pushValue(t, stream, 1)
	state := stream.Snapshot()
	var pushErr, restoreErr error
	allocations := testing.AllocsPerRun(1000, func() {
		restoreErr = stream.Restore(state)
		_, pushErr = stream.Push(ctx, -math.MaxFloat64)
	})
	if allocations != 0 || pushErr != nil || restoreErr != nil {
		t.Fatalf("transition allocs=%g push=%v restore=%v", allocations, pushErr, restoreErr)
	}
}

// TestStream_ConstructionCopiesOptions checks copied construction configuration and independent stream value state.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Mutate the source options and advance a copied stream.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - An initialized First stream before its first observation.
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
//   - Public option values are copied, not retained through pointers.
func TestStream_ConstructionCopiesOptions(t *testing.T) {
	opts := ema.Options{Period: 3, Seed: ema.SeedFirst}
	stream := newStream(t, opts)
	opts.Period = 14
	if stream.Snapshot().Options.Period != 3 {
		t.Fatal("options aliased")
	}
	fork := *stream
	pushValue(t, &fork, 123)
	if stream.Snapshot().Count != 0 {
		t.Fatal("value-state copy aliases original")
	}
}

// TestStream_CancelBeforeLastCountCommit checks cancellation while the last uint64 count slot is still available.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Cancel after candidate computation, compare the snapshot, then retry successfully.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Restored Count MaxUint64-1 with a finite mature current.
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
//   - The final context check must precede count publication.
func TestStream_CancelBeforeLastCountCommit(t *testing.T) {
	stream := newStream(t, ema.Options{Period: 14, Seed: ema.SeedFirst})
	state := stream.Snapshot()
	state.Count, state.Current = math.MaxUint64-1, 100
	if err := stream.Restore(state); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := cancelAfter(2)
	value, err := stream.Push(ctx, -math.MaxFloat64)
	cancel()
	assertValue(t, value, ema.Value{})
	if !errors.Is(err, context.Canceled) || !sameSnapshot(state, stream.Snapshot()) {
		t.Fatal("cancellation consumed the last available count")
	}
	want := oracleHistory([]float64{state.Current, -math.MaxFloat64}, state.Options)
	assertValue(t, pushValue(t, stream, -math.MaxFloat64), want[1])
	if stream.Snapshot().Count != math.MaxUint64 {
		t.Fatal("retry did not consume the last count")
	}
}

// TestStream_WarmupForkOwnsAccumulator checks that value-copy forks do not alias partial seed storage.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Advance a fork on a different suffix, then continue and validate the original.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - P14 copied after five samples; original and negated suffixes.
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
//   - Independent stream copies must include all accumulator limbs.
func TestStream_WarmupForkOwnsAccumulator(t *testing.T) {
	inputs := failoverHistory()
	opts := ema.Options{Period: 14}
	original := newStream(t, opts)
	for _, x := range inputs[:5] {
		pushValue(t, original, x)
	}
	before := original.Snapshot()
	fork := *original
	// Mutating a copied partial accumulator must not touch its original owner.
	for _, x := range inputs[5:] {
		pushValue(t, &fork, -x)
	}
	if !sameSnapshot(before, original.Snapshot()) {
		t.Fatal("fork aliases original seed")
	}
	for _, x := range inputs[5:] {
		pushValue(t, original, x)
	}
	want := oracleHistory(inputs, opts)
	if math.Float64bits(original.Snapshot().Current) != math.Float64bits(want[len(want)-1].Value) {
		t.Fatal("original continuation changed after fork")
	}
	fork.Reset()
	if original.Snapshot().Count != uint64(len(inputs)) {
		t.Fatal("reset of fork leaked")
	}
}

// TestStream_AdversarialInputsRemainAllocationFree measures allocations while alternating large and subnormal signed inputs.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Repeat public Push under three mature/warmup regimes and validate finite output.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Extreme cancellation and very small coefficients.
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
//   - A benchmark on nearby positive prices alone does not characterize the numerical regimes.
func TestStream_AdversarialInputsRemainAllocationFree(t *testing.T) {
	values := []float64{math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, 1, -1, 0}
	ctx := context.Background()
	for _, opts := range []ema.Options{{Period: 3, Seed: ema.SeedFirst}, {Period: math.MaxInt - 1, Seed: ema.SeedFirst}, {Period: math.MaxInt - 1, Seed: ema.SeedSMA}} {
		stream := newStream(t, opts)
		var callErr error
		var output ema.Value
		index := 0
		allocations := testing.AllocsPerRun(1000, func() {
			output, callErr = stream.Push(ctx, values[index%len(values)])
			index++
		})
		if callErr != nil || allocations != 0 || math.IsNaN(output.Value) || math.IsInf(output.Value, 0) {
			t.Fatalf("opts=%+v allocs=%g err=%v value=%+v", opts, allocations, callErr, output)
		}
	}
}
