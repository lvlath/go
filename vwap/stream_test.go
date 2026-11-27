// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap_test protects stream ownership, reset ordering, and atomic failures.
// This file compares public snapshots instead of exposing production-private helpers.
package vwap_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/vwap"
)

// TestVWAP_NegativeVolumeErrVolume protects finite negative-volume rejection without state loss.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: start a valid stream and submit a negative weight.
//   - Stage 3: require ErrVolume and full checkpoint equality.
//
// Behavior highlights:
//   - This test protects finite negative-volume rejection without state loss without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Signed prices remain valid; negative weights do not.
func TestVWAP_NegativeVolumeErrVolume(t *testing.T) {
	stream := newStream(t)
	mustPush(t, stream, 24, 3, true)
	for _, reset := range []bool{false, true} {
		before := stream.Snapshot()
		value, err := stream.Push(context.Background(), -15, -1, reset)
		requireError(t, err, vwap.ErrVolume)
		requireValue(t, value, vwap.Value{})
		requireUnchanged(t, stream, before)
	}
}

// TestVWAP_FailedResetRollback protects a live segment against invalid boundary observations.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: attempt reset=true with non-finite or negative-volume rows.
//   - Stage 3: require the old exact continuation to survive every rejected reset.
//
// Behavior highlights:
//   - This test protects a live segment against invalid boundary observations without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Changing only the displayed mean back cannot repair lost exact sums.
func TestVWAP_FailedResetRollback(t *testing.T) {
	stream := newStream(t)
	mustPush(t, stream, 24, 3, true)
	before := stream.Snapshot()
	cases := []struct {
		p, v float64
		want error
	}{
		{math.NaN(), 1, vwap.ErrNonFinite}, {math.Inf(1), 0, vwap.ErrNonFinite},
		{math.Inf(-1), 0, vwap.ErrNonFinite}, {1, math.NaN(), vwap.ErrNonFinite},
		{1, math.Inf(-1), vwap.ErrNonFinite}, {1, math.Inf(1), vwap.ErrNonFinite},
		{0, -1, vwap.ErrVolume},
	}
	for _, tc := range cases {
		value, err := stream.Push(context.Background(), tc.p, tc.v, true)
		requireError(t, err, tc.want)
		requireValue(t, value, vwap.Value{})
		requireUnchanged(t, stream, before)
	}
	requireValue(t, mustPush(t, stream, 30, 3, false), vwap.Value{Value: 27, Valid: true})
}

// TestStream_NilZeroAndPrecedence protects nil and unconstructed receiver behavior.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: exercise Push, Restore, Snapshot, and Reset on nil and zero streams.
//   - Stage 3: require the documented no-op or sentinel behavior and context precedence.
//
// Behavior highlights:
//   - This test protects nil and unconstructed receiver behavior without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Reset is not an alternative constructor for Stream{}.
func TestStream_NilZeroAndPrecedence(t *testing.T) {
	var missing *vwap.Stream
	var zero vwap.Stream
	cases := []struct {
		stream *vwap.Stream
		want   error
	}{
		{missing, vwap.ErrNilInput}, {&zero, vwap.ErrStateVersion},
	}
	for _, tc := range cases {
		value, err := tc.stream.Push(context.Background(), math.NaN(), -1, false)
		requireError(t, err, tc.want)
		requireValue(t, value, vwap.Value{})
		requireError(t, tc.stream.Restore(vwap.Snapshot{}), tc.want)
		tc.stream.Reset()
		if tc.stream.Snapshot() != (vwap.Snapshot{}) {
			t.Fatal("nil/zero snapshot must be zero")
		}
		_, err = tc.stream.Push(nil, 1, 1, true)
		requireError(t, err, vwap.ErrNilContext)
	}
	stream := newStream(t)
	before := stream.Snapshot()
	value, err := stream.Push(context.Background(), math.NaN(), -1, false)
	requireError(t, err, vwap.ErrAnchor)
	requireValue(t, value, vwap.Value{})
	requireUnchanged(t, stream, before)
	_, err = stream.Push(context.Background(), math.NaN(), -1, true)
	requireError(t, err, vwap.ErrNonFinite)
	requireUnchanged(t, stream, before)
	_, err = stream.Push(context.Background(), 1, 1, false)
	requireError(t, err, vwap.ErrAnchor)
}

// TestStream_ResetWarmupAndZeroContinuity protects masked/ready phase transitions and zero-weight continuity.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: replay phase fixtures and explicitly Reset a used stream.
//   - Stage 3: require correct Count, Started, and cached output at each stage.
//
// Behavior highlights:
//   - This test protects masked/ready phase transitions and zero-weight continuity without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Every successfully accepted zero-weight row still increments Count.
func TestStream_ResetWarmupAndZeroContinuity(t *testing.T) {
	stream := newStream(t)
	empty := stream.Snapshot()
	fixture := phaseFixture()
	p, v, r := fixture.price, fixture.volume, fixture.reset
	want := rationalOracle(p, v, r)
	count := uint64(0)
	for i := range p {
		if r[i] {
			count = 0
		}
		count++
		requireValue(t, mustPush(t, stream, p[i], v[i], r[i]), want[i])
		state := stream.Snapshot()
		if state.Count != count || !state.Started {
			t.Fatal("segment count or Started incorrect")
		}
	}
	stream.Reset()
	if stream.Snapshot() != empty {
		t.Fatal("Reset is not equivalent to New")
	}
	_, err := stream.Push(context.Background(), 1, 1, false)
	requireError(t, err, vwap.ErrAnchor)
	requireValue(t, mustPush(t, stream, -8, 2, true), vwap.Value{Value: -8, Valid: true})
}

// TestStream_CancellationRollback protects entry and pre-commit streaming cancellation.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: cancel a counted context before publication of a candidate.
//   - Stage 3: require Value{} and an unchanged exact checkpoint.
//
// Behavior highlights:
//   - This test protects entry and pre-commit streaming cancellation without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - A final cancellation poll belongs before the live-state assignment.
func TestStream_CancellationRollback(t *testing.T) {
	for _, reset := range []bool{false, true} {
		for _, at := range []int{1, 2} {
			stream := newStream(t)
			mustPush(t, stream, 10, 1, true)
			before := stream.Snapshot()
			value, err := stream.Push(newCountedContext(at), -10, 1, reset)
			requireError(t, err, context.Canceled)
			requireValue(t, value, vwap.Value{})
			requireUnchanged(t, stream, before)
			requireValue(t, mustPush(t, stream, 20, 1, false), vwap.Value{Value: 15, Valid: true})
		}
	}
	fresh := newStream(t)
	before := fresh.Snapshot()
	_, err := fresh.Push(newCountedContext(2), 1, 1, true)
	requireError(t, err, context.Canceled)
	requireUnchanged(t, fresh, before)
	_, err = fresh.Push(context.Background(), 1, 1, false)
	requireError(t, err, vwap.ErrAnchor)
}

// TestStream_CountOverflowAndExplicitReset protects count exhaustion independently of numerical magnitudes.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: Restore MaxUint64 and attempt ordinary, invalid, and explicitly reset rows.
//   - Stage 3: require atomic overflow rejection and a legal fresh reset.
//
// Behavior highlights:
//   - This test protects count exhaustion independently of numerical magnitudes without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Do not reject an explicit valid reset merely because the preceding segment is full.
func TestStream_CountOverflowAndExplicitReset(t *testing.T) {
	stream := newStream(t)
	mustPush(t, stream, 7, 1, true)
	state := stream.Snapshot()
	state.Count = ^uint64(0)
	if err := stream.Restore(state); err != nil {
		t.Fatal(err)
	}
	before := stream.Snapshot()
	for _, volume := range []float64{0, 1} {
		value, err := stream.Push(context.Background(), 7, volume, false)
		requireError(t, err, vwap.ErrOverflow)
		requireValue(t, value, vwap.Value{})
		requireUnchanged(t, stream, before)
	}
	_, err := stream.Push(context.Background(), math.NaN(), -1, false)
	requireError(t, err, vwap.ErrNonFinite)
	requireUnchanged(t, stream, before)
	_, err = stream.Push(context.Background(), 7, -1, false)
	requireError(t, err, vwap.ErrVolume)
	requireUnchanged(t, stream, before)
	requireValue(t, mustPush(t, stream, -17, 1, true), vwap.Value{Value: -17, Valid: true})
	if stream.Snapshot().Count != 1 {
		t.Fatal("explicit reset did not restart count")
	}
}

// TestStream_ValueCopiesAreIndependent protects fixed-value stream copying against hidden aliases.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: copy a constructed Stream value and evolve/reset each copy separately.
//   - Stage 3: require independent means and no cross-instance mutation.
//
// Behavior highlights:
//   - This test protects fixed-value stream copying against hidden aliases without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Adding pointer-backed scratch state would weaken this value-copy property.
func TestStream_ValueCopiesAreIndependent(t *testing.T) {
	first := newStream(t)
	mustPush(t, first, 1, 1, true)
	second := *first
	requireValue(t, mustPush(t, first, 3, 1, false), vwap.Value{Value: 2, Valid: true})
	requireValue(t, mustPush(t, &second, -3, 1, false), vwap.Value{Value: -1, Valid: true})
	second.Reset()
	requireValue(t, mustPush(t, first, 5, 1, false), vwap.Value{Value: 3, Valid: true})
}

// TestStream_NoInternalHeapAllocations protects allocation-free admitted and rejected Push paths.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: measure fixed valid, zero-weight, extreme, and invalid rows after setup.
//   - Stage 3: require the intended error classification and zero measured allocations.
//
// Behavior highlights:
//   - This test protects allocation-free admitted and rejected Push paths without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - A custom allocating context is outside this internal allocation measurement.
func TestStream_NoInternalHeapAllocations(t *testing.T) {
	stream := newStream(t)
	ctx := context.Background()
	cases := []struct {
		price, volume float64
		reset         bool
		want          error
	}{
		{10, 0, true, nil}, {10, 2, false, nil}, {-1e308, 1e308, false, nil},
		{1e308, 1e308, false, nil}, {1, 0, false, nil}, {math.NaN(), 1, true, vwap.ErrNonFinite},
		{1, -1, true, vwap.ErrVolume}, {1, 1, true, nil},
	}
	for _, tc := range cases {
		var got error
		allocations := testing.AllocsPerRun(100, func() {
			_, got = stream.Push(ctx, tc.price, tc.volume, tc.reset)
		})
		if tc.want != nil {
			requireError(t, got, tc.want)
		} else if got != nil {
			t.Fatal(got)
		}
		if allocations != 0 {
			t.Fatalf("Push allocated %g times", allocations)
		}
	}
}

// TestStream_IndependentParallelInstances protects independent stream calls under parallel test execution.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: launch separate seeded histories in owning parallel subtests.
//   - Stage 3: require each stream and batch to match its own rational oracle.
//
// Behavior highlights:
//   - This test protects independent stream calls under parallel test execution without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - This does not authorize concurrent mutation of the same Stream.
func TestStream_IndependentParallelInstances(t *testing.T) {
	for seed := int64(101); seed <= 108; seed++ {
		t.Run("independent", func(t *testing.T) {
			t.Parallel()
			p, v, r := randomHistory(seed, 73)
			verifyParity(t, p, v, r)
		})
	}
}
