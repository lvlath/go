// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr_test verifies transactional single-owner streaming and allocations.
// AI-Hints: errors must leave the exact seed and PreviousClose unchanged together.
package atr_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/atr"
)

// TestStream_BatchParityAndReset compares repeated reset/replay cycles to the owning batch result.
// multiple periods with warmup, first-ready, and mature transitions.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - multiple periods with warmup, first-ready, and mature transitions.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Reset preserves configuration and erases the prior close.
func TestStream_BatchParityAndReset(t *testing.T) {
	h, l, c := marketBars(96)
	for _, period := range []int{1, 2, 14, 64, 512} {
		s := newStream(t, period)
		initial := s.Snapshot()
		want := compute(t, h, l, c, period)
		for run := 0; run < 2; run++ {
			for i := range h {
				got := push(t, s, candle{h[i], l[i], c[i]})
				mustValue(t, got, atr.Value{TR: want.TR[i], ATR: want.ATR[i], Valid: want.Valid[i]})
			}
			s.Reset()
			mustSnapshot(t, s.Snapshot(), initial)
		}
	}
}

// TestStream_NilZeroAndContextPrecedence anchors nil and unconstructed receiver behavior after context checks.
// nil context, canceled context, absent stream, and zero Stream.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - nil context, canceled context, absent stream, and zero Stream.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Do not inspect fields of a nil receiver or fabricate default configuration.
func TestStream_NilZeroAndContextPrecedence(t *testing.T) {
	var absent *atr.Stream
	zero := &atr.Stream{}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	for _, s := range []*atr.Stream{absent, zero} {
		s.Reset()
		mustSnapshot(t, s.Snapshot(), atr.Snapshot{})
		v, err := s.Push(nil, 1, 0, 0)
		mustErrorIs(t, err, atr.ErrNilContext)
		mustValue(t, v, atr.Value{})
		v, err = s.Push(canceled, 1, 0, 0)
		mustErrorIs(t, err, context.Canceled)
		mustValue(t, v, atr.Value{})
		want := atr.ErrStateVersion
		if s == nil {
			want = atr.ErrNilInput
		}
		v, err = s.Push(context.Background(), math.NaN(), 0, 0)
		mustErrorIs(t, err, want)
		mustValue(t, v, atr.Value{})
		mustErrorIs(t, s.Restore(atr.Snapshot{}), want)
	}
}

// TestStream_RollbackAtEveryPhase protects complete state rollback before and after seed completion.
// geometry, finite-domain, range overflow, and final-context failures.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - geometry, finite-domain, range overflow, and final-context failures.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Compare SeedSum and PreviousClose as well as the visible ATR.
func TestStream_RollbackAtEveryPhase(t *testing.T) {
	for _, count := range []int{0, 1, 2, 3} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			s := newStream(t, 3)
			for i := 0; i < count; i++ {
				push(t, s, candle{10, 8, 9})
			}
			before := s.Snapshot()
			bad := []struct {
				bar  candle
				want error
			}{
				{candle{math.NaN(), 0, 0}, atr.ErrNonFinite},
				{candle{1, math.Inf(-1), 0}, atr.ErrNonFinite},
				{candle{1, 0, math.Inf(1)}, atr.ErrNonFinite},
				{candle{0, 1, 0}, atr.ErrBar},
				{candle{1, 0, -1}, atr.ErrBar},
				{candle{1, 0, 2}, atr.ErrBar},
				{candle{math.MaxFloat64, -math.MaxFloat64, 0}, atr.ErrOverflow},
			}
			for _, tc := range bad {
				v, err := s.Push(context.Background(), tc.bar.high, tc.bar.low, tc.bar.close)
				mustErrorIs(t, err, tc.want)
				mustValue(t, v, atr.Value{})
				mustSnapshot(t, s.Snapshot(), before)
			}
			v, err := s.Push(logicalCancellation(t, 2), 13, 11, 12)
			mustErrorIs(t, err, context.Canceled)
			mustValue(t, v, atr.Value{})
			mustSnapshot(t, s.Snapshot(), before)
			fresh := newStream(t, 3)
			if err = fresh.Restore(before); err != nil {
				t.Fatal(err)
			}
			mustValue(t, push(t, s, candle{13, 11, 12}), push(t, fresh, candle{13, 11, 12}))
		})
	}
}

// TestStream_CountOverflow rejects the next bar after MaxUint64 accepted observations.
// a restorable mature maximum counter and an explicit subsequent Reset.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - a restorable mature maximum counter and an explicit subsequent Reset.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - A counter may not wrap silently to a first-bar state.
func TestStream_CountOverflow(t *testing.T) {
	s := newStream(t, 2)
	state := s.Snapshot()
	state.Count = ^uint64(0)
	if err := s.Restore(state); err != nil {
		t.Fatal(err)
	}
	before := s.Snapshot()
	v, err := s.Push(context.Background(), 1, 0, 0)
	mustErrorIs(t, err, atr.ErrOverflow)
	mustValue(t, v, atr.Value{})
	mustSnapshot(t, s.Snapshot(), before)
	s.Reset()
	got := push(t, s, candle{1, 0, 0})
	mustValue(t, got, atr.Value{TR: 1})
}

// TestStream_ZeroAllocsPush measures internal allocations throughout warmup and mature updates.
// periods 1/14/64/512 and extreme finite range magnitudes.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - periods 1/14/64/512 and extreme finite range magnitudes.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Keep Snapshot serialization and fixture construction outside measured calls.
func TestStream_ZeroAllocsPush(t *testing.T) {
	ctx := context.Background()
	for _, period := range []int{1, 14, 64, 512} {
		t.Run(fmt.Sprint(period), func(t *testing.T) {
			s := newStream(t, period)
			for i := 0; i < period; i++ {
				push(t, s, candle{3, 1, 2})
			}
			var failure error
			allocations := testing.AllocsPerRun(1000, func() { _, failure = s.Push(ctx, 3, 1, 2) })
			if failure != nil {
				t.Fatal(failure)
			}
			if allocations != 0 {
				t.Fatalf("mature Push allocated: %g", allocations)
			}
			allocations = testing.AllocsPerRun(20, func() {
				s.Reset()
				for i := 0; i < period+1; i++ {
					_, err := s.Push(ctx, math.MaxFloat64, 0, 0)
					if err != nil {
						failure = err
					}
				}
			})
			if failure != nil {
				t.Fatal(failure)
			}
			if allocations != 0 {
				t.Fatalf("seed/readiness Push allocated: %g", allocations)
			}
		})
	}
}

// TestStream_IndependentParallelOwners runs independent streams concurrently without sharing mutable state.
// parallel subtests with separately owned configured streams.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - parallel subtests with separately owned configured streams.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - This does not authorize concurrent mutation of one Stream.
func TestStream_IndependentParallelOwners(t *testing.T) {
	h, l, c := marketBars(100)
	want := compute(t, h, l, c, 14)
	for worker := 0; worker < 4; worker++ {
		t.Run(fmt.Sprint(worker), func(t *testing.T) {
			t.Parallel()
			s := newStream(t, 14)
			for i := range h {
				mustValue(t, push(t, s, candle{h[i], l[i], c[i]}), atr.Value{TR: want.TR[i], ATR: want.ATR[i], Valid: want.Valid[i]})
			}
		})
	}
}
