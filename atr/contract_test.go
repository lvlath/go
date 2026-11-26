// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr_test protects replay and exact transformation laws on public interfaces.
// Code-defined scenarios complement, rather than replace, adversarial rational tests.
package atr_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/atr"
)

// TestATR_FrozenSourceOracleAndEverySplit checks the complete code-defined daily history.
// Every possible split restores precisely the accepted prefix, including empty and ready state.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - Every possible split restores precisely the accepted prefix, including empty and ready state.
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
//   - Preserve exact payload, mask, error identity, and ownership assertions.
func TestATR_FrozenSourceOracleAndEverySplit(t *testing.T) {
	bars := dailyBars()
	if len(bars) != 20 || bars[0].date != "2026-10-26" || bars[19].date != "2026-11-20" {
		t.Fatal("fixed scenario identity changed")
	}
	high, low, closed := dailyChannels(bars)
	for _, period := range []int{1, 2, 14, 64} {
		want := rationalHistory(high, low, closed, period)
		mustResult(t, compute(t, high, low, closed, period), want)
		for split := 0; split <= len(bars); split++ {
			primary := newStream(t, period)
			for _, bar := range bars[:split] {
				push(t, primary, bar.candle)
			}
			standby := newStream(t, period)
			mustErrorIs(t, standby.Restore(primary.Snapshot()), nil)
			for i := split; i < len(bars); i++ {
				got := push(t, standby, bars[i].candle)
				mustValue(t, got, atr.Value{TR: want.TR[i], ATR: want.ATR[i], Valid: want.Valid[i]})
			}
		}
	}
}

// TestATR_ExactUnitScaleAndSignReversal defends two valid metamorphic relations.
// Power-of-two scaling stays in normal finite range; negation swaps high and low.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - Power-of-two scaling stays in normal finite range; negation swaps high and low.
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
//   - Preserve exact payload, mask, error identity, and ownership assertions.
func TestATR_ExactUnitScaleAndSignReversal(t *testing.T) {
	high, low, closed := marketBars(96)
	want := compute(t, high, low, closed, 14)
	for _, exponent := range []int{-40, 0, 40} {
		h, l, c := make([]float64, len(high)), make([]float64, len(high)), make([]float64, len(high))
		for i := range high {
			h[i], l[i], c[i] = -math.Ldexp(low[i], exponent), -math.Ldexp(high[i], exponent), -math.Ldexp(closed[i], exponent)
		}
		got := compute(t, h, l, c, 14)
		mustResult(t, got, rationalHistory(h, l, c, 14))
		for i := range high {
			mustFloatBits(t, got.TR[i], math.Ldexp(want.TR[i], exponent))
			mustFloatBits(t, got.ATR[i], math.Ldexp(want.ATR[i], exponent))
			if got.Valid[i] != want.Valid[i] {
				t.Fatal("unit conversion changed readiness")
			}
		}
	}
}

// TestStream_CorrectedHistoryReplaysFromEarlierCheckpoint anchors revision handling.
// A corrected bar is replayed from its preceding checkpoint, not appended as a new observation.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - A corrected bar is replayed from its preceding checkpoint, not appended as a new observation.
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
//   - Preserve exact payload, mask, error identity, and ownership assertions.
func TestStream_CorrectedHistoryReplaysFromEarlierCheckpoint(t *testing.T) {
	high, low, closed := marketBars(48)
	const correctionIndex = 19
	stream := newStream(t, 14)
	for i := 0; i < correctionIndex; i++ {
		push(t, stream, candle{high[i], low[i], closed[i]})
	}
	before := stream.Snapshot()
	for i := correctionIndex; i < len(high); i++ {
		push(t, stream, candle{high[i], low[i], closed[i]})
	}
	// A revised high changes the affected TR without invalidating the original close.
	high[correctionIndex] += 8
	want := rationalHistory(high, low, closed, 14)
	mustErrorIs(t, stream.Restore(before), nil)
	for i := correctionIndex; i < len(high); i++ {
		mustValue(t, push(t, stream, candle{high[i], low[i], closed[i]}), atr.Value{TR: want.TR[i], ATR: want.ATR[i], Valid: want.Valid[i]})
	}
	if stream.Snapshot().Count != uint64(len(high)) {
		t.Fatal("revision was counted as an extra event")
	}
}

// TestStream_RejectedInputHasNoAllocations verifies the hot rejection path and rollback.
// The measurement excludes snapshot serialization and the test framework's formatting.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - The measurement excludes snapshot serialization and the test framework's formatting.
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
//   - Preserve exact payload, mask, error identity, and ownership assertions.
func TestStream_RejectedInputHasNoAllocations(t *testing.T) {
	stream := newStream(t, 14)
	push(t, stream, candle{10, 8, 9})
	before := stream.Snapshot()
	ctx := context.Background()
	var got atr.Value
	var err error
	allocations := testing.AllocsPerRun(1000, func() {
		got, err = stream.Push(ctx, 10, 8, 11)
	})
	mustErrorIs(t, err, atr.ErrBar)
	mustValue(t, got, atr.Value{})
	mustSnapshot(t, stream.Snapshot(), before)
	if allocations != 0 {
		t.Fatalf("rejected Push allocated %g objects", allocations)
	}
}

// TestATR_InCodeScenarioOwnership protects independent source and column ownership.
// Editing one scenario or projection must not contaminate later ATR calculations.
//
// Implementation:
//   - Stage 1: construct two equal histories and obtain independent numeric columns.
//   - Stage 2: mutate one history and one projection, then build fresh inputs.
//   - Stage 3: compare the untouched inputs with the rational oracle and public facade.
//
// Behavior highlights:
//   - Nil and allocated-empty histories both produce nonnil empty columns.
//
// Inputs:
//   - Fixed Go-literal session records and a separate empty-history boundary.
//
// Returns:
//   - No value; the owning test fails on aliasing, ordering, or numeric differences.
//
// Errors:
//   - Any unexpected public ATR error fails through mustErrorIs.
//
// Determinism:
//   - Original record order and represented float64 values are compared directly.
//
// Complexity:
//   - O(n) projection work plus the independent rational-history cost, n=20.
//
// Notes:
//   - The constructor does not validate candles on behalf of the public API.
//
// AI-Hints:
//   - Do not replace caller-owned literal construction with a mutable package variable.
//   - A pure projection must not sort records, repair prices, or introduce hidden data.
func TestATR_InCodeScenarioOwnership(t *testing.T) {
	// Two constructors must produce the same values in independent backing arrays.
	first, second := dailyBars(), dailyBars()
	if len(first) != len(second) || len(first) <= atr.DefaultPeriod {
		t.Fatal("scenario must contain warmup and mature observations")
	}
	for i := range first {
		// Session labels remain ordered metadata rather than numerical ATR inputs.
		if first[i].date != second[i].date || (i > 0 && first[i].date <= first[i-1].date) {
			t.Fatalf("scenario label mismatch at %d", i)
		}
		// Compare represented values rather than formatted decimal strings.
		mustFloatBits(t, first[i].high, second[i].high)
		mustFloatBits(t, first[i].low, second[i].low)
		mustFloatBits(t, first[i].close, second[i].close)
	}
	// Build the reference before mutation; expected values never come from production code.
	high, low, closed := dailyChannels(first)
	want := rationalHistory(high, low, closed, atr.DefaultPeriod)
	original := second[0]
	// An edited record must not modify a previously projected numeric column.
	first[0].date = "edited-label"
	first[0].candle = candle{high: -1, low: -3, close: -2}
	mustFloatBits(t, high[0], original.high)
	mustFloatBits(t, low[0], original.low)
	mustFloatBits(t, closed[0], original.close)
	// An edited column must not modify its sibling columns or another input history.
	high[0] = -1
	mustFloatBits(t, low[0], original.low)
	mustFloatBits(t, closed[0], original.close)
	if second[0] != original {
		t.Fatal("input histories share mutable record storage")
	}
	// A new call must be unaffected by both edits and must retain exact ATR outputs.
	high, low, closed = dailyChannels(dailyBars())
	got, err := atr.ComputeDefault(context.Background(), high, low, closed)
	mustErrorIs(t, err, nil)
	mustResult(t, got, want)
	// Empty projection is still a valid input to the documented empty batch path.
	for _, bars := range [][]datedCandle{nil, {}} {
		high, low, closed = dailyChannels(bars)
		if high == nil || low == nil || closed == nil || len(high)+len(low)+len(closed) != 0 {
			t.Fatal("empty projection must own three nonnil empty columns")
		}
		got, err = atr.ComputeDefault(context.Background(), high, low, closed)
		mustErrorIs(t, err, nil)
		mustResult(t, got, rationalHistory(high, low, closed, atr.DefaultPeriod))
	}
}
