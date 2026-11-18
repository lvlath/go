// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad_test exercises the exported, transactional, zero-allocation Stream contract.
// Every failure is checked against detached snapshots rather than internal fields.
package ad_test

import (
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/ad"
)

// TestStream_BatchBitParity verifies every value of a nontrivial, variable-pressure history.
func TestStream_BatchBitParity(t *testing.T) {
	candles := append(standardBars(), variedBars(513)...)
	columns := barsFrom(candles)
	batch, err := ad.Compute(columns)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := ad.ComputeCandles(candles)
	if err != nil {
		t.Fatal(err)
	}
	stream := ad.New()
	var zeroValue ad.Stream

	for index, bar := range candles {
		value, valid, pushErr := stream.Push(bar)
		if pushErr != nil {
			t.Fatalf("Push[%d]: %v", index, pushErr)
		}
		zeroValueResult, zeroValueValid, zeroValueErr := zeroValue.Push(bar)
		if zeroValueErr != nil {
			t.Fatalf("zero value Push[%d]: %v", index, zeroValueErr)
		}
		mustEqualBool(t, valid, true, "stream validity")
		mustEqualBool(t, zeroValueValid, true, "zero-value validity")
		mustBits(t, value, batch.Values[index])
		mustBits(t, rows.Values[index], batch.Values[index])
		mustBits(t, zeroValueResult, value)
	}
	mustSnapshotEqual(t, stream.Snapshot(), zeroValue.Snapshot())
}

// TestStream_FlatAndZeroVolumeStillAdvanceCount anchors explicit observation counting.
func TestStream_FlatAndZeroVolumeStillAdvanceCount(t *testing.T) {
	stream := ad.New()
	candles := []ad.Bar{
		{Open: 100, High: 100, Low: 100, Close: 100, Volume: math.MaxFloat64},
		{Open: 100, High: 102, Low: 98, Close: 98, Volume: 0},
		{Open: 100, High: 102, Low: 98, Close: 100, Volume: 777},
		{Open: 100, High: 102, Low: 98, Close: 102, Volume: 400},
	}
	for index, bar := range candles {
		value, valid, err := stream.Push(bar)
		if err != nil {
			t.Fatal(err)
		}
		mustEqualBool(t, valid, true, "valid even for zero contribution")
		if index < len(candles)-1 {
			mustBits(t, value, 0)
		} else {
			mustBits(t, value, 400)
		}
		if stream.Snapshot().Count != uint64(index+1) {
			t.Fatalf("after bar %d count=%d, want %d", index, stream.Snapshot().Count, index+1)
		}
	}
}

// TestStream_AD08_AtomicBadCandles proves domain and nonfinite failures preserve all state.
func TestStream_AD08_AtomicBadCandles(t *testing.T) {
	stream := ad.New()
	if _, _, err := stream.Push(standardBars()[0]); err != nil {
		t.Fatal(err)
	}
	before := stream.Snapshot()
	invalidOpen := standardBars()[0]
	invalidOpen.Open = 0
	invalidClose := standardBars()[0]
	invalidClose.Close = 100
	nonfinite := standardBars()[0]
	nonfinite.Volume = math.Inf(1)

	for _, tc := range []struct {
		name string
		bar  ad.Bar
		want error
	}{
		{"openBelowRange", invalidOpen, ad.ErrDomain},
		{"closeAboveRange", invalidClose, ad.ErrDomain},
		{"infiniteVolume", nonfinite, ad.ErrNonFinite},
	} {
		t.Run(tc.name, func(t *testing.T) {
			value, valid, err := stream.Push(tc.bar)
			mustErrorIs(t, err, tc.want)
			mustEqualBool(t, valid, false, "failure validity")
			mustBits(t, value, 0)
			mustSnapshotEqual(t, stream.Snapshot(), before)
		})
	}

	// A later valid candle must produce the same output as if no rejected
	// observations had been presented in between the accepted bars.
	value, valid, err := stream.Push(standardBars()[1])
	if err != nil {
		t.Fatal(err)
	}
	mustEqualBool(t, valid, true, "valid continuation")
	mustBits(t, value, 1)
}

// TestStream_AtomicOverflow rejects two positive MaxFloat64 contributions with no mutation.
func TestStream_AtomicOverflow(t *testing.T) {
	bar := ad.Bar{Open: 1.5, High: 2, Low: 1, Close: 2, Volume: math.MaxFloat64}
	stream := ad.New()
	first, valid, err := stream.Push(bar)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualBool(t, valid, true, "first accepted")
	mustBits(t, first, math.MaxFloat64)
	before := stream.Snapshot()

	value, valid, err := stream.Push(bar)
	mustErrorIs(t, err, ad.ErrOverflow)
	mustEqualBool(t, valid, false, "overflow valid")
	mustBits(t, value, 0)
	mustSnapshotEqual(t, stream.Snapshot(), before)

	// Batch overflow returns no partial result, even though the first bar
	// independently fits into float64.
	result, err := ad.ComputeCandles([]ad.Bar{bar, bar})
	mustErrorIs(t, err, ad.ErrOverflow)
	mustNilState(t, result, true, "batch overflow")
	result, err = ad.Compute(barsFrom([]ad.Bar{bar, bar}))
	mustErrorIs(t, err, ad.ErrOverflow)
	mustNilState(t, result, true, "column overflow")

	// A negative contribution can still exactly cancel the accepted first bar.
	bar.Close = bar.Low
	next, valid, err := stream.Push(bar)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualBool(t, valid, true, "canceling bar")
	mustBits(t, next, 0)
}

// TestStream_NilResetAndCounterExhaustion protects receiver behavior and Count arithmetic.
func TestStream_NilResetAndCounterExhaustion(t *testing.T) {
	var missing *ad.Stream
	value, valid, err := missing.Push(standardBars()[0])
	mustErrorIs(t, err, ad.ErrNilInput)
	mustBits(t, value, 0)
	mustEqualBool(t, valid, false, "nil stream valid")
	missing.Reset() // Exactly the documented safe nil no-op.
	mustSnapshotEqual(t, missing.Snapshot(), ad.Snapshot{})

	stream := ad.New()
	state := stream.Snapshot()
	state.Count = math.MaxUint64
	state.FloatState[0] = 1
	if err := stream.Restore(state); err != nil {
		t.Fatal(err)
	}
	before := stream.Snapshot()
	value, valid, err = stream.Push(standardBars()[0])
	mustErrorIs(t, err, ad.ErrOverflow)
	mustEqualBool(t, valid, false, "exhausted valid")
	mustBits(t, value, 0)
	mustSnapshotEqual(t, stream.Snapshot(), before)

	// Single-bar input failures retain their error priority, even when the
	// counter cannot accept an additional valid observation.
	invalid := standardBars()[0]
	invalid.Open = math.NaN()
	_, _, err = stream.Push(invalid)
	mustErrorIs(t, err, ad.ErrNonFinite)
	mustSnapshotEqual(t, stream.Snapshot(), before)

	stream.Reset()
	mustSnapshotEqual(t, stream.Snapshot(), ad.New().Snapshot())
	value, valid, err = stream.Push(standardBars()[0])
	if err != nil {
		t.Fatal(err)
	}
	mustBits(t, value, 3)
	mustEqualBool(t, valid, true, "post-reset valid")
}

// TestStream_RebaseAfterCancelingCheckpointCorrection anchors representable overflow recovery.
func TestStream_RebaseAfterCancelingCheckpointCorrection(t *testing.T) {
	// Both coordinates are finite and the effective imported total is zero.
	// Adding MaxFloat64 directly to the high component overflows, but folding
	// its opposing correction first makes the true next state representable.
	stream := ad.New()
	checkpoint := stream.Snapshot()
	checkpoint.Count = 2
	checkpoint.FloatState[0] = math.MaxFloat64
	checkpoint.FloatState[1] = -math.MaxFloat64
	if err := stream.Restore(checkpoint); err != nil {
		t.Fatal(err)
	}
	bar := ad.Bar{Open: 1.5, High: 2, Low: 1, Close: 2, Volume: math.MaxFloat64}
	value, valid, err := stream.Push(bar)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualBool(t, valid, true, "finite rebase accepted")
	mustBits(t, value, math.MaxFloat64)
	if stream.Snapshot().Count != 3 {
		t.Fatal("rebased transition must advance Count")
	}
}

// TestStream_ZeroAllocsPush checks actual signed nonzero contributions, not a constant zero path.
func TestStream_ZeroAllocsPush(t *testing.T) {
	candles := []ad.Bar{
		{Open: 100, Low: 99, High: 103, Close: 103, Volume: 1000},
		{Open: 100, Low: 99, High: 103, Close: 99, Volume: 999},
		{Open: 100, Low: 99, High: 103, Close: 102, Volume: 888},
	}
	stream := ad.New()
	index := 0
	var pushErr error
	allocations := testing.AllocsPerRun(10_000, func() {
		_, _, pushErr = stream.Push(candles[index%len(candles)])
		index++
	})
	if pushErr != nil {
		t.Fatal(pushErr)
	}
	if allocations != 0 {
		t.Fatalf("Push allocations/op = %.2f; contract requires exactly 0", allocations)
	}
}

// TestStream_IndependentConcurrency runs two immutable batch readers and many independent engines.
func TestStream_IndependentConcurrency(t *testing.T) {
	const workers = 12
	candles := variedBars(512)
	columns := barsFrom(candles)
	baseline, err := ad.Compute(columns)
	if err != nil {
		t.Fatal(err)
	}
	var wait sync.WaitGroup
	errorsFromWorkers := make(chan error, workers)
	for worker := 0; worker < workers; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			var stream ad.Stream
			batch, computeErr := ad.Compute(columns) // Immutable shared input only.
			if computeErr != nil {
				errorsFromWorkers <- computeErr
				return
			}
			for index, bar := range candles {
				value, valid, pushErr := stream.Push(bar)
				if pushErr != nil {
					errorsFromWorkers <- pushErr
					return
				}
				if !valid || math.Float64bits(value) != math.Float64bits(baseline.Values[index]) ||
					math.Float64bits(batch.Values[index]) != math.Float64bits(value) {
					errorsFromWorkers <- ad.ErrDomain // Difference is reported only to the owner test goroutine.
					return
				}
			}
		}()
	}
	wait.Wait()
	close(errorsFromWorkers)
	for workerErr := range errorsFromWorkers {
		t.Errorf("independent stream or read-only batch drift: %v", workerErr)
	}
}
