// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File stream_test.go asserts parity, failure rollback and allocation discipline.
package rsi_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/rsi"
)

// TestStream_BatchParityAndReset compares every sample bitwise over long deterministic traces.
// The second pass verifies a complete re-warmup after Reset under the same options.
func TestStream_BatchParityAndReset(t *testing.T) {
	for _, period := range []int{1, 2, 7, 14, 1000} {
		closeR := make([]float64, 2400)
		for i := range closeR {
			closeR[i] = 2000 + float64((i*7919)%129) - float64((i*17)%51)
		}
		stream, e := rsi.New(rsi.Options{Period: period})
		if e != nil {
			t.Fatal(e)
		}
		batch, e := rsi.Compute(closeR, rsi.Options{Period: period})
		if e != nil {
			t.Fatal(e)
		}
		for i, x := range closeR {
			v, valid, err := stream.Push(x)
			if err != nil {
				t.Fatal(err)
			}
			mustSameFloat(t, v, batch.Values[i])
			mustEqualBool(t, valid, batch.Valid[i], "batch/stream")
		}
		stream.Reset()
		for i := 0; i < 20; i++ {
			v, valid, e := stream.Push(closeR[i])
			if e != nil {
				t.Fatal(e)
			}
			mustSameFloat(t, v, batch.Values[i])
			mustEqualBool(t, valid, batch.Valid[i], "reset parity")
		}
	}
}

// TestRSI06FailedPushRollback rejects nonfinite/nonpositive quotes without changing checkpoint bits.
// Continuing with a valid quote must match a reference stream without the rejected event.
func TestRSI06FailedPushRollback(t *testing.T) {
	s, e := rsi.New(rsi.Options{Period: 2})
	if e != nil {
		t.Fatal(e)
	}
	for _, x := range []float64{3, 2, 1, 2} {
		if _, _, e = s.Push(x); e != nil {
			t.Fatal(e)
		}
	}
	before := s.Snapshot()
	for _, tc := range []struct {
		x   float64
		err error
	}{{math.NaN(), rsi.ErrNonFinite}, {math.Inf(1), rsi.ErrNonFinite}, {0, rsi.ErrDomain}, {-1, rsi.ErrDomain}} {
		v, valid, err := s.Push(tc.x)
		mustErrorIs(t, err, tc.err)
		mustSameFloat(t, v, 0)
		mustEqualBool(t, valid, false, "failed valid")
		after := s.Snapshot()
		assertSnapshotEqual(t, before, after)
	}
	ref, e := rsi.New(rsi.Options{Period: 2})
	if e != nil {
		t.Fatal(e)
	}
	if e = ref.Restore(before); e != nil {
		t.Fatal(e)
	}
	a, va, e := s.Push(3)
	if e != nil {
		t.Fatal(e)
	}
	b, vb, e := ref.Push(3)
	if e != nil {
		t.Fatal(e)
	}
	mustSameFloat(t, a, b)
	mustEqualBool(t, va, vb, "continuation")
}

// TestStream_ZeroAllocsPush measures the warmed production hot path without an allocation wrapper.
func TestStream_ZeroAllocsPush(t *testing.T) {
	s, e := rsi.New(rsi.Options{Period: 14})
	if e != nil {
		t.Fatal(e)
	}
	for i := 0; i < 40; i++ {
		if _, _, e = s.Push(100 + float64(i%7)); e != nil {
			t.Fatal(e)
		}
	}
	allocations := testing.AllocsPerRun(1000, func() {
		_, _, e = s.Push(100)
		if e != nil {
			t.Fatal(e)
		}
	})
	if allocations != 0 {
		t.Fatalf("Push allocations/op=%v, want 0", allocations)
	}
}

// TestNilAndZeroStream anchors the deliberately different nil and uninitialized receiver semantics.
func TestNilAndZeroStream(t *testing.T) {
	var ptr *rsi.Stream
	_, _, e := ptr.Push(1)
	mustErrorIs(t, e, rsi.ErrNilInput)
	mustErrorIs(t, ptr.Restore(rsi.Snapshot{}), rsi.ErrNilInput)
	ptr.Reset()
	if ptr.Snapshot().Algorithm != "" {
		t.Fatal("nil snapshot not zero")
	}
	var zero rsi.Stream
	_, _, e = zero.Push(1)
	mustErrorIs(t, e, rsi.ErrStateVersion)
	mustErrorIs(t, zero.Restore(rsi.Snapshot{}), rsi.ErrStateVersion)
}
