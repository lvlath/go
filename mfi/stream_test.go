// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Stream acceptance: exact batch parity, atomic failures, bounded allocations.
package mfi_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/mfi"
)

func TestStream_BatchParityAndReset(t *testing.T) {
	const n = 2200
	prices, volumes := make([]float64, n), make([]float64, n)
	for i := 0; i < n; i++ {
		prices[i] = float64(100 + i%29)
		volumes[i] = float64(i % 13)
	}
	bars := sampleBars(prices, volumes)
	batch, err := mfi.Compute(bars, mfi.Options{Period: 31})
	if err != nil {
		t.Fatal(err)
	}
	s, err := mfi.New(mfi.Options{Period: 31})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		v, ok, e := s.Push(sliceBars(bars, i))
		if e != nil {
			t.Fatal(e)
		}
		if math.Float64bits(v) != math.Float64bits(batch.Values[i]) || ok != batch.Valid[i] {
			t.Fatalf("divergence at %d", i)
		}
	}
	s.Reset()
	if s.Snapshot().Count != 0 {
		t.Fatal("reset count")
	}
}
func TestStream_InvalidPushAtomic(t *testing.T) {
	s, err := mfi.New(mfi.Options{Period: 2})
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range []float64{1, 2, 1} {
		if _, _, e := s.Push(sampleBar(v, 1)); e != nil {
			t.Fatal(e)
		}
	}
	before := s.Snapshot()
	bad := sampleBar(2, -1)
	_, valid, e := s.Push(bad)
	mustErrorIs(t, e, mfi.ErrDomain)
	mustEqualBool(t, valid, false, "bad valid")
	after := s.Snapshot()
	mustEqualInt(t, int(after.Count), int(before.Count), "count")
	for i := range before.FloatState {
		if math.Float64bits(before.FloatState[i]) != math.Float64bits(after.FloatState[i]) {
			t.Fatalf("float changed at %d", i)
		}
	}
	for i := range before.IntState {
		if before.IntState[i] != after.IntState[i] {
			t.Fatalf("int changed at %d", i)
		}
	}
}
func TestStream_NilAndZeroReceivers(t *testing.T) {
	var nilStream *mfi.Stream
	_, _, e := nilStream.Push(sampleBar(1, 1))
	mustErrorIs(t, e, mfi.ErrNilInput)
	mustErrorIs(t, nilStream.Restore(mfi.Snapshot{}), mfi.ErrNilInput)
	nilStream.Reset()
	if nilStream.Snapshot().Algorithm != "" {
		t.Fatal("nil snapshot")
	}
	zero := &mfi.Stream{}
	_, _, e = zero.Push(sampleBar(1, 1))
	mustErrorIs(t, e, mfi.ErrStateVersion)
	mustErrorIs(t, zero.Restore(mfi.Snapshot{}), mfi.ErrStateVersion)
}
func TestStream_AmortizedConstantPush(t *testing.T) {
	for _, period := range []int{14, 1000} {
		s, e := mfi.New(mfi.Options{Period: period})
		if e != nil {
			t.Fatal(e)
		}
		allocs := testing.AllocsPerRun(100, func() {
			_, _, err := s.Push(sampleBar(100, 1))
			if err != nil {
				t.Fatal(err)
			}
		})
		if allocs != 0 {
			t.Fatalf("period=%d Push allocations=%g", period, allocs)
		}
	}
}
func FuzzMFIStreamParity(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0})
	f.Fuzz(func(t *testing.T, bs []byte) {
		if len(bs) > 256 {
			return
		}
		p := 1 + len(bs)%17
		s, e := mfi.New(mfi.Options{Period: p})
		if e != nil {
			t.Fatal(e)
		}
		prices, volumes := make([]float64, len(bs)+1), make([]float64, len(bs)+1)
		prices[0] = 100
		for i, x := range bs {
			prices[i+1] = float64(100 + int(x)%37)
			volumes[i+1] = float64(int(x) % 7)
		}
		bars := sampleBars(prices, volumes)
		batch, e := mfi.Compute(bars, mfi.Options{Period: p})
		if e != nil {
			t.Fatal(e)
		}
		for i := range prices {
			value, valid, err := s.Push(sliceBars(bars, i))
			if err != nil {
				t.Fatal(err)
			}
			if valid != batch.Valid[i] || math.Float64bits(value) != math.Float64bits(batch.Values[i]) {
				t.Fatalf("mismatch at %d", i)
			}
		}
	})
}
