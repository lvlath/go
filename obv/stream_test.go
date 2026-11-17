// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package obv_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/obv"
)

func TestOBV05_OverflowAtomicRollback(t *testing.T) {
	s := obv.New()
	for _, b := range []obv.Bar{{1, 1, 1, 1, 2}, {2, 2, 2, 2, math.MaxFloat64}} {
		if _, _, e := s.Push(b); e != nil {
			t.Fatal(e)
		}
	}
	before := s.Snapshot()
	v, valid, e := s.Push(obv.Bar{3, 3, 3, 3, math.MaxFloat64})
	mustErrorIs(t, e, obv.ErrOverflow)
	if v != 0 || valid || !sameSnapshot(before, s.Snapshot()) {
		t.Fatal("partial mutation")
	}
	v, valid, e = s.Push(obv.Bar{1, 1, 1, 1, math.MaxFloat64})
	if e != nil || !valid || v != 0 {
		t.Fatalf("bad recovery: %v %v %v", v, valid, e)
	}
}
func TestOBV07_InvalidBarAtomicRollback(t *testing.T) {
	s := obv.New()
	_, _, _ = s.Push(obv.Bar{1, 1, 1, 1, 0})
	before := s.Snapshot()
	_, _, e := s.Push(obv.Bar{3, 2, 1, 2, 9})
	mustErrorIs(t, e, obv.ErrDomain)
	if !sameSnapshot(before, s.Snapshot()) {
		t.Fatal("state changed")
	}
}
func TestBatchStreamBitParity(t *testing.T) {
	c := []float64{2, 4, 4, 3, 8, 1, 9, 7}
	v := []float64{99, 1e16, 0, 1e16, 1, 1e16, 1e16, 1}
	b := bars(c, v)
	r, e := obv.Compute(b)
	if e != nil {
		t.Fatal(e)
	}
	var s obv.Stream
	for i := range c {
		x, valid, e := s.Push(obv.Bar{c[i], c[i], c[i], c[i], v[i]})
		if e != nil || !valid || math.Float64bits(x) != math.Float64bits(r.Values[i]) {
			t.Fatalf("divergence at %d: %v %v", i, x, e)
		}
	}
}
func TestNilAndReset(t *testing.T) {
	var s *obv.Stream
	_, _, e := s.Push(obv.Bar{1, 1, 1, 1, 0})
	mustErrorIs(t, e, obv.ErrNilInput)
	if s.Snapshot().Count != 0 {
		t.Fatal("nil snapshot")
	}
	s.Reset()
	var z obv.Stream
	_, valid, e := z.Push(obv.Bar{1, 1, 1, 1, 55})
	if e != nil || !valid {
		t.Fatal(e)
	}
	z.Reset()
	if z.Snapshot().Count != 0 {
		t.Fatal("reset failed")
	}
}
func TestStreamZeroAllocsPush(t *testing.T) {
	s := obv.New()
	b := obv.Bar{1, 1, 1, 1, 0}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, e := s.Push(b)
		if e != nil {
			t.Fatal(e)
		}
	})
	if allocs != 0 {
		t.Fatalf("Push allocations: %g", allocs)
	}
}
