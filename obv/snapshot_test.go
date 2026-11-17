// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package obv_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/obv"
)

func TestOBV08_RestoreAndSnapshotDetached(t *testing.T) {
	s := obv.New()
	states := []obv.Snapshot{s.Snapshot()}
	seq := []obv.Bar{{1, 1, 1, 1, 0}, {2, 2, 2, 2, 1e16}, {3, 3, 3, 3, 1}, {2, 2, 2, 2, 1e16}}
	for _, b := range seq {
		_, _, e := s.Push(b)
		if e != nil {
			t.Fatal(e)
		}
		states = append(states, s.Snapshot())
	}
	if states[0].Count != 0 || states[1].Count != 1 || states[1].FloatState[1] != 0 || states[4].FloatState[1]+states[4].FloatState[2] != 1 {
		t.Fatal("bad snapshots")
	}
	for _, snap := range states {
		var restored obv.Stream
		if e := restored.Restore(snap); e != nil {
			t.Fatal(e)
		}
		if !sameSnapshot(snap, restored.Snapshot()) {
			t.Fatal("roundtrip mismatch")
		}
		follow := obv.Bar{4, 4, 4, 4, 5}
		a := obv.New()
		if e := a.Restore(snap); e != nil {
			t.Fatal(e)
		}
		av, _, ae := a.Push(follow)
		bv, _, be := restored.Push(follow)
		if ae != nil || be != nil || math.Float64bits(av) != math.Float64bits(bv) {
			t.Fatal("continuation mismatch")
		}
	}
	x := s.Snapshot()
	x.FloatState[1] = 999
	if sameSnapshot(x, s.Snapshot()) {
		t.Fatal("snapshot aliases internal memory")
	}
	r := obv.New()
	original := s.Snapshot()
	if e := r.Restore(original); e != nil {
		t.Fatal(e)
	}
	original.FloatState[1] = 999
	if r.Snapshot().FloatState[1] == 999 {
		t.Fatal("restore retained caller memory")
	}
}
func TestRestoreRejectsCorruptStateAtomically(t *testing.T) {
	s := obv.New()
	_, _, _ = s.Push(obv.Bar{1, 1, 1, 1, 10})
	good := s.Snapshot()
	cases := []obv.Snapshot{{}, good, good, good, good, good, good}
	cases[1].Algorithm = "other"
	cases[2].Config = "forged"
	cases[3].FloatState = []float64{1, 0}
	cases[4].FloatState = []float64{1, math.Inf(1), 0}
	cases[5].FloatState = []float64{0, 0, 0}
	cases[6].IntState = []uint64{1}
	for i, snap := range cases {
		e := s.Restore(snap)
		mustErrorIs(t, e, obv.ErrStateVersion)
		if !sameSnapshot(good, s.Snapshot()) {
			t.Fatalf("mutation at %d", i)
		}
	}
	var nilStream *obv.Stream
	mustErrorIs(t, nilStream.Restore(good), obv.ErrNilInput)
}
func TestRestoreMaxCount(t *testing.T) {
	s := obv.New()
	_, _, _ = s.Push(obv.Bar{1, 1, 1, 1, 0})
	snap := s.Snapshot()
	snap.Count = math.MaxUint64
	snap.FloatState[1] = 1
	if e := s.Restore(snap); e != nil {
		t.Fatal(e)
	}
	before := s.Snapshot()
	_, _, e := s.Push(obv.Bar{2, 2, 2, 2, 1})
	mustErrorIs(t, e, obv.ErrOverflow)
	if !sameSnapshot(before, s.Snapshot()) {
		t.Fatal("count overflow mutated state")
	}
}
