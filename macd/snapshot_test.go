// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd_test pins detached snapshots, phased seed recovery,
// version/config admission and deterministic state replay through public APIs.
package macd_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/macd"
)

// TestMACD06_SnapshotRestoreAllPhases Splits histories around the three EMA seed boundaries and checks bitwise continuation.
func TestMACD06_SnapshotRestoreAllPhases(t *testing.T) {
	v := []float64{1, 4, 2, 5, 3, 7, 2, 8, 5, 6, 10, 4}
	o := macd.Options{Fast: 2, Slow: 4, Signal: 3}
	for _, split := range []int{0, 1, 2, 3, 4, 6, 7, 12} {
		a, _ := macd.New(o)
		for _, x := range v[:split] {
			if _, e := a.Push(x); e != nil {
				t.Fatal(e)
			}
		}
		snap := a.Snapshot()
		b, _ := macd.New(o)
		if e := b.Restore(snap); e != nil {
			t.Fatal(split, e)
		}
		for _, x := range v[split:] {
			p, e := a.Push(x)
			if e != nil {
				t.Fatal(e)
			}
			q, e := b.Push(x)
			if e != nil {
				t.Fatal(e)
			}
			mustBits(t, p.Line, q.Line, "line")
			mustBits(t, p.Signal, q.Signal, "signal")
			mustBits(t, p.Histogram, q.Histogram, "hist")
			mustEqualBool(t, p.LineValid, q.LineValid, "line mask")
			mustEqualBool(t, p.SignalValid, q.SignalValid, "signal mask")
		}
		mustSnapshotsEqual(t, a.Snapshot(), b.Snapshot())
	}
}

// TestSnapshot_ValidationAndDetached Rejects corrupt state without mutation and proves exported snapshots never alias the live stream.
func TestSnapshot_ValidationAndDetached(t *testing.T) {
	a, _ := macd.New(macd.Options{Fast: 2, Slow: 4, Signal: 3})
	for _, x := range []float64{1, 2, 3, 4, 5, 6, 7} {
		if _, e := a.Push(x); e != nil {
			t.Fatal(e)
		}
	}
	original := a.Snapshot()
	mutation := a.Snapshot()
	mutation.FloatState[0] = math.NaN()
	b, _ := macd.New(macd.Options{Fast: 2, Slow: 4, Signal: 3})
	before := b.Snapshot()
	mustErrorIs(t, b.Restore(mutation), macd.ErrStateVersion)
	mustSnapshotsEqual(t, before, b.Snapshot())
	mustSnapshotsEqual(t, original, a.Snapshot())
	mutation = a.Snapshot()
	mutation.IntState[0] = 0
	mustErrorIs(t, b.Restore(mutation), macd.ErrStateVersion)
	mutation = a.Snapshot()
	mutation.Config = "bad"
	mustErrorIs(t, b.Restore(mutation), macd.ErrStateVersion)
	foreign, _ := macd.New(macd.Options{})
	mustErrorIs(t, foreign.Restore(original), macd.ErrStateVersion)
	if e := b.Restore(original); e != nil {
		t.Fatal(e)
	}
	mustSnapshotsEqual(t, original, b.Snapshot())
	original.FloatState[0] = 99
	original.IntState[0] = 0
	mustSnapshotsEqual(t, a.Snapshot(), b.Snapshot())
	b.Reset()
	if b.Snapshot().Count != 0 {
		t.Fatal("reset count")
	}
}

// TestSnapshot_CounterOverflowAtomicity Restores a saturated count and proves Push rejects overflow without changing state.
func TestSnapshot_CounterOverflowAtomicity(t *testing.T) {
	s, err := macd.New(macd.Options{Fast: 2, Slow: 3, Signal: 2})
	if err != nil {
		t.Fatal(err)
	}
	snapshot := s.Snapshot()
	snapshot.Count = ^uint64(0)
	snapshot.IntState = []uint64{2, 3, 2}
	snapshot.FloatState = []float64{100, 100, 0}
	if err = s.Restore(snapshot); err != nil {
		t.Fatal(err)
	}
	before := s.Snapshot()
	_, err = s.Push(100)
	mustErrorIs(t, err, macd.ErrOverflow)
	mustSnapshotsEqual(t, before, s.Snapshot())
}

// TestSnapshot_InvalidPhaseAndSignedSignal Confirms negative signal EMA state is valid while inconsistent phase counts are not.
func TestSnapshot_InvalidPhaseAndSignedSignal(t *testing.T) {
	s, _ := macd.New(macd.Options{Fast: 2, Slow: 3, Signal: 2})
	for _, v := range []float64{9, 8, 7, 6} {
		if _, err := s.Push(v); err != nil {
			t.Fatal(err)
		}
	}
	state := s.Snapshot()
	if state.FloatState[2] >= 0 {
		t.Fatalf("expected negative signal mean; got %g", state.FloatState[2])
	}
	clone, _ := macd.New(macd.Options{Fast: 2, Slow: 3, Signal: 2})
	if err := clone.Restore(state); err != nil {
		t.Fatal(err)
	}
	bad := s.Snapshot()
	bad.IntState[2]--
	before := clone.Snapshot()
	mustErrorIs(t, clone.Restore(bad), macd.ErrStateVersion)
	mustSnapshotsEqual(t, before, clone.Snapshot())
}

// FuzzSnapshot_AtomicRejectNoPanic explores malformed lengths, counters,
// float signs and version identity without touching private state or using reflect.
func FuzzSnapshot_AtomicRejectNoPanic(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3, 4, 5})
	f.Add([]byte{255, 0, 17, 7, 254, 19, 66})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 || len(data) > 256 {
			return
		}
		stream, err := macd.NewWithPeriods(2, 4, 3)
		if err != nil {
			t.Fatal(err)
		}
		for _, price := range []float64{10, 9, 12, 11, 8, 13} {
			if _, err = stream.Push(price); err != nil {
				t.Fatal(err)
			}
		}
		before := stream.Snapshot()
		invalid := stream.Snapshot()
		for index, token := range data {
			switch index % 6 {
			case 0:
				invalid.Count += uint64(token)
			case 1:
				invalid.IntState[0] = uint64(token)
			case 2:
				invalid.FloatState[1] = float64(int(token) - 128)
			case 3:
				invalid.FloatState[2] = float64(int(token) - 128)
			case 4:
				invalid.Config = "bad"
			case 5:
				invalid.Algorithm = "legacy"
			}
		}
		// Restore can succeed for a numerically plausible perturbation; failure
		// must leave the stream fully unchanged and success must be roundtrippable.
		restoreErr := stream.Restore(invalid)
		if restoreErr != nil {
			mustErrorIs(t, restoreErr, macd.ErrStateVersion)
			mustSnapshotsEqual(t, before, stream.Snapshot())
			return
		}
		mustSnapshotsEqual(t, invalid, stream.Snapshot())
	})
}

// TestSnapshot_ConfigDigestIndependentVectors pins the published SHA-256 input
// wire format using independently computed ASCII+LE64 golden digests.
// The constants were generated independently of configHash, not by replaying
// the package helper under test.
func TestSnapshot_ConfigDigestIndependentVectors(t *testing.T) {
	for _, tc := range []struct {
		opts   macd.Options
		config string
	}{
		{
			opts:   macd.Options{},
			config: "05e527ad85a2a33a6723537e3fb700be704df27ee733487e4e7a12d9b24b4191",
		},
		{
			opts:   macd.Options{Fast: 2, Slow: 3, Signal: 2},
			config: "b4f0d77f86ca4c46b5a9d9c248f59c22f0bb9e018537aadd94c4d510f4ad7e74",
		},
		{
			opts:   macd.Options{Fast: 5, Slow: 35, Signal: 5},
			config: "fcacaff906dbfdca40892d916b7d7c5d41589bb1152fe39b135a0cce1c20a5a9",
		},
	} {
		stream, err := macd.New(tc.opts)
		if err != nil {
			t.Fatal(err)
		}
		state := stream.Snapshot()
		mustEqualString(t, state.Algorithm, "macd/sma-ema/v1", "algorithm")
		mustEqualString(t, state.Config, tc.config, "independent config SHA-256")
		if len(state.FloatState) != 3 || len(state.IntState) != 3 || state.Count != 0 {
			t.Fatalf("uninitialized snapshot invariant failed: %+v", state)
		}
	}
}
