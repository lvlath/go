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
			config: "445b75fd55d24a7e556a9028d92e9bff144f4c8bc4b16b7cea0425c9f7655e21",
		},
		{
			opts:   macd.Options{Fast: 2, Slow: 3, Signal: 2},
			config: "ccb5a820c2b78df6b8a658d48698605418b2de7c14f116b355354dec4896ba70",
		},
		{
			opts:   macd.Options{Fast: 5, Slow: 35, Signal: 5},
			config: "36e83efa7221ab417a25468bdbf223b303d279d2530c514da92cc0dc22d70458",
		},
	} {
		stream, err := macd.New(tc.opts)
		if err != nil {
			t.Fatal(err)
		}
		state := stream.Snapshot()
		mustEqualString(t, state.Algorithm, "macd/sma-ema/v2", "algorithm")
		mustEqualString(t, state.Config, tc.config, "independent config SHA-256")
		if len(state.FloatState) != 3 || len(state.IntState) != 3 || state.Count != 0 {
			t.Fatalf("uninitialized snapshot invariant failed: %+v", state)
		}
	}
}

// TestSnapshot_V1_Rejected pins the incompatible numeric identity and tests
// the only supported migration: replay the original accepted price prefix.
// A legitimate old checkpoint and a relabeled old digest must both fail.
//
// Implementation:
//   - Stage 1: build a v2 stream, independently compute v1's known config hash
//     and supply the exact v1 state for a period-one cancellation fixture.
//   - Stage 2: reject v1, mismatched hashes, and a label-only replacement.
//   - Stage 3: replay the original prefix, then restore and compare continuation.
//
// Determinism:
//   - The old fixture records exact binary64 values and a literal v1 SHA-256.
//   - v2 is produced only by New/Push, never by relabeling legacy state.
//
// Complexity:
//   - Time O(n) for replay, Space O(1) for the fixed-size stream and checkpoints.
//
// AI-Hints:
//   - Reject a v1 checkpoint even when no period equals one; schema identity is
//     unconditional, and the digest is not authentication of numeric history.
func TestSnapshot_V1_Rejected(t *testing.T) {
	// This benign v1 checkpoint is compatible in shape and all numeric fields,
	// but version admission must reject it even with ordinary periods.
	ordinary, err := macd.NewWithPeriods(2, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, price := range []float64{1, 2, 3, 2} {
		if _, err = ordinary.Push(price); err != nil {
			t.Fatal(err)
		}
	}
	prior := ordinary.Snapshot()
	const legacyDigest = "b4f0d77f86ca4c46b5a9d9c248f59c22f0bb9e018537aadd94c4d510f4ad7e74"
	legacy := ordinary.Snapshot()
	legacy.Algorithm, legacy.Config = "macd/sma-ema/v1", legacyDigest
	mustErrorIs(t, ordinary.Restore(legacy), macd.ErrStateVersion)
	mustSnapshotsEqual(t, prior, ordinary.Snapshot())
	// Neither half of the identifier/digest pair may come from v1.
	legacy.Algorithm = prior.Algorithm
	mustErrorIs(t, ordinary.Restore(legacy), macd.ErrStateVersion)
	mustSnapshotsEqual(t, prior, ordinary.Snapshot())
	legacy.Algorithm, legacy.Config = "macd/sma-ema/v1", prior.Config
	mustErrorIs(t, ordinary.Restore(legacy), macd.ErrStateVersion)
	mustSnapshotsEqual(t, prior, ordinary.Snapshot())

	// Recover a historically corrupted Fast=1 state from the original closes,
	// not from the lost mean or a substituted epsilon.
	const first = 1e308
	const second = 1.0
	active, err := macd.NewWithPeriods(1, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	broken := macd.Snapshot{
		Algorithm:  "macd/sma-ema/v1",
		Config:     "0e353be9329d7065c27341cabfa1e3b7eb4900ef73c8002adddaf60932311e61",
		Count:      2,
		FloatState: []float64{0, 5e307, -5e307},
		IntState:   []uint64{1, 2, 1},
	}
	initial := active.Snapshot()
	mustErrorIs(t, active.Restore(broken), macd.ErrStateVersion)
	mustSnapshotsEqual(t, initial, active.Snapshot())
	for _, price := range []float64{first, second} {
		if _, err = active.Push(price); err != nil {
			t.Fatal(err)
		}
	}
	corrected := active.Snapshot()
	mustBits(t, corrected.FloatState[0], second, "replayed exact Fast")
	mustEqualString(t, corrected.Algorithm, "macd/sma-ema/v2", "new algorithm")
	standby, err := macd.NewWithPeriods(1, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err = standby.Restore(corrected); err != nil {
		t.Fatal(err)
	}
	for _, price := range []float64{3, 1e100, 4} {
		point, pushErr := active.Push(price)
		if pushErr != nil {
			t.Fatal(pushErr)
		}
		want, pushErr := standby.Push(price)
		if pushErr != nil {
			t.Fatal(pushErr)
		}
		mustMACDPointsEqual(t, point, want)
	}
	mustSnapshotsEqual(t, active.Snapshot(), standby.Snapshot())
}
