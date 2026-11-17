// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Versioned Snapshot ABI adversarial tests use only the exported mfi surface.
package mfi_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/mfi"
)

func TestSnapshot_RejectsCorruptionAtomically(t *testing.T) {
	stream, err := mfi.NewPeriod(4)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 13; i++ {
		price := float64(100 + (i*13)%19)
		if _, _, err = stream.Push(sampleBar(price, float64(i%6+1))); err != nil {
			t.Fatal(err)
		}
	}
	original := stream.Snapshot()
	const ringHeader = 5
	const period = 4

	cases := []struct {
		name   string
		mutate func(*mfi.Snapshot)
	}{
		{"algorithm", func(s *mfi.Snapshot) { s.Algorithm = "mfi/rolling-neumaier/v2" }},
		{"config", func(s *mfi.Snapshot) { s.Config = "0000" }},
		{"float-shape", func(s *mfi.Snapshot) { s.FloatState = s.FloatState[:len(s.FloatState)-1] }},
		{"int-shape", func(s *mfi.Snapshot) { s.IntState = s.IntState[:2] }},
		{"count", func(s *mfi.Snapshot) { s.Count = 0 }},
		{"used", func(s *mfi.Snapshot) { s.IntState[0] = 0 }},
		{"head", func(s *mfi.Snapshot) { s.IntState[1]++ }},
		{"gosec-overflow-used", func(s *mfi.Snapshot) { s.IntState[0] = math.MaxUint64 }},
		{"gosec-overflow-head", func(s *mfi.Snapshot) { s.IntState[1] = math.MaxUint64 }},
		{"extra-invalid", func(s *mfi.Snapshot) { s.IntState[2] = 2 }},
		{"extra-on-scheduled", func(s *mfi.Snapshot) { s.IntState[2] = 1 }},
		{"negative-previous", func(s *mfi.Snapshot) { s.FloatState[0] = -1 }},
		{"nonfinite-previous", func(s *mfi.Snapshot) { s.FloatState[0] = math.NaN() }},
		{"nonfinite-sum", func(s *mfi.Snapshot) { s.FloatState[1] = math.Inf(1) }},
		{"overflow-sum", func(s *mfi.Snapshot) { s.FloatState[1] = math.MaxFloat64 }},
		{"bad-correction", func(s *mfi.Snapshot) { s.FloatState[3] = 1e90 }},
		{"negative-sum", func(s *mfi.Snapshot) { s.FloatState[1] = -1e9 }},
		{"negative-ring", func(s *mfi.Snapshot) { s.FloatState[ringHeader] = -1 }},
		{"nonfinite-ring", func(s *mfi.Snapshot) { s.FloatState[ringHeader+1] = math.NaN() }},
		{"two-sides-same-slot", func(s *mfi.Snapshot) {
			s.FloatState[ringHeader] = 1
			s.FloatState[ringHeader+period] = 1
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			broken := stream.Snapshot()
			tc.mutate(&broken)
			mustErrorIs(t, stream.Restore(broken), mfi.ErrStateVersion)
			mustSnapshotBits(t, stream.Snapshot(), original)
		})
	}
}

func TestSnapshot_WarmupCorruptionAndSnapshotDetachment(t *testing.T) {
	stream, err := mfi.NewPeriod(6)
	if err != nil {
		t.Fatal(err)
	}
	initial := stream.Snapshot()
	bad := stream.Snapshot()
	bad.FloatState[0] = 1
	mustErrorIs(t, stream.Restore(bad), mfi.ErrStateVersion)
	mustSnapshotBits(t, stream.Snapshot(), initial)

	if _, _, err = stream.Push(sampleBar(100, 50)); err != nil {
		t.Fatal(err)
	}
	oneBar := stream.Snapshot()
	bad = stream.Snapshot()
	bad.FloatState[1] = 1
	mustErrorIs(t, stream.Restore(bad), mfi.ErrStateVersion)
	mustSnapshotBits(t, stream.Snapshot(), oneBar)

	if _, _, err = stream.Push(sampleBar(101, 5)); err != nil {
		t.Fatal(err)
	}
	partial := stream.Snapshot()
	bad = stream.Snapshot()
	bad.FloatState[5+5] = 1 // Unused positive slot must remain zero.
	mustErrorIs(t, stream.Restore(bad), mfi.ErrStateVersion)
	mustSnapshotBits(t, stream.Snapshot(), partial)

	bad = stream.Snapshot()
	bad.IntState[2] = 1 // Extra rebase cannot occur before any slot eviction.
	mustErrorIs(t, stream.Restore(bad), mfi.ErrStateVersion)
	mustSnapshotBits(t, stream.Snapshot(), partial)

	// Arrays exported by Snapshot are detached from both Stream and other snapshots.
	snapshotA, snapshotB := stream.Snapshot(), stream.Snapshot()
	snapshotA.IntState[0], snapshotA.FloatState[0] = 999, math.NaN()
	mustSnapshotBits(t, stream.Snapshot(), snapshotB)
}

func TestStream_MaxCountCannotAdvance(t *testing.T) {
	const period = 2
	stream, err := mfi.NewPeriod(period)
	if err != nil {
		t.Fatal(err)
	}
	for _, price := range []float64{100, 101, 102} {
		if _, _, err = stream.Push(sampleBar(price, 1)); err != nil {
			t.Fatal(err)
		}
	}

	// A checkpoint with MaxUint64 Count is representable and can describe a valid
	// full ring. Once restored, no further transition can increment that count.
	state := stream.Snapshot()
	state.Count = math.MaxUint64
	state.IntState[0] = period
	state.IntState[1] = (math.MaxUint64 - 1) % period
	if err = stream.Restore(state); err != nil {
		t.Fatalf("valid high-count checkpoint rejected: %v", err)
	}
	before := stream.Snapshot()
	value, valid, err := stream.Push(sampleBar(103, 1))
	mustErrorIs(t, err, mfi.ErrOverflow)
	if value != 0 || valid {
		t.Fatalf("failed Push returned value=%g valid=%t", value, valid)
	}
	mustSnapshotBits(t, stream.Snapshot(), before)
}

func FuzzSnapshot_AtomicValidation(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3, 4})
	f.Add([]byte{5, 6, 7, 8, 9, 10, 11, 12})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 128 {
			return
		}
		stream, err := mfi.NewPeriod(4)
		if err != nil {
			t.Fatal(err)
		}
		for _, price := range []float64{10, 11, 10, 12, 13, 11, 14, 13, 14, 15} {
			if _, _, err = stream.Push(sampleBar(price, 1)); err != nil {
				t.Fatal(err)
			}
		}
		before := stream.Snapshot()
		candidate := stream.Snapshot()
		for i, value := range data {
			switch value % 5 {
			case 0:
				candidate.IntState[i%len(candidate.IntState)] ^= uint64(value)
			case 1:
				index := i % len(candidate.FloatState)
				candidate.FloatState[index] = math.Float64frombits(math.Float64bits(candidate.FloatState[index]) ^ uint64(value))
			case 2:
				candidate.Count ^= uint64(value)
			case 3:
				candidate.FloatState[i%len(candidate.FloatState)] = math.Inf(1)
			default:
				candidate.FloatState[i%len(candidate.FloatState)] = -1
			}
		}
		err = stream.Restore(candidate)
		if err != nil {
			mustErrorIs(t, err, mfi.ErrStateVersion)
			mustSnapshotBits(t, stream.Snapshot(), before)
		}
	})
}
