// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File snapshot_test.go protects the canonical checkpoint ABI and atomic restore.
package rsi_test

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"testing"

	"github.com/lvlath/go/rsi"
)

// assertSnapshotEqual compares the public checkpoint metadata and numeric arrays explicitly.
// Float comparisons are bitwise, making phase rollback and detached state auditable.
func assertSnapshotEqual(t *testing.T, a, b rsi.Snapshot) {
	t.Helper()
	mustEqualString(t, a.Algorithm, b.Algorithm, "algorithm")
	mustEqualString(t, a.Config, b.Config, "config")
	if a.Count != b.Count {
		t.Fatalf("count %d != %d", a.Count, b.Count)
	}
	mustEqualInt(t, len(a.FloatState), len(b.FloatState), "float length")
	mustEqualInt(t, len(a.IntState), len(b.IntState), "int length")
	for i := range a.FloatState {
		mustSameFloat(t, a.FloatState[i], b.FloatState[i])
	}
	for i := range a.IntState {
		if a.IntState[i] != b.IntState[i] {
			t.Fatalf("int[%d] mismatch", i)
		}
	}
}

// TestRSI07SnapshotAcrossPhases restores across empty, baseline, seed, and ready states.
// The reference replays history from the public facade instead of trusting round-trip alone.
func TestRSI07SnapshotAcrossPhases(t *testing.T) {
	p := 3
	prices := []float64{100, 102, 101, 108, 105, 110, 107, 120, 115, 114, 118, 111}
	original, e := rsi.New(rsi.Options{Period: p})
	if e != nil {
		t.Fatal(e)
	}
	for cut := 0; cut <= p+2; cut++ {
		if cut > 0 {
			if _, _, e := original.Push(prices[cut-1]); e != nil {
				t.Fatal(e)
			}
		}
		snap := original.Snapshot()
		other, e := rsi.New(rsi.Options{Period: p})
		if e != nil {
			t.Fatal(e)
		}
		if e = other.Restore(snap); e != nil {
			t.Fatal(e)
		}
		first, e := rsi.New(rsi.Options{Period: p})
		if e != nil {
			t.Fatal(e)
		}
		for _, historical := range prices[:cut] {
			if _, _, e := first.Push(historical); e != nil {
				t.Fatal(e)
			}
		}
		for _, price := range prices[cut:] {
			a, va, ea := first.Push(price)
			if ea != nil {
				t.Fatal(ea)
			}
			b, vb, eb := other.Push(price)
			if eb != nil {
				t.Fatal(eb)
			}
			mustSameFloat(t, a, b)
			mustEqualBool(t, va, vb, "restore")
		}
	}
}

// TestSnapshotHashABI reconstructs the documented configuration digest from independent SHA-256 bytes.
// It protects little-endian normalized-period encoding and exact state-array lengths.
func TestSnapshotHashABI(t *testing.T) {
	s, e := rsi.New(rsi.Options{Period: 0})
	if e != nil {
		t.Fatal(e)
	}
	snap := s.Snapshot()
	var raw [len("rsi/wilder/v1") + 8]byte
	copy(raw[:], "rsi/wilder/v1")
	binary.LittleEndian.PutUint64(raw[len("rsi/wilder/v1"):], 14)
	sum := sha256.Sum256(raw[:])
	mustEqualString(t, snap.Algorithm, "rsi/wilder/v1", "algorithm")
	mustEqualString(t, snap.Config, hex.EncodeToString(sum[:]), "config")
	mustEqualInt(t, len(snap.FloatState), 3, "float abi")
	mustEqualInt(t, len(snap.IntState), 0, "int abi")
}

// TestRSI08DetachedSnapshotAndRestore proves snapshot mutation does not alias either stream.
// A restoration must copy numeric values instead of retaining caller backing slices.
func TestRSI08DetachedSnapshotAndRestore(t *testing.T) {
	s, e := rsi.New(rsi.Options{Period: 2})
	if e != nil {
		t.Fatal(e)
	}
	for _, p := range []float64{7, 8, 6, 5} {
		if _, _, e = s.Push(p); e != nil {
			t.Fatal(e)
		}
	}
	before := s.Snapshot()
	copySnap := s.Snapshot()
	copySnap.FloatState[0] = 100000
	assertSnapshotEqual(t, before, s.Snapshot())
	restored, e := rsi.New(rsi.Options{Period: 2})
	if e != nil {
		t.Fatal(e)
	}
	if e := restored.Restore(before); e != nil {
		t.Fatal(e)
	}
	before.FloatState[0] = 1
	assertSnapshotEqual(t, s.Snapshot(), restored.Snapshot())
}

// TestRestoreRejectsCorruptionAtomically rejects foreign or malformed state without partial mutation.
func TestRestoreRejectsCorruptionAtomically(t *testing.T) {
	s, e := rsi.New(rsi.Options{Period: 2})
	if e != nil {
		t.Fatal(e)
	}
	for _, p := range []float64{3, 4, 2, 5} {
		if _, _, e := s.Push(p); e != nil {
			t.Fatal(e)
		}
	}
	before := s.Snapshot()
	cases := []func(*rsi.Snapshot){
		func(x *rsi.Snapshot) { x.Algorithm = "foreign" },
		func(x *rsi.Snapshot) { x.Config = "bad" },
		func(x *rsi.Snapshot) { x.FloatState = x.FloatState[:2] },
		func(x *rsi.Snapshot) { x.IntState = []uint64{1} },
		func(x *rsi.Snapshot) { x.FloatState[1] = -1 },
		func(x *rsi.Snapshot) { x.FloatState[1] = math.NaN() },
		func(x *rsi.Snapshot) { x.FloatState[0] = 0 },
		func(x *rsi.Snapshot) { x.Count = 0 },
	}
	for i, change := range cases {
		candidate := s.Snapshot()
		change(&candidate)
		err := s.Restore(candidate)
		if err == nil {
			t.Fatalf("case %d unexpectedly accepted", i)
		}
		mustErrorIs(t, err, rsi.ErrStateVersion)
		assertSnapshotEqual(t, before, s.Snapshot())
	}
}

// TestRSI10CounterOverflow forces a valid maximum Count through the public Restore ABI.
// The next valid price must fail before any previous, mean, or counter mutation.
func TestRSI10CounterOverflow(t *testing.T) {
	s, e := rsi.New(rsi.Options{Period: 2})
	if e != nil {
		t.Fatal(e)
	}
	snap := s.Snapshot()
	snap.Count = math.MaxUint64
	snap.FloatState = []float64{5, 1, 2}
	if e := s.Restore(snap); e != nil {
		t.Fatal(e)
	}
	before := s.Snapshot()
	_, _, e = s.Push(6)
	mustErrorIs(t, e, rsi.ErrOverflow)
	assertSnapshotEqual(t, before, s.Snapshot())
}
