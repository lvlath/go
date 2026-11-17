// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Snapshot ABI: every split, wrap, rebase, detached ownership, corrupt rejection.
package mfi_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/mfi"
)

func TestMFI09_SnapshotEverySplit(t *testing.T) {
	const n = 90
	prices, volume := make([]float64, n), make([]float64, n)
	for i := range prices {
		prices[i] = float64(30 + (i*7)%13)
		volume[i] = float64(i % 11)
	}
	b := sampleBars(prices, volume)
	for _, period := range []int{1, 2, 7, 14} {
		uninterrupted, e := mfi.New(mfi.Options{Period: period})
		if e != nil {
			t.Fatal(e)
		}
		expected := make([]float64, n)
		expectedValid := make([]bool, n)
		for i := 0; i < n; i++ {
			expected[i], expectedValid[i], e = uninterrupted.Push(sliceBars(b, i))
			if e != nil {
				t.Fatal(e)
			}
		}
		for split := 0; split <= n; split++ {
			first, e := mfi.New(mfi.Options{Period: period})
			if e != nil {
				t.Fatal(e)
			}
			for i := 0; i < split; i++ {
				if _, _, e = first.Push(sliceBars(b, i)); e != nil {
					t.Fatal(e)
				}
			}
			snap := first.Snapshot()
			restored, e := mfi.New(mfi.Options{Period: period})
			if e != nil {
				t.Fatal(e)
			}
			if e = restored.Restore(snap); e != nil {
				t.Fatalf("period=%d split=%d: %v", period, split, e)
			}
			for i := split; i < n; i++ {
				v, ok, err := restored.Push(sliceBars(b, i))
				if err != nil {
					t.Fatal(err)
				}
				if ok != expectedValid[i] || math.Float64bits(v) != math.Float64bits(expected[i]) {
					t.Fatalf("period=%d split=%d index=%d", period, split, i)
				}
			}
		}
	}
}
func TestSnapshot_DetachmentAndCorruption(t *testing.T) {
	s, e := mfi.New(mfi.Options{Period: 2})
	if e != nil {
		t.Fatal(e)
	}
	for _, p := range []float64{1, 2, 1, 3} {
		if _, _, e = s.Push(sampleBar(p, 1)); e != nil {
			t.Fatal(e)
		}
	}
	original := s.Snapshot()
	corrupt := s.Snapshot()
	corrupt.FloatState[5] = math.NaN()
	mustErrorIs(t, s.Restore(corrupt), mfi.ErrStateVersion)
	if math.Float64bits(s.Snapshot().FloatState[5]) != math.Float64bits(original.FloatState[5]) {
		t.Fatal("corrupt restore mutated state")
	}
	corrupt = s.Snapshot()
	corrupt.Config = "wrong"
	mustErrorIs(t, s.Restore(corrupt), mfi.ErrStateVersion)
	corrupt = s.Snapshot()
	corrupt.IntState[1] = 999
	mustErrorIs(t, s.Restore(corrupt), mfi.ErrStateVersion)
	copied := s.Snapshot()
	copied.FloatState[5] = 999
	if math.Float64bits(s.Snapshot().FloatState[5]) == math.Float64bits(999) {
		t.Fatal("snapshot alias")
	}
}

func TestSnapshot_PreservesUnexpectedMutationOnFailure(t *testing.T) {
	s, e := mfi.New(mfi.Options{Period: 3})
	if e != nil {
		t.Fatal(e)
	}
	for _, p := range []float64{1, 2, 1, 3, 2, 4, 3, 5} {
		if _, _, e = s.Push(sampleBar(p, 2)); e != nil {
			t.Fatal(e)
		}
	}
	saved := s.Snapshot()
	malformed := s.Snapshot()
	malformed.FloatState[1] = math.MaxFloat64
	mustErrorIs(t, s.Restore(malformed), mfi.ErrStateVersion)
	after := s.Snapshot()
	if saved.Count != after.Count {
		t.Fatal("count mutated")
	}
	for i := range saved.FloatState {
		if math.Float64bits(saved.FloatState[i]) != math.Float64bits(after.FloatState[i]) {
			t.Fatalf("state mutated at %d", i)
		}
	}
}
