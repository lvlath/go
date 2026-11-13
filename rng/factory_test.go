// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng_test tests immutable factory replay, ownership and scheduling invariants.
// This file accesses the implementation only through the exported RNG facade.
package rng_test

import (
	"sync"
	"testing"

	"github.com/lvlath/go/rng"
)

// TestFactory_NilReceiverSafe verifies that nil and zero-value factories retain
// the exact canonical version and seed-zero first raw output without panic.
func TestFactory_NilReceiverSafe(t *testing.T) {
	var f *rng.Factory
	if f.Version() != rng.NewFactory().Version() {
		t.Fatal("nil factory version")
	}
	if f.Stream(0, 0).Uint64() != rng.NewFactory().Stream(0, 0).Uint64() {
		t.Fatal("nil factory stream")
	}
	if (new(rng.Factory)).Stream(0, 0).Uint64() != f.Stream(0, 0).Uint64() {
		t.Fatal("zero factory")
	}
}

// TestFactory_StreamIndependenceAndRace constructs 100 logical paths concurrently
// and compares them with sequential replay; state mutation is isolated by stream.
func TestFactory_StreamIndependenceAndRace(t *testing.T) {
	const n = 100
	f := rng.NewFactory()
	got := make([][10]uint64, n)
	var wg sync.WaitGroup
	for k := n - 1; k >= 0; k-- {
		wg.Add(1)
		go func(path int) {
			defer wg.Done()
			s := f.Stream(2026, uint64(path))
			for i := range got[path] {
				got[path][i] = s.Uint64()
			}
		}(k)
	}
	wg.Wait()
	for path := 0; path < n; path++ {
		s := f.Stream(2026, uint64(path))
		for i, want := range got[path] {
			if v := s.Uint64(); v != want {
				t.Fatalf("path %d draw %d", path, i)
			}
		}
	}
	a, b := f.Stream(5, 8), f.Stream(5, 8)
	a.Uint64()
	if got, want := b.Uint64(), f.Stream(5, 8).Uint64(); got != want {
		t.Fatal("streams alias")
	}
}
