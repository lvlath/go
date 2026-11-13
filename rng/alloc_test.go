// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng_test asserts the no-allocation draw loop contract on live streams.
// Stream construction is intentionally excluded, because it returns a new object.
package rng_test

import (
	"testing"

	"github.com/lvlath/go/rng"
)

// TestHotPath_ZeroAllocations checks repeated draws after a stream is created.
// This supplements benchmark allocations with a regression assertion.
func TestHotPath_ZeroAllocations(t *testing.T) {
	stream := rng.NewStream(123, 456)
	for _, test := range []struct {
		name string
		draw func()
	}{
		{"uint64", func() { sinkWord = stream.Uint64() }},
		{"uniform", func() { sinkFloat = stream.UniformOpen() }},
		{"normal", func() { sinkFloat = stream.Normal() }},
	} {
		t.Run(test.name, func(t *testing.T) {
			allocs := testing.AllocsPerRun(50, func() {
				for i := 0; i < 1000; i++ {
					test.draw()
				}
			})
			if allocs != 0 {
				t.Fatalf("draw allocated %.3f times per 1000 operations", allocs)
			}
		})
	}
}
