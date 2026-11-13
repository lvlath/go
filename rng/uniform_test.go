// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng_test tests the public uniform conversion on forced extreme raw words.
// A black-box numerical oracle must exercise both endpoints, not merely random draws.
package rng_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/rng"
)

// TestUniformOpen_StrictOpenInterval verifies exact endpoints and one-word
// consumption using the exported pure conversion and a parallel raw stream.
func TestUniformOpen_StrictOpenInterval(t *testing.T) {
	const minOpen = 0x1p-53
	maxOpen := math.Nextafter(1, 0)
	cases := []struct {
		raw  uint64
		want float64
	}{
		{0, minOpen},
		{1, minOpen},
		{(uint64(1) << 12) - 1, minOpen},
		{(uint64(1) << 12), 3 * minOpen},
		{^uint64(0), maxOpen},
		{^uint64(0) - 4095, maxOpen},
	}
	for _, c := range cases {
		got := rng.UniformOpenFromUint64(c.raw)
		mustSameFloatBits(t, got, c.want, "forced endpoint")
		if !(got > 0 && got < 1) {
			t.Fatalf("raw %x violates open interval: %.17g", c.raw, got)
		}
	}

	// A 53-bit midpoint at its upper edge rounds to exactly one; this test
	// anchors why the 52-bit extraction cannot be widened as an optimization.
	unsafeTop := (float64((uint64(1)<<53)-1) + 0.5) * 0x1p-53
	mustSameFloatBits(t, unsafeTop, 1, "53-bit rounding counterexample")

	stream := rng.NewStream(77, 23)
	raw := rng.NewStream(77, 23)
	for i := 0; i < 100000; i++ {
		x := raw.Uint64()
		got := stream.UniformOpen()
		want := rng.UniformOpenFromUint64(x)
		mustSameFloatBits(t, got, want, "single-draw parity")
		if !(got > 0 && got < 1) {
			t.Fatalf("draw %d outside open interval: %.17g", i, got)
		}
	}
}
