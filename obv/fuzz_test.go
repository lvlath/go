// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package obv_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/obv"
)

// FuzzBatchStreamParity verifies that arbitrary finite positive input transitions
// use the same recurrence and preserve state across checkpoint reconstruction.
func FuzzBatchStreamParity(f *testing.F) {
	f.Add(uint64(1), uint64(2), uint64(3))
	f.Add(uint64(999999), uint64(0), uint64(42))
	f.Fuzz(func(t *testing.T, a, b, c uint64) {
		closes := []float64{float64(a%10000 + 1), float64(b%10000 + 1), float64(c%10000 + 1)}
		volume := []float64{float64(a % 100000), float64(b % 100000), float64(c % 100000)}
		batch, err := obv.Compute(bars(closes, volume))
		if err != nil {
			t.Fatal(err)
		}
		var s obv.Stream
		for i := range closes {
			x, v, e := s.Push(obv.Bar{closes[i], closes[i], closes[i], closes[i], volume[i]})
			if e != nil || !v || math.Float64bits(x) != math.Float64bits(batch.Values[i]) {
				t.Fatal("batch-stream mismatch")
			}
			var restored obv.Stream
			if e := restored.Restore(s.Snapshot()); e != nil {
				t.Fatal(e)
			}
			if !sameSnapshot(s.Snapshot(), restored.Snapshot()) {
				t.Fatal("snapshot mismatch")
			}
			s = restored
		}
	})
}
