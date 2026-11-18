// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
// Fuzz validates stable input classification, prefix publication and positivity.

// Package jumpdiffusion_test fuzzes validation, ownership, and prefix invariants.
// The fuzz target uses only exported APIs with fixed per-input stream keys.
package jumpdiffusion_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/jumpdiffusion"
	"github.com/lvlath/go/rng"
)

func FuzzSimulate_Contract(f *testing.F) {
	f.Add(float64(0.5), float64(-0.1), float64(0.1), uint64(42))
	f.Add(float64(30), float64(0), float64(0), uint64(0))
	f.Add(math.NaN(), float64(0), float64(0), uint64(1))
	f.Fuzz(func(t *testing.T, intensity, mean, std float64, seed uint64) {
		p := jdParams()
		p.Intensity = intensity
		p.LogJumpMean = mean
		p.LogJumpStdDev = std
		result, err := jumpdiffusion.Simulate(context.Background(), p, jdGrid(), rng.NewStream(seed, 0))
		if err != nil && result == nil {
			return
		}
		if result == nil {
			t.Fatal("nil result without error")
		}
		if len(result.Times) != len(result.Values) || len(result.Times) == 0 {
			t.Fatal("invalid prefix shape")
		}
		if err == nil && !result.Complete {
			t.Fatal("incomplete successful trajectory")
		}
		if err != nil && result.Complete {
			t.Fatal("complete failed trajectory")
		}
		for _, v := range result.Values {
			if !(v > 0) || math.IsInf(v, 0) || math.IsNaN(v) {
				t.Fatal("nonpositive or nonfinite result")
			}
		}
	})
}
