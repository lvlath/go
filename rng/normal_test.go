// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng_test tests Box-Muller spare preservation and Gaussian sanity moments.
// This file accesses the implementation only through the exported RNG facade.
package rng_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/rng"
)

// TestNormal_BoxMullerCacheProtocol checks the analytic cosine/sine pair and
// confirms that interleaved raw/uniform calls never consume cached Gaussian state.
func TestNormal_BoxMullerCacheProtocol(t *testing.T) {
	f := rng.NewFactory()
	s, raw := f.Stream(19, 65), f.Stream(19, 65)
	u1, u2 := raw.UniformOpen(), raw.UniformOpen()
	radius := math.Sqrt(-2 * math.Log(u1))
	angle := 2 * math.Pi * u2
	first, spare := radius*math.Cos(angle), radius*math.Sin(angle)
	if got := s.Normal(); math.Float64bits(got) != math.Float64bits(first) {
		t.Fatalf("cos got %g want %g", got, first)
	}
	if x, y := s.Uint64(), raw.Uint64(); x != y {
		t.Fatalf("raw interleave: %x vs %x", x, y)
	}
	if got := s.Normal(); math.Float64bits(got) != math.Float64bits(spare) {
		t.Fatalf("spare changed: %.17g want %.17g", got, spare)
	}
	if x, y := s.Uint64(), raw.Uint64(); x != y {
		t.Fatalf("cached call consumed raw draw")
	}
	if got := s.Normal(); math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("nonfinite normal")
	}
	// Third Normal consumed two raw words. Advance the oracle identically.
	raw.Uint64()
	raw.Uint64()
	// UniformOpen may also intervene without invalidating the pending spare.
	ux, uy := s.UniformOpen(), raw.UniformOpen()
	if ux != uy {
		t.Fatalf("uniform interleave failed")
	}
	if got := s.Normal(); math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("nonfinite spare")
	}
	if x, y := s.Uint64(), raw.Uint64(); x != y {
		t.Fatalf("uniform/cache protocol consumed raw")
	}
}

// TestRNG_StatisticalSmokeMoments checks one million fixed-seed normal outputs
// against intentionally wide moment tolerances, not a formal randomness proof.
func TestRNG_StatisticalSmokeMoments(t *testing.T) {
	const n = 1000000
	s := rng.NewFactory().Stream(0, 9)
	var m2, m3, mean float64
	for i := 1; i <= n; i++ {
		x := s.Normal()
		delta := x - mean
		dn := delta / float64(i)
		term := delta * dn * float64(i-1)
		m3 += -3*dn*m2 + term*dn*float64(i-2)
		m2 += term
		mean += dn
	}
	variance := m2 / float64(n)
	skew := m3 / float64(n) / math.Pow(variance, 1.5)
	if math.Abs(mean) > 0.03 || math.Abs(variance-1) > 0.05 || math.Abs(skew) > 0.06 {
		t.Fatalf("smoke mean=%g variance=%g skew=%g", mean, variance, skew)
	}
}
