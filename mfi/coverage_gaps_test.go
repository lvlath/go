// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Public Restore/Push probes of the compensated ring's exceptional branches.
// Fixtures emulate finite-but-adversarial saved accumulators, never private hooks.
package mfi_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/mfi"
)

// TestCoverage_ImplRing_AllGaps distinguishes normal arithmetic overflow,
// effective-sum overflow and negative-drift recovery with its bounded epoch flag.
func TestCoverage_ImplRing_AllGaps(t *testing.T) {
	const p = 4
	maxValue := math.MaxFloat64
	halfULP := math.Ldexp(1, 970)
	ulp := 2 * halfULP
	zeros := make([]float64, p)
	t.Run("correction overflow on insertion", func(t *testing.T) {
		// The saved effective sum is exactly one ULP, but the two finite halves
		// almost cancel. Adding half an ULP to the even negative sum loses that
		// residual, causing the positive correction itself to overflow.
		s := restoredRingFixture(t, p, p+1, 1, []float64{0, ulp, 0, 0}, zeros, -maxValue+ulp, 0, maxValue, 0, 0)
		assertOverflowAtomic(t, s, sampleBar(2, halfULP/2))
	})
	t.Run("positive eviction overflow", func(t *testing.T) {
		// Restore's relative envelope admits a few ULPs of mismatch at this scale.
		// Subtracting the old contribution from an adversarial negative saved half
		// would overflow even though the effective saved window was finite.
		old := math.Nextafter(math.Nextafter(maxValue/2, math.Inf(1)), math.Inf(1))
		s := restoredRingFixture(t, p, p+1, 1, []float64{old, 0, 0, 0}, zeros, -maxValue/2, 0, maxValue, 0, 0)
		assertOverflowAtomic(t, s, sampleBar(1, 0))
	})
	t.Run("negative eviction overflow", func(t *testing.T) {
		old := math.Nextafter(math.Nextafter(maxValue/2, math.Inf(1)), math.Inf(1))
		s := restoredRingFixture(t, p, p+1, 1, zeros, []float64{old, 0, 0, 0}, 0, -maxValue/2, 0, maxValue, 0)
		assertOverflowAtomic(t, s, sampleBar(1, 0))
	})
	for _, positive := range []bool{true, false} {
		name := "negative effective overflow"
		if positive {
			name = "positive effective overflow"
		}
		t.Run(name, func(t *testing.T) {
			pos, neg := zeros, []float64{0, maxValue, 0, 0}
			ps, ns, pc, nc := 0.0, maxValue/2, 0.0, maxValue/2
			bar := sampleBar(1, maxValue/4)
			previous := 2.0
			if positive {
				pos, neg = neg, pos
				ps, ns = ns, ps
				pc, nc = nc, pc
				bar = sampleBar(2, maxValue/8)
				previous = 1
			}
			s := restoredRingFixture(t, p, p+1, previous, pos, neg, ps, ns, pc, nc, 0)
			assertOverflowAtomic(t, s, bar)
		})
	}
	t.Run("scheduled negative rebase arithmetic overflow", func(t *testing.T) {
		s, err := mfi.NewPeriod(2)
		if err != nil {
			t.Fatal(err)
		}
		for _, bar := range []mfi.Bar{sampleBar(4, 0), sampleBar(2, maxValue*0.4)} {
			if _, _, err = s.Push(bar); err != nil {
				t.Fatal(err)
			}
		}
		assertOverflowAtomic(t, s, sampleBar(1, maxValue*0.4))
	})
	t.Run("emergency rebase and second drift", func(t *testing.T) {
		// A small accepted deficit simulates residual subtraction drift. Evicting
		// the only nonzero contribution makes the local effective sum negative.
		s := restoredRingFixture(t, p, p+1, 1, []float64{10, 0, 0, 0}, zeros, math.Nextafter(10, 0), 0, 0, 0, 0)
		value, ready, err := s.Push(sampleBar(1, 0))
		if err != nil || !ready || value != 50 {
			t.Fatalf("rebase: (%g,%t,%v)", value, ready, err)
		}
		after := s.Snapshot()
		if after.IntState[mfiExtra] != 1 || after.IntState[mfiHead] != 1 || after.FloatState[mfiPositiveSum] != 0 || after.FloatState[mfiPositiveCorrection] != 0 {
			t.Fatalf("rebase state: %+v", after)
		}
		// Model another admitted checkpoint later in that same epoch. The flag
		// must survive Restore; no second O(P) emergency scan may be attempted.
		second := restoredRingFixture(t, p, p+2, 1, []float64{0, 10, 0, 0}, zeros, math.Nextafter(10, 0), 0, 0, 0, 1)
		assertOverflowAtomic(t, second, sampleBar(1, 0))
		// Recovery after the scheduled boundary clears the flag without an extra scan.
		for i := 0; i < p-1; i++ {
			if _, _, err = s.Push(sampleBar(1, 0)); err != nil {
				t.Fatal(err)
			}
		}
		if s.Snapshot().IntState[mfiExtra] != 0 {
			t.Fatal("scheduled rebase did not clear flag")
		}
	})
	t.Run("both signs and all-zero window", func(t *testing.T) {
		for _, tc := range []struct {
			prices []float64
			volume []float64
			want   float64
		}{
			{[]float64{1, 2, 3}, []float64{0, 1, 1}, 100},
			{[]float64{3, 2, 1}, []float64{0, 1, 1}, 0},
			{[]float64{1, 2, 3}, []float64{0, 0, 0}, 50},
		} {
			result, err := mfi.ComputePeriod(sampleBars(tc.prices, tc.volume), 2)
			if err != nil || result == nil || !result.Valid[2] || result.Values[2] != tc.want {
				t.Fatalf("ratio: %+v %v want %g", result, err, tc.want)
			}
		}
	})
}

// TestCoverage_ScheduledEffectiveOverflow tests a naturally produced window
// where each raw addition and correction is finite, but their effective sum
// rounds beyond MaxFloat64 at the scheduled replay. Both signs are independent.
func TestCoverage_ScheduledEffectiveOverflow(t *testing.T) {
	maxValue := math.MaxFloat64
	small := math.Ldexp(1, 969)
	for _, tc := range []struct{ prices, volumes []float64 }{
		{[]float64{1, 2, 4, 8}, []float64{0, maxValue / 2, small / 4, small / 8}},
		{[]float64{8, 4, 2, 1}, []float64{0, maxValue / 4, small / 2, small}},
	} {
		s, err := mfi.NewPeriod(3)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 3; i++ {
			if _, _, err = s.Push(sampleBar(tc.prices[i], tc.volumes[i])); err != nil {
				t.Fatal(err)
			}
		}
		assertOverflowAtomic(t, s, sampleBar(tc.prices[3], tc.volumes[3]))
	}
}
