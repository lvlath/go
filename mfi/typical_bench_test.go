// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Nonflat price benchmarks distinguish ordinary and full-exponent candles.
package mfi_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/mfi"
)

// BenchmarkCanonicalTypicalPush measures the full public transition, including
// classification and periodic rebases, for compact and wide integer arithmetic.
func BenchmarkCanonicalTypicalPush(b *testing.B) {
	for _, tc := range []struct {
		name string
		bars [2]mfi.Bar
	}{
		{"compact", [2]mfi.Bar{sortedPriceBar(99, 100, 101), sortedPriceBar(100, 101, 102)}},
		{"wide", [2]mfi.Bar{sortedPriceBar(math.SmallestNonzeroFloat64, 1, math.MaxFloat64/2), sortedPriceBar(math.SmallestNonzeroFloat64, 2, math.MaxFloat64)}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			tc.bars[0].Volume, tc.bars[1].Volume = 0x1p-10, 0x1p-10
			s, err := mfi.NewPeriod(14)
			if err != nil {
				b.Fatal(err)
			}
			for i := 0; i <= 14; i++ {
				if _, _, err = s.Push(tc.bars[i%2]); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, _, err = s.Push(tc.bars[i%2]); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
