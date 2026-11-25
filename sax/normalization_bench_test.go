// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// The benchmark measures the same fixed small-spread window with and without
// a large, exactly representable common offset. Setup is outside the timer.
package sax_test

import (
	"fmt"
	"testing"

	"github.com/lvlath/go/sax"
)

func BenchmarkEncodeOffset(b *testing.B) {
	for _, n := range []int{1000, 100000} {
		for _, offset := range []float64{0, 1e16, -1e16} {
			b.Run(fmt.Sprintf("n=%d/offset=%g", n, offset), func(b *testing.B) {
				const segments = 64
				values := make([]float64, n)
				for i := range values {
					// All increments are even integers, exactly representable at 1e16.
					values[i] = offset + 2*float64(i%31)
				}
				encoder, err := sax.NewEncoder(sax.DefaultOptions(segments))
				if err != nil {
					b.Fatal(err)
				}
				if _, err := encoder.Encode(values); err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := encoder.Encode(values); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
