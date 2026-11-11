// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Benchmarks measure repeated immutable-model inference and scalar symbolic
// comparisons on reproducible aligned windows. All construction, validation,
// and positive-distance admission checks occur before the measurement loop.
// AI-Hint: Never benchmark a zero-gap result when claiming separated-bin cost.
package sax_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/sax"
)

// BenchmarkEncode measures population-Z normalization, fractional PAA and
// logarithmic bin searches, excluding model construction and source generation.
// The n/w values deliberately include both divisible and fractional partitions;
// increasing alphabet size isolates the effect of quantization searches.
// Complexity: O(n+w*log(a)) per operation; O(n+w) per-call workspace.
func BenchmarkEncode(b *testing.B) {
	for _, n := range []int{1000, 100000} {
		for _, w := range []int{16, 64} {
			for _, a := range []int{4, 8, 32} {
				b.Run(fmt.Sprintf("n=%d/w=%d/a=%d", n, w, a), func(b *testing.B) {
					data := make([]float64, n)
					for i := range data {
						// A ramp with deterministic oscillation spans multiple
						// normalized bins without relying on random sampling.
						data[i] = -2 + 4*float64(i)/float64(n) + 0.15*math.Sin(float64(i)*0.017)
					}
					opts := sax.DefaultOptions(w)
					opts.Alphabet = a
					enc, err := sax.NewEncoder(opts)
					if err != nil {
						b.Fatal(err)
					}
					// Admit a healthy baseline before timing to avoid
					// measuring a nonrepresentable or erroneous input regime.
					if _, err = enc.Encode(data); err != nil {
						b.Fatal(err)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if _, err = enc.Encode(data); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

// BenchmarkMindist measures symbol-gap lookup and scaled Euclidean norm over
// two preencoded opposing-trend windows. The opposite trends produce genuinely
// separated bins, so the benchmark cannot accidentally measure only gap-zero
// early continuations. Result validation is intentionally included in the call.
// Complexity: O(w) per operation without optional PAA; O(1) additional space.
func BenchmarkMindist(b *testing.B) {
	for _, n := range []int{1000, 100000} {
		for _, w := range []int{16, 64} {
			for _, a := range []int{4, 8, 32} {
				b.Run(fmt.Sprintf("n=%d/w=%d/a=%d", n, w, a), func(b *testing.B) {
					opts := sax.DefaultOptions(w)
					opts.Alphabet = a
					opts.Normalization = sax.Raw
					enc, err := sax.NewEncoder(opts)
					if err != nil {
						b.Fatal(err)
					}

					// Keep both windows finite and equal-length, with a
					// sustained opposite trend in the same physical units.
					x, y := make([]float64, n), make([]float64, n)
					for i := range x {
						trend := 4 * float64(i) / float64(n)
						x[i] = -2 + trend + 0.05*math.Sin(float64(i)*0.021)
						y[i] = 2 - trend + 0.05*math.Cos(float64(i)*0.021)
					}
					lhs, err := enc.Encode(x)
					if err != nil {
						b.Fatal(err)
					}
					rhs, err := enc.Encode(y)
					if err != nil {
						b.Fatal(err)
					}
					bound, err := enc.Mindist(lhs, rhs)
					if err != nil {
						b.Fatal(err)
					}
					if bound <= 0 {
						b.Fatalf("benchmark must exercise separated bins: bound=%g", bound)
					}

					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if _, err = enc.Mindist(lhs, rhs); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}
