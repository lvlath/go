// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match_test benchmarks public preparation and solver cost together.
// Inputs and seeds are fixed; internal solver-only benchmarks are migrated too.
package match_test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/lvlath/go/match"
	"github.com/lvlath/go/matrix"
)

// BenchmarkPerfect measures the complete public matching call with allocations.
// It uses deterministic dense costs and separates exact/heuristic regimes.
func BenchmarkPerfect(b *testing.B) {
	for _, k := range []int{10, 50, 100, 200} {
		for _, algo := range []match.Algorithm{match.Blossom, match.Greedy} {
			name := "Blossom"
			if algo == match.Greedy {
				name = "Greedy"
			}
			b.Run(fmt.Sprintf("%s/k%d", name, k), func(b *testing.B) {
				rng := rand.New(rand.NewSource(int64(31000 + k)))
				data := make([]float64, k*k)
				for i := 0; i < k; i++ {
					for j := i + 1; j < k; j++ {
						weight := float64(1+rng.Intn(100000)) / 10
						data[i*k+j], data[j*k+i] = weight, weight
					}
				}
				m, err := matrix.NewDense(k, k)
				if err != nil {
					b.Fatal(err)
				}
				if err = m.Fill(data); err != nil {
					b.Fatal(err)
				}
				vertices := make([]int, k)
				for i := range vertices {
					vertices[i] = i
				}
				policy := match.WithAlgorithm(algo)
				var lastResult *match.Result
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					result, err := match.Perfect(m, vertices, policy)
					if err != nil {
						b.Fatal(err)
					}
					lastResult = result
				}
				b.StopTimer()
				// Validate the published cardinality outside the measured loop.
				// This also keeps the last result observably used without global state.
				if lastResult == nil || len(lastResult.Pairs) != k/2 {
					b.Fatalf("incomplete benchmark result for %s/k%d: %+v", name, k, lastResult)
				}
			})
		}
	}
}
