// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test benchmarks certified pruning against the identical exact scorer.
// Fixtures are prepared outside the timed section and DP work is reported explicitly.
package dtw_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/dtw"
)

// BenchmarkSearch_CertifiedVsExhaustive compares machine-certified pruning
// with the same band-only recurrence under a budget that omits optional envelopes.
// Both branches publish exactly the same complete global top-K ranking.
//
// Implementation:
//   - Stage 1: prepare a 64-step query and 1000-step candidate series.
//   - Stage 2: execute Search with or without an optional envelope under identical options.
//   - Stage 3: report cells/op and pruned/op alongside Go benchmark allocations.
//
// Behavior highlights:
//   - Easy separation tests pruning; all-equal data is an adversarial no-pruning regime.
//
// Inputs:
//   - b: Go benchmark state; all fixtures are constructed before ResetTimer.
//
// Returns:
//   - Standard benchmark metrics, with no global package state mutation.
//
// Errors:
//   - Any incomplete result, error, or rank mismatch aborts that benchmark iteration.
//
// Determinism:
//   - Workers=1 fixes the exact publication order and cutoff updates.
//
// Complexity:
//   - Worst-case O(C*L*b) for both variants; no asymptotic universal speedup implied.
//
// Notes:
//   - Compare numeric results only on identical Go toolchains and machines.
//
// AI-Hints:
//   - Never benchmark preparation of fixtures inside b.N iterations.
//   - Treat CellsVisited as algorithmic work, not as a replacement for ns/op.
func BenchmarkSearch_CertifiedVsExhaustive(b *testing.B) {
	const length = 64
	const dataLength = 1000
	query := make([]float64, length)
	for i := range query {
		query[i] = float64(i % 7)
	}
	for _, regime := range []struct {
		name     string
		allEqual bool
	}{
		{name: "easy-separated"},
		{name: "no-pruning", allEqual: true},
	} {
		data := make([]float64, dataLength)
		for i := range data {
			if regime.allEqual {
				data[i] = 0
			} else if i < length {
				data[i] = query[i]
			} else {
				data[i] = float64(100 + i%7)
			}
		}
		for _, variant := range []struct {
			name     string
			maxBytes uint64
		}{
			{name: "certified", maxBytes: 0},
			{name: "exhaustive-budget", maxBytes: 900},
		} {
			b.Run(regime.name+"/"+variant.name, func(b *testing.B) {
				opts := dtw.DefaultSearchOptions()
				opts.Radius, opts.Workers, opts.TopK = 8, 1, 1
				opts.MaxWorkingBytes = variant.maxBytes
				var totalCells, totalPruned uint64
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					result, err := dtw.Search(context.Background(), query, data, opts)
					if err != nil || result == nil || !result.Complete || len(result.Hits) != 1 {
						b.Fatalf("incomplete benchmark run: result=%+v err=%v", result, err)
					}
					// Both regimes have a canonical earliest winner at start zero;
					// this prevents accidental benchmarking of a different ranking.
					if result.Hits[0].Window.Start != 0 ||
						(!regime.allEqual && result.Hits[0].SquaredDistance != 0) {
						b.Fatalf("benchmark ranking changed: %+v", result.Hits)
					}
					totalCells += result.CellsVisited
					totalPruned += result.CandidatesPruned
				}
				b.StopTimer()
				b.ReportMetric(float64(totalCells)/float64(b.N), "cells/op")
				b.ReportMetric(float64(totalPruned)/float64(b.N), "pruned/op")
			})
		}
	}
}
