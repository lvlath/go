// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap_test measures only package operations, excluding fixture generation.
// This file treats segment lengths as explicit reset cadence, not a rolling-window VWAP parameter.
package vwap_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/lvlath/go/vwap"
)

// BenchmarkCompute measures exact segmented batch throughput across explicit input sizes and reset cadences.
// Each run includes validation and owning result allocation, but excludes fixture generation.
//
// Implementation:
//   - Stage 1: generate n-row histories for each declared reset cadence before timing.
//   - Stage 2: time only the public Compute calls and minimal error/shape checks.
//   - Stage 3: report bytes, time, and allocations for every size/cadence combination.
//
// Behavior highlights:
//   - Fixture allocation and constructor checks occur before ResetTimer.
//
// Inputs:
//   - b: benchmark harness; n is 1e3, 1e5, or 1e6; segment is 14, 64, or 512.
//
// Returns:
//   - Per-operation batch timings with input payload bytes and allocation counts.
//
// Errors:
//   - Any unexpected public operation error terminates the benchmark rather than timing rejection.
//
// Determinism:
//   - The same public Compute behavior for every measured iteration.
//
// Complexity:
//   - O(n*L) time and O(n+L) storage per operation, L=67.
//
// Notes:
//   - The 17-byte throughput estimate counts two float64 inputs and one reset bool, not allocator overhead.
//
// AI-Hints:
//   - Do not describe the cadence as a sliding-window period or include setup in measured cost.
func BenchmarkCompute(b *testing.B) {
	for _, n := range []int{1000, 100000, 1000000} {
		for _, segment := range []int{14, 64, 512} {
			b.Run(fmt.Sprintf("n=%d/segment=%d", n, segment), func(b *testing.B) {
				p, v, r := benchmarkHistory(n, segment)
				ctx := context.Background()
				b.ReportAllocs()
				b.SetBytes(int64(n) * 17)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					result, err := vwap.Compute(ctx, p, v, r, vwap.Options{})
					if err != nil || len(result.Values) != n {
						b.Fatalf("Compute: %v", err)
					}
				}
			})
		}
	}
}

// BenchmarkPush measures ordinary-price transactional stream updates with periodic explicit resets.
// The workload includes occasional resets and changing prices without storing history.
//
// Implementation:
//   - Stage 1: construct the stream and context before starting the timer.
//   - Stage 2: time the unchanged per-row price pattern and public Push transition.
//   - Stage 3: report steady workload time and internal allocations.
//
// Behavior highlights:
//   - Fixture allocation and constructor checks occur before ResetTimer.
//
// Inputs:
//   - b: benchmark harness; the explicit reset cadence is 64 rows.
//
// Returns:
//   - Per-row streaming timings including both reset and continuation work.
//
// Errors:
//   - Any unexpected public operation error terminates the benchmark rather than timing rejection.
//
// Determinism:
//   - The price pattern and reset sequence depend only on iteration index.
//
// Complexity:
//   - O(L) time and workspace per Push, L=67; no history-sized allocation.
//
// Notes:
//   - The small index-to-price expression is retained consistently when comparing revisions.
//
// AI-Hints:
//   - Compare identical workloads and toolchains; do not attribute noisy single-run changes to arithmetic.
func BenchmarkPush(b *testing.B) {
	stream := newStream(b)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := stream.Push(ctx, 100+float64(i%7), 3, i%64 == 0); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkRestore measures validation and reconstruction of an extreme signed exact checkpoint.
// Canonical decimal parsing and cache rounding are included; Snapshot formatting is setup only.
//
// Implementation:
//   - Stage 1: construct an extreme finite signed stream and capture its snapshot before timing.
//   - Stage 2: repeatedly Restore that same valid public record.
//   - Stage 3: report restore cost and allocation counts.
//
// Behavior highlights:
//   - Fixture allocation and constructor checks occur before ResetTimer.
//
// Inputs:
//   - b: benchmark harness; state comes from price=-1e308 and volume=1e308.
//
// Returns:
//   - Per-restore timings for a long exact numerator and denominator.
//
// Errors:
//   - Any unexpected public operation error terminates the benchmark rather than timing rejection.
//
// Determinism:
//   - The exact input snapshot is fixed across all iterations.
//
// Complexity:
//   - O(D*L) time and O(L) workspace per Restore, D<=1300 per sum and L=67.
//
// Notes:
//   - This does not measure JSON I/O, storage, authentication, or snapshot encoding.
//
// AI-Hints:
//   - Do not put Snapshot inside the measured loop when labeling the result Restore.
func BenchmarkRestore(b *testing.B) {
	stream := newStream(b)
	mustPush(b, stream, -1e308, 1e308, true)
	state := stream.Snapshot()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := stream.Restore(state); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkPushExtremeCancellation measures preservation of a small term between opposite extremes.
// Every three accepted rows form an independent anchored segment with an exact final mean of RN64(1/3).
// Implementation:
//   - Stage 1: construct the fixed cancellation fixture, stream, and context before timing.
//   - Stage 2: cycle through all three rows using the public Push method.
//   - Stage 3: report per-row cost and allocations without decimal snapshots in the hot loop.
//
// Behavior highlights:
//   - The middle price contributes low-order bits that naive floating addition would lose.
//
// Inputs:
//   - b: benchmark harness; fixture prices are [1e308,1,-1e308] with unit weights.
//
// Returns:
//   - Timings for extreme addition, exact cancellation, and explicit resets in a fixed mix.
//
// Errors:
//   - Any unexpected Push error fails the benchmark.
//
// Determinism:
//   - The fixture and modulo-three replay order are fixed.
//
// Complexity:
//   - O(L) time and workspace per accepted row, L=67.
//
// Notes:
//   - Correctness is checked independently by TestVWAP_CancellationPrecision and rational-oracle tests.
//
// AI-Hints:
//   - Do not add Snapshot, string formatting, or oracle work to the measured loop.
func BenchmarkPushExtremeCancellation(b *testing.B) {
	fixture := cancellationFixture()
	stream := newStream(b)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index := i % len(fixture.price)
		if _, err := stream.Push(ctx, fixture.price[index], fixture.volume[index], fixture.reset[index]); err != nil {
			b.Fatal(err)
		}
	}
}
