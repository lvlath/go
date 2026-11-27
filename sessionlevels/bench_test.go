// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels_test exercises only the public sessionlevels facade.
// This file measures the complete batch API with fixture generation excluded.
package sessionlevels_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/lvlath/go/sessionlevels"
)

var benchmarkResult *sessionlevels.Result

// BenchmarkCompute varies input length and session count, not an invented period option.
// The 14/64/512 values are synthetic bars per session, not calendar-day assumptions.
func BenchmarkCompute(b *testing.B) {
	for _, n := range []int{1000, 100000, 1000000} {
		for _, width := range []int{1, 14, 64, 512, n} {
			b.Run(fmt.Sprintf("n=%d/sessionBars=%d", n, width), func(b *testing.B) {
				f := sizedFixture(n, width)
				ctx := context.Background()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					result, err := sessionlevels.Compute(ctx, f.high, f.low, f.closePrices, f.sessions, sessionlevels.Options{})
					if err != nil {
						b.Fatal(err)
					}
					benchmarkResult = result
				}
				b.StopTimer()
				b.ReportMetric(float64(len(f.sessions)), "sessions")
			})
		}
	}
}
