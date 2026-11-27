// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package drawdown_test benchmarks full validated batches, including allocation.
// Fixture setup is excluded; arithmetic regimes are not interchangeable.
package drawdown_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/lvlath/go/drawdown"
)

// BenchmarkCompute measures complete validated batches in shallow and deep arithmetic regimes.
func BenchmarkCompute(b *testing.B) {
	for _, n := range []int{1000, 100000, 1000000} {
		for _, regime := range []struct {
			name string
			deep bool
		}{{"Shallow", false}, {"Deep", true}} {
			b.Run(fmt.Sprintf("%s/n=%d", regime.name, n), func(b *testing.B) {
				equity := sawEquity(n, regime.deep)
				ctx := context.Background()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := drawdown.Compute(ctx, equity); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
