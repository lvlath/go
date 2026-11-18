// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo_test benchmarks real driver regimes with reproducible paths.
// Workload preparation is excluded from timed setup; measured allocations include
// the sampler's returned slices and the driver's required immediate detached copy.
package montecarlo_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
)

// BenchmarkRun compares scalar versus 32-feature experiments, serial and
// four-worker parallelism, and mean-only, retained and exact quantile memory modes.
//
// Each fixture generates a deterministic Gaussian residual per coordinate, with
// a fixed phase-like path-ID offset. The measured Run call includes real RNG
// streams, sampler output allocation, driver copy/reduction and mode finalization.
//
// Complexity: mean-only O(Paths*d); quantiles add O(d*Paths*log(Paths)) sorting.
// AI-Hints: read bytes/op and allocs/op together with throughput: exact quantiles
//
//	necessarily require Paths*d materialization even when Samples is nil.
func BenchmarkRun(b *testing.B) {
	for _, dimension := range []int{1, 32} {
		for _, paths := range []uint64{1000, 100000} {
			for _, workers := range []int{1, 4} {
				for _, mode := range []string{"mean", "retained", "quantiles"} {
					name := fmt.Sprintf("D=%d/Paths=%d/W=%d/Mode=%s", dimension, paths, workers, mode)
					b.Run(name, func(b *testing.B) {
						// The fixture and Plan are assembled once outside the timed loop.
						sampler := &funcSampler{dimension, func(_ context.Context, id uint64, random rng.RNG) ([]float64, error) {
							values := make([]float64, dimension)
							for column := range values {
								values[column] = float64(id%19) + random.Normal()
							}
							return values, nil
						}}
						plan := montecarlo.Plan{Paths: paths, Seed: 43, Workers: workers}
						switch mode {
						case "retained":
							plan.RetainSamples = true
						case "quantiles":
							plan.Quantiles = []float64{.01, .5, .99}
							plan.MaxSamplesBytes = 128 << 20
						}

						b.ReportAllocs()
						b.ResetTimer()
						for iteration := 0; iteration < b.N; iteration++ {
							result, err := montecarlo.Run(context.Background(), plan, sampler)
							if err != nil || result == nil || !result.Complete {
								b.Fatalf("unexpected benchmark failure: %v", err)
							}
						}
					})
				}
			}
		}
	}
}

// BenchmarkRunCancellationDrain measures the cost of stopping a fixed four-worker
// run after one deterministic sampler failure, including accepted-work drain/join.
//
// Every execution uses a fresh cancellation context and one explicitly failing
// third logical path. Those fixtures belong inside the measured operation because
// creating the cancellable lifecycle is part of this benchmark's workload.
//
// AI-Hints: this is coordinator shutdown overhead with an honoring sampler,
// not a wall-time upper bound for arbitrary user models ignoring context.
func BenchmarkRunCancellationDrain(b *testing.B) {
	cause := errors.New("benchmark instrument interrupt")
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		ctx, cancel := context.WithCancel(context.Background())
		sampler := &funcSampler{1, func(ctx context.Context, id uint64, _ rng.RNG) ([]float64, error) {
			if id == 2 {
				return nil, cause
			}
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			return []float64{float64(id)}, nil
		}}
		result, err := montecarlo.Run(ctx, montecarlo.Plan{
			Paths: 100000, Workers: 4,
		}, sampler)
		cancel()
		if !errors.Is(err, cause) || result == nil || result.Complete {
			b.Fatalf("expected terminal sampler cause with partial result: %v", err)
		}
	}
}
