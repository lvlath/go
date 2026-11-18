// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
package montecarlo_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
)

func TestMC03_BitwiseWorkersParity(t *testing.T) {
	s := &funcSampler{3, func(_ context.Context, id uint64, r rng.RNG) ([]float64, error) {
		return []float64{r.Normal() + float64(id%7), r.UniformOpen(), r.Normal()}, nil
	}}
	plan := montecarlo.Plan{Paths: 257, Seed: 4839, Workers: 1, RetainSamples: true, Quantiles: []float64{.01, .5, .95}}
	baseline, e := montecarlo.Run(context.Background(), plan, s)
	if e != nil {
		t.Fatal(e)
	}
	for _, workers := range []int{2, 4, 8} {
		plan.Workers = workers
		got, err := montecarlo.Run(context.Background(), plan, s)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 3; i++ {
			for _, pair := range [][2]float64{{got.Mean[i], baseline.Mean[i]}, {got.SampleVariance[i], baseline.SampleVariance[i]}, {got.MeanStandardError[i], baseline.MeanStandardError[i]}} {
				if math.Float64bits(pair[0]) != math.Float64bits(pair[1]) {
					t.Fatalf("worker parity failed %d feature %d", workers, i)
				}
			}
		}
		for _, mat := range [][2]interface {
			Rows() int
			Cols() int
			At(int, int) (float64, error)
		}{{got.Samples, baseline.Samples}, {got.Quantiles, baseline.Quantiles}} {
			for i := 0; i < mat[0].Rows(); i++ {
				for j := 0; j < mat[0].Cols(); j++ {
					a, _ := mat[0].At(i, j)
					b, _ := mat[1].At(i, j)
					if math.Float64bits(a) != math.Float64bits(b) {
						t.Fatalf("matrix bitwise mismatch workers %d (%d,%d)", workers, i, j)
					}
				}
			}
		}
	}
}
