// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
package montecarlo_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
)

func TestMC01AndMC02_AnalyticalOracle(t *testing.T) {
	p := montecarlo.Plan{Paths: 4, Workers: 2, RetainSamples: true, Quantiles: []float64{0, .25, .5, 1}}
	r, err := montecarlo.Run(context.Background(), p, oracleSampler())
	if err != nil {
		t.Fatal(err)
	}
	if !r.Complete || r.Completed != 4 || r.Requested != 4 {
		t.Fatalf("unexpected counts: %+v", r)
	}
	for j, w := range []float64{1.5, 3} {
		closeFloat(t, r.Mean[j], w)
	}
	for j, w := range []float64{5.0 / 3, 20.0 / 3} {
		closeFloat(t, r.SampleVariance[j], w)
	}
	for j, w := range []float64{math.Sqrt(5.0 / 12), math.Sqrt(5.0 / 3)} {
		closeFloat(t, r.MeanStandardError[j], w)
	}
	if r.Samples.Rows() != 4 || r.Samples.Cols() != 2 || r.Quantiles.Rows() != 4 || r.Quantiles.Cols() != 2 {
		t.Fatal("matrix shape")
	}
	for i, w := range [][]float64{{0, 0}, {.75, 1.5}, {1.5, 3}, {3, 6}} {
		for j, v := range w {
			got, e := r.Quantiles.At(i, j)
			if e != nil {
				t.Fatal(e)
			}
			closeFloat(t, got, v)
		}
	}
	if e := r.Samples.Set(0, 0, 200); e != nil {
		t.Fatal(e)
	}
	if r.Mean[0] != 1.5 {
		t.Fatal("result metadata aliases samples")
	}
}

func TestMC09_RejectInvalidVectors(t *testing.T) {
	for _, tc := range []struct {
		name   string
		values []float64
		target error
	}{
		{"shape", []float64{1}, montecarlo.ErrShape},
		{"nan", []float64{math.NaN(), 2}, montecarlo.ErrNonFinite},
		{"positive_inf", []float64{math.Inf(1), 2}, montecarlo.ErrNonFinite},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := &funcSampler{2, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
				if id == 2 {
					return tc.values, nil
				}
				return []float64{float64(id), 2 * float64(id)}, nil
			}}
			r, e := montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 4, Workers: 1, RetainSamples: true}, s)
			mustErrorIs(t, e, tc.target)
			if r == nil || r.Completed != 2 || r.Complete || r.Samples.Rows() != 2 || r.Quantiles != nil {
				t.Fatalf("invalid prefix: %+v", r)
			}
		})
	}
}

func TestErrorPreservation(t *testing.T) {
	cause := errors.New("fixture failure")
	s := &funcSampler{1, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == 3 {
			return nil, cause
		}
		return []float64{float64(id)}, nil
	}}
	r, err := montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 5, Workers: 1}, s)
	mustErrorIs(t, err, cause)
	if r == nil || r.Completed != 3 || r.Complete {
		t.Fatalf("bad prefix %+v", r)
	}
}
func TestResult_IsNilInterface(t *testing.T) {
	var p *montecarlo.Result
	mustNilState(t, p, true, "Result")
	p = &montecarlo.Result{}
	mustNilState(t, p, false, "Result")
}
func TestMC08_ExtremeQuantileAndVarianceOverflow(t *testing.T) {
	s := &funcSampler{1, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == 0 {
			return []float64{-math.MaxFloat64}, nil
		}
		return []float64{math.MaxFloat64}, nil
	}}
	r, e := montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 2, Quantiles: []float64{.5}}, s)
	mustErrorIs(t, e, montecarlo.ErrOverflow)
	if r == nil || r.Complete || r.Quantiles != nil {
		t.Fatalf("must not publish quantile on failed variance: %+v", r)
	}
}
func TestConstantMaxFloat(t *testing.T) {
	s := &funcSampler{1, func(_ context.Context, _ uint64, _ rng.RNG) ([]float64, error) {
		return []float64{math.MaxFloat64}, nil
	}}
	r, e := montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 4}, s)
	if e != nil {
		t.Fatal(e)
	}
	if r.Mean[0] != math.MaxFloat64 || r.SampleVariance[0] != 0 || r.MeanStandardError[0] != 0 {
		t.Fatalf("constant stability: %+v", r)
	}
}
func TestRetainFalseWithQuantiles(t *testing.T) {
	r, e := montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 4, Quantiles: []float64{.25, .5}}, oracleSampler())
	if e != nil {
		t.Fatal(e)
	}
	if r.Samples != nil || r.Quantiles == nil {
		t.Fatal("quantiles must not force sample publication")
	}
}
