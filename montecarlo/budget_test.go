// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
package montecarlo_test

import (
	"context"
	"math"
	"sync/atomic"
	"testing"

	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
)

func TestMC06_BudgetRejectsBeforeSampling(t *testing.T) {
	var called atomic.Int64
	s := &funcSampler{2, func(_ context.Context, _ uint64, _ rng.RNG) ([]float64, error) {
		called.Add(1)
		return []float64{1, 2}, nil
	}}
	r, e := montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 100, RetainSamples: true, MaxSamplesBytes: 64}, s)
	mustErrorIs(t, e, montecarlo.ErrBudget)
	if r != nil || called.Load() != 0 {
		t.Fatal("budget violation sampled")
	}
	// raw 100*2*8=1600, sort=800, output 1*2*8=16: total 2416.
	r, e = montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 100, Quantiles: []float64{.5}, MaxSamplesBytes: 2415}, s)
	mustErrorIs(t, e, montecarlo.ErrBudget)
	if r != nil || called.Load() != 0 {
		t.Fatal("quantile budget violation sampled")
	}
}
func TestPlanValidation(t *testing.T) {
	for _, p := range []montecarlo.Plan{{Paths: 1}, {Paths: 2, Workers: -1}, {Paths: 2, Quantiles: []float64{.5, .5}}, {Paths: 2, Quantiles: []float64{.7, .3}}, {Paths: 2, Quantiles: []float64{math.NaN()}}, {Paths: 2, Quantiles: []float64{1.1}}} {
		r, e := montecarlo.Run(context.Background(), p, oracleSampler())
		mustErrorIs(t, e, montecarlo.ErrInvalidOption)
		if r != nil {
			t.Fatal("non-nil invalid result")
		}
	}
	r, e := montecarlo.Run(nil, montecarlo.Plan{Paths: 2}, oracleSampler())
	mustErrorIs(t, e, montecarlo.ErrInvalidOption)
	if r != nil {
		t.Fatal("non-nil")
	}
	r, e = montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 2}, (*funcSampler)(nil))
	mustErrorIs(t, e, montecarlo.ErrNilInput)
	if r != nil {
		t.Fatal("non-nil")
	}
}
func TestPreCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r, e := montecarlo.Run(ctx, montecarlo.Plan{Paths: 2}, nil)
	mustErrorIs(t, e, context.Canceled)
	if r != nil {
		t.Fatal("nil result required")
	}
}

func TestQuantileExactBudgetBoundary(t *testing.T) {
	s := &funcSampler{2, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		return []float64{float64(id), 2 * float64(id)}, nil
	}}
	// 100*2*8 raw + 100*8 scratch + 1*2*8 quantile output.
	r, e := montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 100, Quantiles: []float64{.5}, MaxSamplesBytes: 2416}, s)
	if e != nil || r == nil || !r.Complete {
		t.Fatalf("expected exact budget acceptance: %v", e)
	}
}

// TestBudgetArithmeticOverflowBeforeSampling protects overflow in raw byte
// multiplication and checked addition of raw plus sorting scratch storage.
func TestBudgetArithmeticOverflowBeforeSampling(t *testing.T) {
	var called atomic.Int64
	sampler := &funcSampler{1, func(_ context.Context, _ uint64, _ rng.RNG) ([]float64, error) {
		called.Add(1)
		return []float64{1}, nil
	}}
	for _, plan := range []montecarlo.Plan{
		// 8*Paths exceeds MaxUint64, even though Paths itself fits native int.
		{Paths: math.MaxUint64/8 + 1, RetainSamples: true, MaxSamplesBytes: math.MaxUint64},
		// Both raw and sorting scratch fit alone, but their sum overflows.
		{Paths: math.MaxUint64 / 8, Quantiles: []float64{.5}, MaxSamplesBytes: math.MaxUint64},
		// A default-budget rejection must happen before any full allocation.
		{Paths: (64<<20)/8 + 1, RetainSamples: true},
	} {
		result, err := montecarlo.Run(context.Background(), plan, sampler)
		if result != nil || called.Load() != 0 {
			t.Fatal("invalid checked allocation reached sampling")
		}
		if plan.MaxSamplesBytes == 0 {
			mustErrorIs(t, err, montecarlo.ErrBudget)
		} else {
			mustErrorIs(t, err, montecarlo.ErrOverflow)
		}
	}
}

// TestWorkerCountOverflowPrecedesAllWork rejects a window capacity which
// cannot be represented safely by an int-sized channel/map bound.
func TestWorkerCountOverflowPrecedesAllWork(t *testing.T) {
	var called atomic.Int64
	sampler := &funcSampler{1, func(_ context.Context, _ uint64, _ rng.RNG) ([]float64, error) {
		called.Add(1)
		return []float64{1}, nil
	}}
	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
		Paths: math.MaxUint64, Workers: math.MaxInt,
	}, sampler)
	mustErrorIs(t, err, montecarlo.ErrOverflow)
	if result != nil || called.Load() != 0 {
		t.Fatal("overflowed worker window reached sampling")
	}
}

// TestWorkerRequestExceedsPathCount checks the documented clamp to Paths
// without silently converting large uint64 values to native int.
func TestWorkerRequestExceedsPathCount(t *testing.T) {
	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
		Paths: 2, Workers: 500,
	}, oracleSampler())
	if err != nil || result == nil || result.Completed != 2 || !result.Complete {
		t.Fatalf("expected two completed paths with worker clamp: %+v %v", result, err)
	}
}
