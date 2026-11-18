// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm_test checks the public GBM-to-Monte-Carlo integration boundary.
// No production GBM package depends on Monte Carlo; integration lives in consumers.
package gbm_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/montecarlo"
)

// TestMonteCarloWorkerCountReplay checks that path-owned streams and deterministic
// driver reduction preserve the option estimate under different worker counts.
func TestMonteCarloWorkerCountReplay(t *testing.T) {
	const paths = 1024
	const maturity = 0.75
	sampler := europeanCallSampler{
		model:    gbm.Parameters{Initial: 100, Drift: 0.03, Volatility: 0.23, Unit: gbm.TimeYears},
		maturity: maturity, strike: 105, rate: 0.04,
	}
	plan := montecarlo.Plan{Paths: paths, Seed: 191919, Workers: 1}
	serial, err := montecarlo.Run(context.Background(), plan, sampler)
	if err != nil {
		t.Fatal(err)
	}
	plan.Workers = 4
	parallel, err := montecarlo.Run(context.Background(), plan, sampler)
	if err != nil {
		t.Fatal(err)
	}
	if serial == nil || parallel == nil || !serial.Complete || !parallel.Complete {
		t.Fatalf("incomplete MC result serial=%+v parallel=%+v", serial, parallel)
	}
	if math.Float64bits(serial.Mean[0]) != math.Float64bits(parallel.Mean[0]) {
		t.Fatalf("worker-dependent mean: %g vs %g", serial.Mean[0], parallel.Mean[0])
	}
	if serial.Mean[0] <= 0 {
		t.Fatal("positive-volatility call should have a positive sampled premium")
	}
}

// TestOptionSamplerPreservesCancellation ensures an incomplete GBM trajectory
// is not converted into a successful terminal payoff by the sampler adapter.
func TestOptionSamplerPreservesCancellation(t *testing.T) {
	ctx := &cancelAfterFirstCheck{Context: context.Background()}
	sampler := europeanCallSampler{
		model:    gbm.Parameters{Initial: 100, Drift: 0.03, Volatility: 0.2, Unit: gbm.TimeYears},
		maturity: 1, strike: 100, rate: 0.04,
	}
	value, err := sampler.Sample(ctx, 42, &scripted{})
	mustErrorIs(t, err, context.Canceled)
	if value != nil {
		t.Fatalf("canceled GBM path must not supply Monte Carlo payoff: %v", value)
	}
}
