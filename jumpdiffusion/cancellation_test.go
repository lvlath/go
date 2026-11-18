// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion_test verifies prefix publication during cancellation.
// Deterministic stream hooks avoid scheduling-dependent or timeout-based tests.
package jumpdiffusion_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/jumpdiffusion"
)

func TestJD08_CancelInsidePoisson(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	p := jdParams()
	p.Intensity = 30
	p.Diffusion.Volatility = 0
	r := &scriptedRNG{uniforms: []float64{0.5}, onUniform: func() { cancel() }}
	out, err := jumpdiffusion.Simulate(ctx, p, oneYear(), r)
	mustErrorIs(t, err, context.Canceled)
	if out == nil || out.Complete || len(out.Times) != 1 || len(out.Values) != 1 || out.Values[0] != p.Diffusion.Initial {
		t.Fatalf("bad partial result: %#v", out)
	}
}
func TestPreCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out, err := jumpdiffusion.Simulate(ctx, jdParams(), jdGrid(), &scriptedRNG{})
	mustErrorIs(t, err, context.Canceled)
	if out != nil {
		t.Fatalf("pre-canceled path must be nil: %#v", out)
	}
}
