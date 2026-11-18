// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion_test protects cross-package sentinel identity at lambda=0.
// Only externally visible errors and completed prefixes are inspected.
package jumpdiffusion_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/jumpdiffusion"
)

func TestGBMOverflowPreservesBothSentinels(t *testing.T) {
	p := jdParams()
	p.Intensity = 0
	p.Diffusion.Drift = 1000
	p.Diffusion.Volatility = 0
	out, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, &scriptedRNG{})
	mustErrorIs(t, err, jumpdiffusion.ErrOverflow)
	mustErrorIs(t, err, gbm.ErrOverflow)
	if out == nil || out.Complete || len(out.Values) != 1 || out.Values[0] != p.Diffusion.Initial {
		t.Fatalf("GBM overflow corrupted prefix: %+v", out)
	}
}

func TestGBMNonFiniteDrawPreservesBothSentinels(t *testing.T) {
	p := jdParams()
	p.Intensity = 0
	p.Diffusion.Volatility = 0.2
	stream := &scriptedRNG{normals: []float64{math.NaN()}}
	out, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, stream)
	mustErrorIs(t, err, jumpdiffusion.ErrNonFinite)
	mustErrorIs(t, err, gbm.ErrNonFinite)
	if out == nil || out.Complete || len(out.Values) != 1 || stream.ui != 0 || stream.ni != 1 {
		t.Fatalf("GBM bad draw corrupted prefix or consumed jump RNG: %+v %+v", out, stream)
	}
}
