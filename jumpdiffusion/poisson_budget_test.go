// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion_test audits rejection budgets and adversarial PTRS proposals.
// All count observations are decoded through the public price trajectory.
package jumpdiffusion_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/jumpdiffusion"
)

func poissonFixture(mean float64) jumpdiffusion.Parameters {
	p := jdParams()
	p.Diffusion.Initial = 1
	p.Diffusion.Drift = 0
	p.Diffusion.Volatility = 0
	p.Intensity = mean
	p.LogJumpMean = 1e-12
	p.LogJumpStdDev = 0
	p.Convention = jumpdiffusion.Uncompensated
	return p
}

func TestPoissonInversionExhaustionIsBounded(t *testing.T) {
	p := poissonFixture(0.1)
	stream := &scriptedRNG{uniforms: []float64{math.Nextafter(1, 0)}}
	result, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, stream)
	mustErrorIs(t, err, jumpdiffusion.ErrBudget)
	if result == nil || result.Complete || len(result.Values) != 1 {
		t.Fatalf("budget rejection leaked output: %+v", result)
	}
	mustEqualInt(t, stream.ui, 1<<20, "inversion attempt limit")
	mustEqualInt(t, stream.ni, 0, "inversion normal count")
}

func TestPoissonPTRSExhaustionIsBounded(t *testing.T) {
	p := poissonFixture(30)
	stream := &scriptedRNG{uniforms: []float64{0.5, 0.999}}
	result, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, stream)
	mustErrorIs(t, err, jumpdiffusion.ErrBudget)
	if result == nil || result.Complete || len(result.Times) != 1 {
		t.Fatalf("PTRS budget rejection leaked output: %+v", result)
	}
	mustEqualInt(t, stream.ui, 2*(1<<20), "PTRS attempt limit")
}

func TestPoissonPTRSRejectsOversizedCount(t *testing.T) {
	p := poissonFixture(float64(uint64(1) << 40))
	stream := &scriptedRNG{uniforms: []float64{math.Nextafter(1, 0), 0.5}}
	result, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, stream)
	mustErrorIs(t, err, jumpdiffusion.ErrDomain)
	if result == nil || result.Complete || len(result.Values) != 1 {
		t.Fatalf("oversized count leaked output: %+v", result)
	}
	mustEqualInt(t, stream.ui, 2, "oversized proposal draws")
}

func TestPoissonPTRSRejectsBeforeNextAcceptedProposal(t *testing.T) {
	cases := []struct {
		name     string
		mean     float64
		uniforms []float64
	}{
		{"zero width region", 30, []float64{math.SmallestNonzeroFloat64, 0.5, 0.5, 0.5}},
		{"negative proposal", 30, []float64{0.001, 0.5, 0.5, 0.5}},
		{"thin rejected band", 50, []float64{0.012, 0.5, 0.5, 0.5}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stream := &scriptedRNG{uniforms: tc.uniforms}
			res, err := jumpdiffusion.SimulateTerminal(context.Background(), poissonFixture(tc.mean), 1, stream)
			if err != nil || !res.Complete {
				t.Fatalf("rejected PTRS proposal prevented subsequent acceptance: %+v %v", res, err)
			}
			mustEqualInt(t, stream.ui, 4, "two complete PTRS attempts")
		})
	}
}
