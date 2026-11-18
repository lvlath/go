// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion_test anchors wide arithmetic, compensator and prices.
// Independent analytic price identities protect operations beyond float64 range.
package jumpdiffusion_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/jumpdiffusion"
)

// TestCompensatorWideCancellation covers an exponent whose separate std^2
// overflows float64, but whose corrected exponent and k are representable.
func TestCompensatorWideCancellation(t *testing.T) {
	p := jdParams()
	p.Diffusion.Initial = 1
	p.Diffusion.Drift = 0
	p.Diffusion.Volatility = 0
	p.Intensity = 1
	p.LogJumpMean = -math.MaxFloat64
	p.LogJumpStdDev = 1.1 * math.Sqrt(math.MaxFloat64)
	p.Convention = jumpdiffusion.Compensated

	// With an N=0 scripted Poisson count, k must be -1 and growth +1.
	stream := &scriptedRNG{uniforms: []float64{0.1}}
	res, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, stream)
	if err != nil {
		t.Fatal(err)
	}
	mustApprox(t, res.Values[1], math.E, 2e-15)
	mustEqualInt(t, stream.ui, 1, "single Poisson draw")
}

func TestUncompensatedSkipsUnneededOverflow(t *testing.T) {
	p := jdParams()
	p.Diffusion.Initial = 1
	p.Diffusion.Drift = 0
	p.Diffusion.Volatility = 0
	p.Intensity = 1
	p.LogJumpMean = 1000
	p.LogJumpStdDev = 0
	p.Convention = jumpdiffusion.Uncompensated

	res, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, &scriptedRNG{uniforms: []float64{0.1}})
	if err != nil || !res.Complete || res.Values[1] != 1 {
		t.Fatalf("uncompensated N=0 path rejected: result=%+v err=%v", res, err)
	}

	p.Convention = jumpdiffusion.Compensated
	stream := &scriptedRNG{}
	res, err = jumpdiffusion.SimulateTerminal(context.Background(), p, 1, stream)
	mustErrorIs(t, err, jumpdiffusion.ErrOverflow)
	if res != nil || stream.ui != 0 || stream.ni != 0 {
		t.Fatalf("compensator failure consumed entropy: %+v %+v", res, stream)
	}
}

func TestPriceAnchoredLargePositiveAndNegativeGrowth(t *testing.T) {
	cases := []struct {
		name           string
		initial, drift float64
	}{
		{"small initial, huge gain", math.SmallestNonzeroFloat64, 1000},
		{"large initial, huge loss", math.MaxFloat64, -1000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := jdParams()
			p.Diffusion.Initial = tc.initial
			p.Diffusion.Drift = tc.drift
			p.Diffusion.Volatility = 0
			p.Intensity = 1
			p.LogJumpMean = 0
			p.LogJumpStdDev = 0
			p.Convention = jumpdiffusion.Uncompensated
			res, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, &scriptedRNG{uniforms: []float64{0.1}})
			if err != nil || !res.Complete {
				t.Fatalf("representable extreme price rejected: %+v %v", res, err)
			}
			// Frexp handles even the smallest subnormal exactly; some Go 1.23
			// math.Log implementations lose the subnormal exponent directly.
			fraction, exponent := math.Frexp(tc.initial)
			want := math.Exp(math.Log(fraction) + float64(exponent)*math.Ln2 + tc.drift)
			if math.Abs(res.Values[1]/want-1) > 1e-12 {
				t.Fatalf("extended price %.17g vs analytic %.17g", res.Values[1], want)
			}
		})
	}
}

func TestPriceUnderflowAndOverflowCompletedPrefix(t *testing.T) {
	for _, tc := range []struct {
		initial, drift float64
	}{
		{math.SmallestNonzeroFloat64, -10},
		{math.MaxFloat64, 1000},
	} {
		p := jdParams()
		p.Diffusion.Initial = tc.initial
		p.Diffusion.Drift = tc.drift
		p.Diffusion.Volatility = 0
		p.Intensity = 1
		p.LogJumpMean = 0
		p.LogJumpStdDev = 0
		p.Convention = jumpdiffusion.Uncompensated
		res, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, &scriptedRNG{uniforms: []float64{0.1}})
		mustErrorIs(t, err, jumpdiffusion.ErrOverflow)
		if res == nil || res.Complete || len(res.Values) != 1 || res.Values[0] != p.Diffusion.Initial {
			t.Fatalf("overflow leaked a partially computed price: %+v", res)
		}
	}
}

func TestPoissonProductOverflowIsDomainError(t *testing.T) {
	p := jdParams()
	p.Diffusion.Volatility = 0.1
	p.Intensity = math.MaxFloat64
	p.Convention = jumpdiffusion.Uncompensated
	stream := &scriptedRNG{}
	res, err := jumpdiffusion.SimulateTerminal(context.Background(), p, math.MaxFloat64, stream)
	mustErrorIs(t, err, jumpdiffusion.ErrDomain)
	if res == nil || res.Complete || len(res.Times) != 1 || stream.ni != 0 || stream.ui != 0 {
		t.Fatalf("out-of-domain mean consumed entropy or corrupted prefix: %+v %+v", res, stream)
	}
}

// TestPositiveMeanUnderflowIsNotTreatedAsZero rejects a mathematically positive
// lambda*dt that cannot be represented as any positive binary64 Poisson mean.
// Otherwise an accidental zero branch would silently suppress a contracted
// UniformOpen draw on this positive-intensity trajectory.
func TestPositiveMeanUnderflowIsNotTreatedAsZero(t *testing.T) {
	p := jdParams()
	p.Convention = jumpdiffusion.Uncompensated
	p.Intensity = math.SmallestNonzeroFloat64
	p.Diffusion.Volatility = 0.4
	stream := &scriptedRNG{}
	grid := gbm.TimeGrid{Values: []float64{0, 0.5}, Unit: p.Diffusion.Unit}
	out, err := jumpdiffusion.Simulate(context.Background(), p, grid, stream)
	mustErrorIs(t, err, jumpdiffusion.ErrOverflow)
	if out == nil || out.Complete || len(out.Times) != 1 || len(out.Values) != 1 {
		t.Fatalf("unrepresentable Poisson mean published a step: %+v", out)
	}
	mustEqualInt(t, stream.ui, 0, "no Poisson draw on preflight failure")
	mustEqualInt(t, stream.ni, 0, "no diffusion draw on preflight failure")
}

// TestPoissonMeanAboveBoundWinsBeforeDiffusion checks the exact numeric gate
// using two individually finite binary64 coefficients with a product >2^40.
func TestPoissonMeanAboveBoundWinsBeforeDiffusion(t *testing.T) {
	p := jdParams()
	p.Convention = jumpdiffusion.Uncompensated
	p.Intensity = float64(uint64(1) << 40)
	p.Diffusion.Volatility = 0.7
	grid := gbm.TimeGrid{Values: []float64{0, math.Nextafter(1, 2)}, Unit: p.Diffusion.Unit}
	stream := &scriptedRNG{}
	out, err := jumpdiffusion.Simulate(context.Background(), p, grid, stream)
	mustErrorIs(t, err, jumpdiffusion.ErrDomain)
	if out == nil || out.Complete || len(out.Values) != 1 {
		t.Fatalf("invalid mean published a step: %+v", out)
	}
	mustEqualInt(t, stream.ui, 0, "no Poisson draw")
	mustEqualInt(t, stream.ni, 0, "no diffusion draw")
}

// TestPathwiseCompensationIdentity uses common random numbers to cancel the
// entire stochastic component, isolating the analytical Merton compensator.
// This metamorphic identity is stronger than merely checking empirical means.
func TestPathwiseCompensationIdentity(t *testing.T) {
	p := jdParams()
	p.Intensity = 2.4
	p.LogJumpMean = -0.13
	p.LogJumpStdDev = 0.18
	p.Diffusion.Volatility = 0.27
	grid := gbm.TimeGrid{Values: []float64{0, 0.03, 0.08, 0.13, 0.21, 0.34, 0.55, 0.71, 1}, Unit: p.Diffusion.Unit}

	p.Convention = jumpdiffusion.Compensated
	compensated, err := jumpdiffusion.SimulateSeeded(context.Background(), p, grid, 4321, 17)
	if err != nil || compensated == nil || !compensated.Complete {
		t.Fatalf("compensated path failure: %+v %v", compensated, err)
	}
	p.Convention = jumpdiffusion.Uncompensated
	uncompensated, err := jumpdiffusion.SimulateSeeded(context.Background(), p, grid, 4321, 17)
	if err != nil || uncompensated == nil || !uncompensated.Complete {
		t.Fatalf("uncompensated path failure: %+v %v", uncompensated, err)
	}

	compensator := math.Expm1(p.LogJumpMean + p.LogJumpStdDev*p.LogJumpStdDev/2)
	for index, instant := range grid.Values {
		// Pathwise noise cancels in S_comp/S_uncomp; only drift remains.
		want := math.Exp(-p.Intensity * compensator * instant)
		got := compensated.Values[index] / uncompensated.Values[index]
		if math.Abs(got-want) > 2e-12 {
			t.Fatalf("at t=%g: ratio %.17g differs from drift identity %.17g", instant, got, want)
		}
	}
}
