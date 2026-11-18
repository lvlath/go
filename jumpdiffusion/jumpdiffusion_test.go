// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion_test tests Merton invariants and public input contracts.
// Fixed-seed statistical oracles complement independent analytic assertions.
package jumpdiffusion_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/jumpdiffusion"
	"github.com/lvlath/go/rng"
)

func oneYear() gbm.TimeGrid { return gbm.TimeGrid{Values: []float64{0, 1}, Unit: gbm.TimeYears} }
func TestJD02_ThreeDeterministicJumps(t *testing.T) {
	p := jdParams()
	p.Intensity = 1
	p.Diffusion.Initial = 1
	p.Diffusion.Drift = 0
	p.Diffusion.Volatility = 0
	p.LogJumpMean = math.Ln2
	p.LogJumpStdDev = 0
	p.Convention = jumpdiffusion.Uncompensated
	r := &scriptedRNG{uniforms: []float64{0.9, 0.9, 0.9, 0.1}}
	path, err := jumpdiffusion.Simulate(context.Background(), p, oneYear(), r)
	if err != nil {
		t.Fatal(err)
	}
	mustApprox(t, path.Values[1], 8, 1e-14)
	mustEqualInt(t, r.ni, 0, "jump normal count")
	mustEqualInt(t, r.ui, 4, "poisson draws")
}
func TestJD03_ZeroSizeJumps(t *testing.T) {
	p := jdParams()
	p.Diffusion.Volatility = 0
	p.LogJumpMean = 0
	p.LogJumpStdDev = 0
	p.Convention = jumpdiffusion.Compensated
	r := &scriptedRNG{uniforms: []float64{0.5}}
	path, err := jumpdiffusion.Simulate(context.Background(), p, jdGrid(), r)
	if err != nil {
		t.Fatal(err)
	}
	for i, tv := range jdGrid().Values {
		mustApprox(t, path.Values[i], 100*math.Exp(p.Diffusion.Drift*tv), 1e-12)
	}
	mustEqualInt(t, r.ni, 0, "zero normal calls")
}
func TestJD04_Expm1NearZero(t *testing.T) {
	p := jdParams()
	p.Diffusion.Initial = 1
	p.Diffusion.Drift = 0
	p.Diffusion.Volatility = 0
	p.Intensity = 1
	p.LogJumpMean = 1e-12
	p.LogJumpStdDev = 0
	r := &scriptedRNG{uniforms: []float64{0.1}}
	path, err := jumpdiffusion.Simulate(context.Background(), p, oneYear(), r)
	if err != nil {
		t.Fatal(err)
	}
	want := math.Exp(-math.Expm1(1e-12))
	mustApprox(t, path.Values[1], want, 5e-15)
}
func TestJD07_PoissonBoundBeforeAnyDraw(t *testing.T) {
	p := jdParams()
	p.Intensity = float64(uint64(1)<<40) + 1
	p.Diffusion.Volatility = 0.2
	r := &scriptedRNG{}
	res, err := jumpdiffusion.Simulate(context.Background(), p, oneYear(), r)
	mustErrorIs(t, err, jumpdiffusion.ErrDomain)
	if res == nil || len(res.Values) != 1 || res.Complete {
		t.Fatalf("invalid prefix %#v", res)
	}
	mustEqualInt(t, r.ni, 0, "no diffusion draw")
	mustEqualInt(t, r.ui, 0, "no poisson draw")
}
func TestResult_IsNilInterface(t *testing.T) {
	var nilResult *jumpdiffusion.Result
	var i interface{ IsNil() bool } = nilResult
	mustNilState(t, i, true, "nil result")
	mustNilState(t, &jumpdiffusion.Result{}, false, "allocated result")
}
func TestValidation(t *testing.T) {
	cases := []struct {
		name string
		edit func(*jumpdiffusion.Parameters, *gbm.TimeGrid)
		want error
	}{
		{"intensity negative", func(p *jumpdiffusion.Parameters, _ *gbm.TimeGrid) { p.Intensity = -1 }, jumpdiffusion.ErrDomain},
		{"jump mean nan", func(p *jumpdiffusion.Parameters, _ *gbm.TimeGrid) { p.LogJumpMean = math.NaN() }, jumpdiffusion.ErrNonFinite},
		{"std negative", func(p *jumpdiffusion.Parameters, _ *gbm.TimeGrid) { p.LogJumpStdDev = -1 }, jumpdiffusion.ErrDomain},
		{"unit mismatch", func(_ *jumpdiffusion.Parameters, g *gbm.TimeGrid) { g.Unit = gbm.TimeSeconds }, jumpdiffusion.ErrInvalidOption},
		{"bad convention", func(p *jumpdiffusion.Parameters, _ *gbm.TimeGrid) { p.Convention = 2 }, jumpdiffusion.ErrInvalidOption},
		{"empty grid", func(_ *jumpdiffusion.Parameters, g *gbm.TimeGrid) { g.Values = []float64{} }, jumpdiffusion.ErrEmptyInput},
		{"nil grid", func(_ *jumpdiffusion.Parameters, g *gbm.TimeGrid) { g.Values = nil }, jumpdiffusion.ErrNilInput},
		{"order", func(_ *jumpdiffusion.Parameters, g *gbm.TimeGrid) { g.Values = []float64{0, 1, 1} }, jumpdiffusion.ErrOrder},
		{"bad timestamp", func(_ *jumpdiffusion.Parameters, g *gbm.TimeGrid) { g.Values = []float64{0, math.Inf(1)} }, jumpdiffusion.ErrNonFinite},
		{"bad start", func(p *jumpdiffusion.Parameters, _ *gbm.TimeGrid) { p.Diffusion.Initial = 0 }, jumpdiffusion.ErrDomain},
		{"compensator overflow", func(p *jumpdiffusion.Parameters, _ *gbm.TimeGrid) { p.LogJumpMean = 1000 }, jumpdiffusion.ErrOverflow},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, g := jdParams(), jdGrid()
			tc.edit(&p, &g)
			got, err := jumpdiffusion.Simulate(context.Background(), p, g, rng.NewStream(1, 1))
			mustErrorIs(t, err, tc.want)
			if got != nil {
				t.Fatalf("invalid input must return nil result: %#v", got)
			}
		})
	}
	if got, err := jumpdiffusion.Simulate(nil, jdParams(), jdGrid(), rng.NewStream(1, 1)); got != nil {
		t.Fatalf("got result %#v err %v", got, err)
	} else {
		mustErrorIs(t, err, jumpdiffusion.ErrInvalidOption)
	}
}
func TestOutputOwnership(t *testing.T) {
	p, g := jdParams(), jdGrid()
	out, err := jumpdiffusion.Simulate(context.Background(), p, g, rng.NewStream(1, 2))
	if err != nil {
		t.Fatal(err)
	}
	g.Values[0] = 100
	g.Values[1] = 123
	if out.Times[0] != 0 || out.Times[1] != 0.25 {
		t.Fatal("output grid aliases input")
	}
}

// TestJD05_CompensatedMartingaleExpectation checks the exact analytic mean
// with a predeclared, broad deterministic Monte Carlo tolerance.
func TestJD05_CompensatedMartingaleExpectation(t *testing.T) {
	const paths = 20000
	p := jdParams()
	p.Diffusion.Initial = 100
	p.Diffusion.Drift = 0
	p.Diffusion.Volatility = 0
	p.Intensity = 1.2
	p.LogJumpMean = -0.12
	p.LogJumpStdDev = 0.08
	p.Convention = jumpdiffusion.Compensated
	total := 0.0
	for i := 0; i < paths; i++ {
		out, err := jumpdiffusion.Simulate(context.Background(), p, oneYear(), rng.NewStream(3001, uint64(i)))
		if err != nil {
			t.Fatal(err)
		}
		total += out.Values[1]
	}
	sampleMean := total / paths
	// The independent Poisson exponential-moment identity gives the exact
	// variance and therefore a preregistered six-standard-error bound.
	k := math.Expm1(p.LogJumpMean + p.LogJumpStdDev*p.LogJumpStdDev/2)
	secondJumpMoment := math.Exp(2*p.LogJumpMean + 2*p.LogJumpStdDev*p.LogJumpStdDev)
	relativeVariance := math.Expm1(p.Intensity * (secondJumpMoment - 1 - 2*k))
	standardError := p.Diffusion.Initial * math.Sqrt(relativeVariance/paths)
	if math.Abs(sampleMean-100) > 6*standardError {
		t.Fatalf("compensated mean %.10g outside six standard errors %.10g", sampleMean, 6*standardError)
	}
}

// TestPriceNearSubnormalBoundary protects the anchored conversion against
// premature float64 exponential underflow.
func TestPriceNearSubnormalBoundary(t *testing.T) {
	p := jdParams()
	p.Diffusion.Initial = math.SmallestNonzeroFloat64
	p.Diffusion.Volatility = 0
	p.Diffusion.Drift = 1
	p.Intensity = 1
	p.LogJumpMean = 0
	p.LogJumpStdDev = 0
	p.Convention = jumpdiffusion.Uncompensated
	result, err := jumpdiffusion.Simulate(context.Background(), p, oneYear(), &scriptedRNG{uniforms: []float64{0.1}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Values[1] <= 0 || !result.Complete {
		t.Fatalf("invalid subnormal result %+v", result)
	}
}
