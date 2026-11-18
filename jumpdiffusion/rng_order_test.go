// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion_test pins down the observable RNG call sequence.
// External fake streams model deterministic cancellation and invalid draws.
package jumpdiffusion_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/jumpdiffusion"
)

// traceRNG records only public RNG interface method invocations.
// Fixed samples let contract tests distinguish diffusion and jump Normals.
type traceRNG struct {
	calls                     []byte
	uniforms, normals         []float64
	uniformIndex, normalIndex int
}

func (r *traceRNG) Uint64() uint64 {
	r.calls = append(r.calls, 'W')
	return 0
}
func (r *traceRNG) UniformOpen() float64 {
	r.calls = append(r.calls, 'U')
	value := r.uniforms[r.uniformIndex]
	r.uniformIndex++
	return value
}
func (r *traceRNG) Normal() float64 {
	r.calls = append(r.calls, 'N')
	value := r.normals[r.normalIndex]
	r.normalIndex++
	return value
}

func TestDiffusionPoissonJumpDrawOrder(t *testing.T) {
	p := jdParams()
	p.Diffusion.Volatility = 0.2
	p.Diffusion.Drift = 0
	p.Intensity = 1
	p.LogJumpMean = 0
	p.LogJumpStdDev = 0.1
	p.Convention = jumpdiffusion.Uncompensated

	stream := &traceRNG{uniforms: []float64{0.999, 0.1}, normals: []float64{0.5, -1}}
	res, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, stream)
	if err != nil || !res.Complete {
		t.Fatalf("unexpected stream error: %+v %v", res, err)
	}
	mustEqualString(t, string(stream.calls), "NUUN", "diffusion-poisson-jump consumption")
}

func TestNoNormalForDeterministicDiffusionAndJumps(t *testing.T) {
	p := jdParams()
	p.Diffusion.Volatility = 0
	p.LogJumpStdDev = 0
	p.LogJumpMean = -0.2
	stream := &traceRNG{uniforms: []float64{0.1}}
	_, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, stream)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualString(t, string(stream.calls), "U", "no unnecessary normal samples")
}

func TestDiffusionFailureConsumesNoPoissonDraw(t *testing.T) {
	p := jdParams()
	stream := &scriptedRNG{normals: []float64{math.NaN()}}
	res, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, stream)
	mustErrorIs(t, err, jumpdiffusion.ErrNonFinite)
	if res == nil || res.Complete || len(res.Values) != 1 || stream.ni != 1 || stream.ui != 0 {
		t.Fatalf("malformed diffusion draw leaked step: %+v %+v", res, stream)
	}
}

func TestJumpFailurePreservesCompletedPrefix(t *testing.T) {
	p := jdParams()
	p.Diffusion.Volatility = 0
	p.Intensity = 1
	p.LogJumpMean = 0
	p.LogJumpStdDev = 0.5
	stream := &scriptedRNG{uniforms: []float64{0.999, 0.1}, normals: []float64{math.NaN()}}
	res, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, stream)
	mustErrorIs(t, err, jumpdiffusion.ErrNonFinite)
	if res == nil || res.Complete || len(res.Values) != 1 || stream.ni != 1 || stream.ui != 2 {
		t.Fatalf("bad jump normal leaked step: %+v %+v", res, stream)
	}
}

func TestCancellationDiscardsSecondStep(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	p := jdParams()
	p.Diffusion.Volatility = 0
	p.Intensity = 1
	p.LogJumpMean = 0
	p.LogJumpStdDev = 0
	stream := &scriptedRNG{uniforms: []float64{0.1}}
	uniformCalls := 0
	stream.onUniform = func() {
		uniformCalls++
		if uniformCalls == 2 {
			cancel()
		}
	}
	grid := gbm.TimeGrid{Values: []float64{0, 1, 2}, Unit: p.Diffusion.Unit}
	res, err := jumpdiffusion.Simulate(ctx, p, grid, stream)
	mustErrorIs(t, err, context.Canceled)
	if res == nil || res.Complete || len(res.Times) != 2 || len(res.Values) != 2 || res.Times[1] != 1 {
		t.Fatalf("canceled step leaked into completed prefix: %+v", res)
	}
	mustEqualInt(t, stream.ui, 2, "consumed uniforms are not rolled back")
}

// TestNoJumpNormalWhenPoissonReturnsZero protects the conditional sampling law.
// A positive jump standard deviation must not consume a Normal at count zero.
func TestNoJumpNormalWhenPoissonReturnsZero(t *testing.T) {
	p := jdParams()
	p.Diffusion.Volatility = 0
	p.LogJumpStdDev = 0.7
	p.Convention = jumpdiffusion.Uncompensated
	stream := &scriptedRNG{uniforms: []float64{0.1}, normals: []float64{math.NaN()}}
	path, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, stream)
	if err != nil || path == nil || !path.Complete {
		t.Fatalf("conditional zero count wrongly drew a Normal: %+v %v", path, err)
	}
	mustEqualInt(t, stream.ni, 0, "conditional jump normals")
}
