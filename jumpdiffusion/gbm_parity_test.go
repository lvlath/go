// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion_test verifies intensity-zero GBM parity and draw counts.
// These tests compare the exported facades on independent copies of one RNG stream.
package jumpdiffusion_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/jumpdiffusion"
	"github.com/lvlath/go/rng"
)

type countingRNG struct {
	under                    rng.RNG
	normals, uniforms, words int
}

func (r *countingRNG) Uint64() uint64       { r.words++; return r.under.Uint64() }
func (r *countingRNG) Normal() float64      { r.normals++; return r.under.Normal() }
func (r *countingRNG) UniformOpen() float64 { r.uniforms++; return r.under.UniformOpen() }
func TestJD01_GBMBitParity(t *testing.T) {
	for _, sigma := range []float64{0, 0.01, 0.4} {
		p := jdParams()
		p.Diffusion.Volatility = sigma
		p.Intensity = 0
		p.LogJumpMean = 30
		p.LogJumpStdDev = 2
		r1 := &countingRNG{under: rng.NewStream(119, 23)}
		r2 := &countingRNG{under: rng.NewStream(119, 23)}
		a, e1 := jumpdiffusion.Simulate(context.Background(), p, jdGrid(), r1)
		b, e2 := gbm.Simulate(context.Background(), p.Diffusion, jdGrid(), r2)
		if e1 != nil || e2 != nil {
			t.Fatalf("sigma %g errors %v %v", sigma, e1, e2)
		}
		mustEqualInt(t, len(a.Values), len(b.Values), "path shape")
		for i := range a.Values {
			if math.Float64bits(a.Values[i]) != math.Float64bits(b.Values[i]) || math.Float64bits(a.Times[i]) != math.Float64bits(b.Times[i]) {
				t.Fatalf("sigma=%g index=%d: parity failed", sigma, i)
			}
		}
		if r1.normals != r2.normals || r1.uniforms != r2.uniforms || r1.words != r2.words {
			t.Fatalf("draw count differs: %+v vs %+v", r1, r2)
		}
		if r1.uniforms != 0 {
			t.Fatalf("unexpected jump uniforms: %d", r1.uniforms)
		}
	}
}

// TestLambdaZeroSkipsUnrepresentableCompensator verifies strict GBM delegation.
// The finite jump inputs would overflow k if it were computed unnecessarily.
func TestLambdaZeroSkipsUnrepresentableCompensator(t *testing.T) {
	p := jdParams()
	p.Intensity = 0
	p.LogJumpMean = 1000
	p.LogJumpStdDev = 1e154
	streamA := &countingRNG{under: rng.NewStream(52, 9)}
	streamB := &countingRNG{under: rng.NewStream(52, 9)}
	jumpPath, jumpErr := jumpdiffusion.Simulate(context.Background(), p, jdGrid(), streamA)
	gbmPath, gbmErr := gbm.Simulate(context.Background(), p.Diffusion, jdGrid(), streamB)
	if jumpErr != nil || gbmErr != nil || jumpPath == nil || gbmPath == nil {
		t.Fatalf("lambda zero incorrectly inspected compensator: %v %v", jumpErr, gbmErr)
	}
	for index := range jumpPath.Values {
		if math.Float64bits(jumpPath.Values[index]) != math.Float64bits(gbmPath.Values[index]) {
			t.Fatalf("GBM output bit mismatch at %d", index)
		}
	}
	if streamA.uniforms != 0 || streamA.normals != streamB.normals {
		t.Fatal("lambda zero changed GBM consumption")
	}
}
