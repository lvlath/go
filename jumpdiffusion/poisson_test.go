// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion_test checks the Poisson distribution and PTRS boundaries.
// Count oracles infer deterministic log jumps through the public simulation facade.
package jumpdiffusion_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/jumpdiffusion"
	"github.com/lvlath/go/rng"
)

// The observed log multiplier is a reversible count oracle for s=0,
// uncompensated drift=0, and a sufficiently small nonzero m.
func TestJD06_PoissonMomentBoundary(t *testing.T) {
	const sampleSize = 15000
	const m = 0.0001
	for _, a := range []float64{29.999, 30} {
		p := jdParams()
		p.Intensity = a
		p.Diffusion.Initial = 1
		p.Diffusion.Volatility = 0
		p.Diffusion.Drift = 0
		p.LogJumpMean = m
		p.LogJumpStdDev = 0
		p.Convention = jumpdiffusion.Uncompensated
		sum, sum2 := 0., 0.
		for pathID := 0; pathID < sampleSize; pathID++ {
			out, err := jumpdiffusion.Simulate(context.Background(), p, oneYear(), rng.NewStream(831, uint64(pathID)))
			if err != nil {
				t.Fatal(err)
			}
			n := math.Round(math.Log(out.Values[1]) / m)
			sum += n
			sum2 += n * n
		}
		mean := sum / sampleSize
		variance := sum2/sampleSize - mean*mean
		if math.Abs(mean-a) > 0.7 || math.Abs(variance-a) > 2.2 {
			t.Fatalf("Poisson(%g) mean=%g variance=%g", a, mean, variance)
		}
	}
}
func TestRNGContractRejectsBadUniform(t *testing.T) {
	for _, v := range []float64{0, 1, math.NaN(), math.Inf(1)} {
		p := jdParams()
		p.Intensity = 30
		p.Diffusion.Volatility = 0
		_, err := jumpdiffusion.Simulate(context.Background(), p, oneYear(), &scriptedRNG{uniforms: []float64{v}})
		if v == 0 || v == 1 {
			mustErrorIs(t, err, jumpdiffusion.ErrDomain)
		} else {
			mustErrorIs(t, err, jumpdiffusion.ErrNonFinite)
		}
	}
}

// TestPTRS_HighPrecisionAcceptanceFixtures uses independent 90-digit mpmath
// log-gamma comparisons. In the 2^40 case direct float64 log-gamma loses
// roughly 1e-3 in the log acceptance margin and must not replace deviance.
func TestPTRS_HighPrecisionAcceptanceFixtures(t *testing.T) {
	cases := []struct {
		name              string
		a, firstU, firstV float64
		expected          uint64
		draws             int
	}{
		{"mean30 reject then squeeze", 30, 0.5, 0.99, 30, 4},
		{"mean30 accepted logPMF", 30, 0.5, 0.94, 30, 2},
		{"maxMean reject then squeeze", float64(uint64(1) << 40), 0.5, 0.99, uint64(1) << 40, 4},
		{"maxMean accepted logPMF", float64(uint64(1) << 40), 0.6, 0.99, 1099511926001, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := jdParams()
			p.Diffusion.Initial = 1
			p.Diffusion.Volatility = 0
			p.Diffusion.Drift = 0
			p.Intensity = tc.a
			p.LogJumpMean = 1e-13
			p.LogJumpStdDev = 0
			p.Convention = jumpdiffusion.Uncompensated
			r := &scriptedRNG{uniforms: []float64{tc.firstU, tc.firstV, 0.5, 0.5}}
			out, err := jumpdiffusion.Simulate(context.Background(), p, oneYear(), r)
			if err != nil {
				t.Fatal(err)
			}
			got := math.Log(out.Values[1]) / p.LogJumpMean
			if math.Abs(got-float64(tc.expected)) > 0.02 {
				t.Fatalf("count %.8f, want %d", got, tc.expected)
			}
			mustEqualInt(t, r.ui, tc.draws, "PTRS uniform draws")
		})
	}
}

// TestPoissonBranchFrequencyOracle compares observed integer-count bins with
// independent analytic Poisson probabilities computed using math.Lgamma.
// The probabilities are evaluated only at small counts, where no unstable
// large-mean gamma subtraction appears in this test oracle.
func TestPoissonBranchFrequencyOracle(t *testing.T) {
	const (
		sampleSize = 12_000
		logJump    = 0.01
		seed       = uint64(11_321)
	)
	for _, mean := range []float64{29.999, 30} {
		p := jdParams()
		p.Diffusion.Initial = 1
		p.Diffusion.Drift = 0
		p.Diffusion.Volatility = 0
		p.Intensity = mean
		p.LogJumpMean = logJump
		p.LogJumpStdDev = 0
		p.Convention = jumpdiffusion.Uncompensated

		counts := make(map[int]int)
		for pathID := uint64(0); pathID < sampleSize; pathID++ {
			res, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, rng.NewStream(seed, pathID))
			if err != nil || res == nil || !res.Complete {
				t.Fatalf("mean=%g pathID=%d: %v", mean, pathID, err)
			}
			decoded := math.Log(res.Values[1]) / logJump
			nearest := math.Round(decoded)
			if math.Abs(decoded-nearest) > 1e-8 {
				t.Fatalf("noninteger inferred count %.12g", decoded)
			}
			counts[int(nearest)]++
		}

		for _, count := range []int{24, 30, 36, 45} {
			logFactorial, _ := math.Lgamma(float64(count + 1))
			mass := math.Exp(-mean + float64(count)*math.Log(mean) - logFactorial)
			expected := sampleSize * mass
			standardDeviation := math.Sqrt(expected * (1 - mass))
			if math.Abs(float64(counts[count])-expected) > 6*standardDeviation+10 {
				t.Fatalf("mean=%.3f count=%d observed=%d expected=%.3f",
					mean, count, counts[count], expected)
			}
		}
	}
}

// TestPTRS_IndependentAcceptanceThresholds checks the exact-count rejection
// decision against loggamma values computed independently at 90 decimal digits.
// Each first candidate lies outside the fast-squeeze region; values immediately
// below and above the high-precision acceptance boundary must diverge.
func TestPTRS_IndependentAcceptanceThresholds(t *testing.T) {
	cases := []struct {
		name     string
		mean     float64
		first    float64
		second   float64
		accepted bool
		count    uint64
	}{
		// For mean=30 and first=0.2, v_critical=0.9398229228..., count=25.
		{"30 accepted", 30, 0.2, 0.9398, true, 25},
		{"30 rejected", 30, 0.2, 0.93985, false, 30},
		// For mean=50 and first=0.4, v_critical=0.9852906938..., count=48.
		{"50 accepted", 50, 0.4, 0.98528, true, 48},
		{"50 rejected", 50, 0.4, 0.98530, false, 50},
		// At the maximum mean the stable mass is decisive, not a float64 Lgamma.
		// first=0.45 yields v_critical=0.9982757299..., count=1099511480493.
		{"2^40 accepted", float64(uint64(1) << 40), 0.45, 0.99827, true, 1099511480493},
		{"2^40 rejected", float64(uint64(1) << 40), 0.45, 0.99828, false, uint64(1) << 40},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			params := poissonFixture(tc.mean)
			stream := &scriptedRNG{uniforms: []float64{tc.first, tc.second, 0.5, 0.5}}
			result, err := jumpdiffusion.SimulateTerminal(context.Background(), params, 1, stream)
			if err != nil || result == nil || !result.Complete {
				t.Fatalf("PTRS fixture failed: %+v %v", result, err)
			}
			wantDraws := 4
			if tc.accepted {
				wantDraws = 2
			}
			mustEqualInt(t, stream.ui, wantDraws, "PTRS proposal draws")
			inferredCount := math.Log(result.Values[1]) / params.LogJumpMean
			if math.Abs(inferredCount-float64(tc.count)) > 0.03 {
				t.Fatalf("count %.7f does not match independent oracle %d", inferredCount, tc.count)
			}
		})
	}
}
