// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test checks adversarial HMM inference against independent analytic
// and rational oracles through the exported API only.
package hmm_test

import (
	"context"
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

// auditGaussian creates a detached, explicit Gaussian model without invoking private kernels.
func auditGaussian(t *testing.T, prior []float64, transition, means, variances [][]float64) *hmm.GaussianModel {
	t.Helper()
	m, err := hmm.NewGaussian(hmm.GaussianSpec{
		Common: hmm.Common{Prior: prior, Transition: dense(t, transition)},
		Means:  dense(t, means), Variances: dense(t, variances),
	})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestAuditGaussianPosteriorNormalization protects causality and checkpoint restoration.
// A common large negative Gaussian log density cannot erase relative state probabilities.
func TestAuditGaussianPosteriorNormalization(t *testing.T) {
	m := auditGaussian(t, []float64{.5, .5}, [][]float64{{1, 0}, {0, 1}}, [][]float64{{0}, {0}}, [][]float64{{1}, {1}})
	for _, x := range []float64{1000, 1e8} {
		state, err := m.FilterStep(context.Background(), nil, []float64{x})
		if err != nil {
			t.Fatalf("x=%g: %v", x, err)
		}
		if state == nil {
			t.Fatal("nil successful state")
		}
		sum := 0.0
		for j, posterior := range state.LogPosterior {
			if math.Abs(posterior-math.Log(.5)) > 1e-12 {
				t.Fatalf("x=%g j=%d posterior=%g", x, j, posterior)
			}
			sum += math.Exp(posterior)
		}
		if math.Abs(sum-1) > 1e-12 {
			t.Fatalf("x=%g sum=%g", x, sum)
		}
		restored := *state
		restored.LogPosterior = append([]float64(nil), state.LogPosterior...)
		next, err := m.FilterStep(context.Background(), &restored, []float64{x})
		if err != nil || next == nil {
			t.Fatalf("x=%g replay: %v", x, err)
		}
		if math.Abs(math.Exp(next.LogPosterior[0])-.5) > 1e-12 {
			t.Fatalf("next posterior=%v", next.LogPosterior)
		}
		if restored.Steps != 1 || restored.LogPosterior[0] != state.LogPosterior[0] {
			t.Fatal("mutated prev")
		}
	}
	biased := auditGaussian(t, []float64{.7, .3}, [][]float64{{1, 0}, {0, 1}}, [][]float64{{0}, {0}}, [][]float64{{1}, {1}})
	prefix, err := biased.Filter(context.Background(), dense(t, [][]float64{{1e8}}), hmm.FilterOptions{ReturnHistory: true})
	if err != nil {
		t.Fatal(err)
	}
	long, err := biased.Filter(context.Background(), dense(t, [][]float64{{1e8}, {1000}}), hmm.FilterOptions{ReturnHistory: true})
	if err != nil {
		t.Fatal(err)
	}
	for j, want := range []float64{.7, .3} {
		v, err := prefix.Posterior.At(0, j)
		if err != nil {
			t.Fatal(err)
		}
		next, err := long.Posterior.At(0, j)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(v-want) > 1e-12 || math.Float64bits(v) != math.Float64bits(next) {
			t.Fatalf("prefix j=%d got=%g next=%g", j, v, next)
		}
	}
	smoothed, err := biased.Smooth(context.Background(), dense(t, [][]float64{{1000}, {1e8}}))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		for j, want := range []float64{.7, .3} {
			v, e := smoothed.Posterior.At(i, j)
			if e != nil {
				t.Fatal(e)
			}
			if math.Abs(v-want) > 1e-12 {
				t.Fatalf("smooth[%d,%d]=%g", i, j, v)
			}
		}
	}
}

// TestAuditGaussianRepresentableDifference compares a huge-offset density with
// an exact rational quadratic form over the original finite float64 operands.
func TestAuditGaussianRepresentableDifference(t *testing.T) {
	variance := math.MaxFloat64
	observation, mean := 1.1e308, -1.1e308
	m := auditGaussian(t, []float64{1}, [][]float64{{1}}, [][]float64{{mean}}, [][]float64{{variance}})
	state, err := m.FilterStep(context.Background(), nil, []float64{observation})
	if err != nil {
		t.Fatal(err)
	}
	if state == nil || math.IsInf(state.LogLikelihood, 0) || math.IsNaN(state.LogLikelihood) {
		t.Fatal("invalid finite density")
	}
	x := new(big.Rat).SetFloat64(observation)
	mu := new(big.Rat).SetFloat64(mean)
	delta := new(big.Rat).Sub(x, mu)
	delta.Mul(delta, delta)
	twiceVariance := new(big.Rat).Mul(new(big.Rat).SetFloat64(variance), big.NewRat(2, 1))
	exactQuadratic := new(big.Rat).Quo(delta, twiceVariance)
	rounded, _ := new(big.Float).SetPrec(512).SetRat(exactQuadratic).Float64()
	want := -rounded - .5*(math.Log(2*math.Pi)+math.Log(variance))
	if math.Abs((state.LogLikelihood-want)/want) > 2e-15 {
		t.Fatalf("density got=%g want=%g", state.LogLikelihood, want)
	}
}

// TestAuditGaussianDecodeOverflowClass separates reachable overflowing paths
// from structurally impossible paths and checks that a finite winner survives.
func TestAuditGaussianDecodeOverflowClass(t *testing.T) {
	single := auditGaussian(t, []float64{1}, [][]float64{{1}}, [][]float64{{0}}, [][]float64{{1}})
	observations := dense(t, [][]float64{{1.5e154}, {1.5e154}})
	decoded, err := single.Decode(context.Background(), observations)
	if decoded != nil || !errors.Is(err, hmm.ErrOverflow) || errors.Is(err, hmm.ErrImpossibleSequence) {
		t.Fatalf("decoded=%v err=%v", decoded, err)
	}
	mixed := auditGaussian(t, []float64{.5, .5}, [][]float64{{1, 0}, {0, 1}}, [][]float64{{0}, {1.5e154}}, [][]float64{{1}, {1}})
	winner, err := mixed.Decode(context.Background(), observations)
	if err != nil || winner == nil {
		t.Fatalf("finite winning path: %v", err)
	}
	if len(winner.States) != 2 || winner.States[0] != 1 || winner.States[1] != 1 || math.IsInf(winner.LogJointProbability, 0) {
		t.Fatalf("winner=%v", winner)
	}
	impossible, err := hmm.NewCategorical(hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})}, Emission: dense(t, [][]float64{{1, 0}})})
	if err != nil {
		t.Fatal(err)
	}
	result, err := impossible.Decode(context.Background(), []uint32{1})
	if result != nil || !errors.Is(err, hmm.ErrImpossibleSequence) || errors.Is(err, hmm.ErrOverflow) {
		t.Fatalf("impossible=%v, err=%v", result, err)
	}
}

// Verify that the audit uses only public matrix input and HMM result contracts.
var _ matrix.Matrix = (*matrix.Dense)(nil)

// TestAuditLogSumExpInvalidInputCannotMasqueradeAsImpossible confirms that an
// invalid NaN is never silently classified as structural zero support.
func TestAuditLogSumExpInvalidInputCannotMasqueradeAsImpossible(t *testing.T) {
	if !math.IsNaN(hmm.LogSumExpSlice([]float64{math.NaN(), math.Inf(-1)})) {
		t.Fatal("invalid input masquerades as impossible vector")
	}
	if !math.IsNaN(hmm.LogSumExp(math.NaN(), math.Inf(1))) {
		t.Fatal("invalid pair masquerades as infinite support")
	}
	if !math.IsInf(hmm.LogSumExpSlice([]float64{math.Inf(-1), math.Inf(-1)}), -1) {
		t.Fatal("all-impossible sentinel lost")
	}
}
