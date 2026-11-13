// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
package hmm_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

func TestConstructor_DetachedParametersAndNormalization(t *testing.T) {
	prior := []float64{.5, .5}
	transition := dense(t, [][]float64{{.7, .3}, {.1, .9}})
	emission := dense(t, [][]float64{{.9, .1}, {.2, .8}})
	m, e := hmm.NewCategorical(hmm.CategoricalSpec{Common: hmm.Common{Prior: prior, Transition: transition}, Emission: emission})
	if e != nil {
		t.Fatal(e)
	}
	before, e := m.Filter(context.Background(), []uint32{0, 1}, hmm.FilterOptions{})
	if e != nil {
		t.Fatal(e)
	}
	prior[0] = 1
	prior[1] = 0
	_ = transition.Set(0, 0, 0)
	_ = emission.Set(0, 0, 0)
	after, e := m.Filter(context.Background(), []uint32{0, 1}, hmm.FilterOptions{})
	if e != nil {
		t.Fatal(e)
	}
	if math.Float64bits(before.LogLikelihood) != math.Float64bits(after.LogLikelihood) {
		t.Fatal("model aliases input")
	}
	before.Last.LogPosterior[0] = 100
	fresh, e := m.Filter(context.Background(), []uint32{0, 1}, hmm.FilterOptions{})
	if e != nil || fresh.Last.LogPosterior[0] == 100 {
		t.Fatal("returned state aliases model")
	}
}
func TestConstructors_ValidationErrors(t *testing.T) {
	goodTransition := dense(t, [][]float64{{1}})
	cases := []struct {
		name string
		spec hmm.CategoricalSpec
		want error
	}{
		{"nil", hmm.CategoricalSpec{}, hmm.ErrNilInput},
		{"empty prior", hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{}, Transition: goodTransition}, Emission: goodTransition}, hmm.ErrShape},
		{"wrong transition shape", hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{.5, .5}, Transition: goodTransition}, Emission: goodTransition}, hmm.ErrShape},
		{"negative", hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{1}, Transition: goodTransition}, Emission: dense(t, [][]float64{{-.1, 1.1}})}, hmm.ErrProbability},
		{"bad row sum", hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{1}, Transition: goodTransition}, Emission: dense(t, [][]float64{{.1, .1}})}, hmm.ErrProbability},
		{"nonfinite", hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{math.NaN()}, Transition: goodTransition}, Emission: goodTransition}, hmm.ErrNonFinite},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			m, e := hmm.NewCategorical(tt.spec)
			if m != nil || !errors.Is(e, tt.want) {
				t.Fatalf("got model=%v error=%v wanted=%v", m, e, tt.want)
			}
		})
	}
	_, e := hmm.NewGaussian(hmm.GaussianSpec{Common: hmm.Common{Prior: []float64{1}, Transition: goodTransition}, Means: dense(t, [][]float64{{0}}), Variances: dense(t, [][]float64{{0}})})
	if !errors.Is(e, hmm.ErrProbability) {
		t.Fatal(e)
	}
}
func TestInvalidContextAndState_Atomicity(t *testing.T) {
	m := categorical(t)
	if r, e := m.Filter(nil, []uint32{0}, hmm.FilterOptions{}); r != nil || !errors.Is(e, hmm.ErrInvalidOption) {
		t.Fatal(e)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if r, e := m.Filter(canceled, []uint32{0}, hmm.FilterOptions{}); r != nil || !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	state, e := m.FilterStep(context.Background(), nil, 0)
	if e != nil {
		t.Fatal(e)
	}
	copyState := *state
	copyState.LogPosterior = append([]float64(nil), state.LogPosterior...)
	for _, mutation := range []func(*hmm.FilterState){
		func(s *hmm.FilterState) { s.Steps = 0 }, func(s *hmm.FilterState) { s.ModelID = "unknown" }, func(s *hmm.FilterState) { s.LogPosterior[0] = math.NaN() }, func(s *hmm.FilterState) { s.LogPosterior[0] = -1e6 }, func(s *hmm.FilterState) { s.LogLikelihood = math.Inf(1) },
	} {
		invalid := copyState
		invalid.LogPosterior = append([]float64(nil), copyState.LogPosterior...)
		mutation(&invalid)
		before := append([]float64(nil), invalid.LogPosterior...)
		if r, e := m.FilterStep(context.Background(), &invalid, 1); r != nil || !errors.Is(e, hmm.ErrStateVersion) {
			t.Fatalf("invalid state accepted: %+v %v", r, e)
		}
		for j, v := range before {
			if math.IsNaN(v) {
				if !math.IsNaN(invalid.LogPosterior[j]) {
					t.Fatal("mutated NaN")
				}
			} else if math.Float64bits(v) != math.Float64bits(invalid.LogPosterior[j]) {
				t.Fatal("input mutated")
			}
		}
	}
	overflow := copyState
	overflow.Steps = ^uint64(0)
	if r, e := m.FilterStep(context.Background(), &overflow, 0); r != nil || !errors.Is(e, hmm.ErrOverflow) {
		t.Fatal(e)
	}
}
func TestGaussian_AnalyticDensityAndInputValidation(t *testing.T) {
	initial := hmm.GaussianSpec{Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})}, Means: dense(t, [][]float64{{0}}), Variances: dense(t, [][]float64{{1}})}
	m, e := hmm.NewGaussian(initial)
	if e != nil {
		t.Fatal(e)
	}
	state, e := m.FilterStep(context.Background(), nil, []float64{0})
	if e != nil {
		t.Fatal(e)
	}
	approx(t, state.LogLikelihood, -.5*math.Log(2*math.Pi))
	if r, e := m.FilterStep(context.Background(), nil, []float64{math.Inf(1)}); r != nil || !errors.Is(e, hmm.ErrNonFinite) {
		t.Fatal(e)
	}
	if r, e := m.FilterStep(context.Background(), nil, []float64{}); r != nil || !errors.Is(e, hmm.ErrShape) {
		t.Fatal(e)
	}
	if r, e := m.Filter(context.Background(), dense(t, [][]float64{{0}, {1}}), hmm.FilterOptions{}); r == nil || e != nil {
		t.Fatal(e)
	}
	if r, e := m.Smooth(context.Background(), dense(t, [][]float64{{0}, {1}})); r == nil || e != nil {
		t.Fatal(e)
	}
	if r, e := m.Decode(context.Background(), dense(t, [][]float64{{0}, {1}})); r == nil || e != nil {
		t.Fatal(e)
	}
	if r, e := hmm.FitGaussian(context.Background(), []matrix.Matrix{dense(t, [][]float64{{0}})}, initial, hmm.FitOptions{VarianceFloor: 2}); r != nil || !errors.Is(e, hmm.ErrInvalidOption) {
		t.Fatal(e)
	}
}
func TestFitting_ValidationAndContext(t *testing.T) {
	initial := hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})}, Emission: dense(t, [][]float64{{.4, .6}})}
	for _, option := range []hmm.FitOptions{{Pseudocount: -1}, {MaxIterations: -1}, {Tolerance: math.NaN()}, {VarianceFloor: math.Inf(1)}} {
		if r, e := hmm.FitCategorical(context.Background(), [][]uint32{{0}}, initial, option); r != nil || !errors.Is(e, hmm.ErrInvalidOption) {
			t.Fatal(e)
		}
	}
	if r, e := hmm.FitCategorical(context.Background(), nil, initial, hmm.FitOptions{}); r != nil || !errors.Is(e, hmm.ErrNilInput) {
		t.Fatal(e)
	}
	if r, e := hmm.FitCategorical(context.Background(), [][]uint32{}, initial, hmm.FitOptions{}); r != nil || !errors.Is(e, hmm.ErrEmptyInput) {
		t.Fatal(e)
	}
	if r, e := hmm.FitCategorical(context.Background(), [][]uint32{{2}}, initial, hmm.FitOptions{}); r != nil || !errors.Is(e, hmm.ErrShape) {
		t.Fatal(e)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if r, e := hmm.FitCategorical(canceled, [][]uint32{{0}}, initial, hmm.FitOptions{}); r != nil || !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
}

// TestGaussian_LargeStandardizedResidual verifies that the likelihood kernel
// scales the quadratic form before multiplication, rather than spuriously
// rejecting a representable log density when z*z alone overflows.
func TestGaussian_LargeStandardizedResidual(t *testing.T) {
	model, err := hmm.NewGaussian(hmm.GaussianSpec{
		Common:    hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})},
		Means:     dense(t, [][]float64{{0}}),
		Variances: dense(t, [][]float64{{1}}),
	})
	if err != nil {
		t.Fatal(err)
	}
	observation := 1.5e154
	state, err := model.FilterStep(context.Background(), nil, []float64{observation})
	if err != nil {
		t.Fatal(err)
	}
	want := -0.5 * observation * observation
	if !finite(state.LogLikelihood) || math.Abs(state.LogLikelihood/want-1) > 1e-14 {
		t.Fatalf("log likelihood: got %.17g, want %.17g", state.LogLikelihood, want)
	}
}

func TestFilter_StochasticPrefixLikelihood(t *testing.T) {
	m := categorical(t)
	a, e := m.Filter(context.Background(), []uint32{0, 0}, hmm.FilterOptions{ReturnHistory: true})
	if e != nil {
		t.Fatal(e)
	}
	approx(t, a.LogLikelihood, math.Log(.41))
	v, e := a.Posterior.At(1, 0)
	if e != nil {
		t.Fatal(e)
	}
	approx(t, v, 81.0/82)
	// The detached state stores normalized log probabilities, not raw probabilities.
	z := hmm.LogSumExpSlice(a.Last.LogPosterior)
	approx(t, z, 0)
}
func TestFilter_ImpossibleAtSecondStep(t *testing.T) {
	transition := dense(t, [][]float64{{1}})
	emission := dense(t, [][]float64{{1, 0}})
	m, e := hmm.NewCategorical(hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{1}, Transition: transition}, Emission: emission})
	if e != nil {
		t.Fatal(e)
	}
	if r, e := m.Filter(context.Background(), []uint32{0, 1}, hmm.FilterOptions{ReturnHistory: true}); r != nil || !errors.Is(e, hmm.ErrImpossibleSequence) {
		t.Fatalf("partial result released: %v %v", r, e)
	}
	if r, e := m.Smooth(context.Background(), []uint32{0, 1}); r != nil || !errors.Is(e, hmm.ErrImpossibleSequence) {
		t.Fatal(e)
	}
	if r, e := m.Decode(context.Background(), []uint32{0, 1}); r != nil || !errors.Is(e, hmm.ErrImpossibleSequence) {
		t.Fatal(e)
	}
}
func TestSmoothing_ReallyUsesFuture(t *testing.T) {
	m := categorical(t)
	f, e := m.Filter(context.Background(), []uint32{0, 1, 1}, hmm.FilterOptions{ReturnHistory: true})
	if e != nil {
		t.Fatal(e)
	}
	s, e := m.Smooth(context.Background(), []uint32{0, 1, 1})
	if e != nil {
		t.Fatal(e)
	}
	a, _ := f.Posterior.At(0, 0)
	b, _ := s.Posterior.At(0, 0)
	if math.Abs(a-b) < .5 {
		t.Fatalf("smoothing unexpectedly causal: %.3f %.3f", a, b)
	}
}
