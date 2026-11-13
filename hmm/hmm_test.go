// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test validates causal inference, hidden-path probabilities, and
// immutable public result contracts using only the exported HMM interface.
package hmm_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

func TestCategorical_ObservationSymbolsChangeLikelihood(t *testing.T) {
	m := categorical(t)
	a, e := m.Filter(context.Background(), []uint32{0}, hmm.FilterOptions{ReturnHistory: true})
	if e != nil {
		t.Fatal(e)
	}
	b, e := m.Filter(context.Background(), []uint32{1}, hmm.FilterOptions{ReturnHistory: true})
	if e != nil {
		t.Fatal(e)
	}
	v, _ := a.Posterior.At(0, 0)
	approx(t, v, .9)
	v, _ = b.Posterior.At(0, 0)
	approx(t, v, .1)
	approx(t, a.LogLikelihood, math.Log(.5))
}
func TestFilter_PrefixInvariant(t *testing.T) {
	m := categorical(t)
	a, e := m.Filter(context.Background(), []uint32{0, 0}, hmm.FilterOptions{ReturnHistory: true})
	if e != nil {
		t.Fatal(e)
	}
	b, e := m.Filter(context.Background(), []uint32{0, 0, 1, 1, 0}, hmm.FilterOptions{ReturnHistory: true})
	if e != nil {
		t.Fatal(e)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			x, _ := a.Posterior.At(i, j)
			y, _ := b.Posterior.At(i, j)
			if math.Float64bits(x) != math.Float64bits(y) {
				t.Fatalf("causal violation %d %d", i, j)
			}
		}
	}
}
func TestFilterStep_RestoreMatchesBatch(t *testing.T) {
	m := categorical(t)
	x := []uint32{0, 1, 1, 0, 0, 1}
	batch, e := m.Filter(context.Background(), x, hmm.FilterOptions{})
	if e != nil {
		t.Fatal(e)
	}
	var state *hmm.FilterState
	for _, v := range x {
		state, e = m.FilterStep(context.Background(), state, v)
		if e != nil {
			t.Fatal(e)
		}
	}
	if state == nil {
		t.Fatal("FilterStep returned nil state without an error")
	}
	if math.Float64bits(state.LogLikelihood) != math.Float64bits(batch.LogLikelihood) {
		t.Fatal("likelihood bits differ")
	}
	for i, v := range state.LogPosterior {
		if math.Float64bits(v) != math.Float64bits(batch.Last.LogPosterior[i]) {
			t.Fatal("posterior bits differ")
		}
	}
	snapshot := append([]float64(nil), state.LogPosterior...)
	state.ModelID = "broken"
	_, e = m.FilterStep(context.Background(), state, 0)
	if !errors.Is(e, hmm.ErrStateVersion) {
		t.Fatal(e)
	}
	for i, v := range snapshot {
		if state.LogPosterior[i] != v {
			t.Fatal("mutated on error")
		}
	}
}
func TestFilter_LongSequenceNoUnderflowCollapse(t *testing.T) {
	m := categorical(t)
	x := make([]uint32, 100000)
	r, e := m.Filter(context.Background(), x, hmm.FilterOptions{})
	if e != nil {
		t.Fatal(e)
	}
	if !finite(r.LogLikelihood) || r.Posterior != nil {
		t.Fatal("numerical collapse or history allocation")
	}
}
func TestImpossibleSequence_NoNaN(t *testing.T) {
	m, e := hmm.NewCategorical(hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})}, Emission: dense(t, [][]float64{{1, 0}})})
	if e != nil {
		t.Fatal(e)
	}
	r, e := m.Filter(context.Background(), []uint32{1}, hmm.FilterOptions{})
	if r != nil || !errors.Is(e, hmm.ErrImpossibleSequence) {
		t.Fatalf("result=%v error=%v", r, e)
	}
	if !math.IsInf(hmm.LogSumExpSlice([]float64{math.Inf(-1), math.Inf(-1)}), -1) {
		t.Fatal("all impossible not -Inf")
	}
}
func TestDecode_TinyMaxPathOracle(t *testing.T) {
	m := categorical(t)
	r, e := m.Decode(context.Background(), []uint32{0, 0, 1})
	if e != nil {
		t.Fatal(e)
	}
	if len(r.States) != 3 || r.States[0] != 0 || r.States[1] != 0 || r.States[2] != 0 {
		t.Fatal(r.States)
	}
}
func TestGaussian_VarianceFloorUnits(t *testing.T) {
	initial := hmm.GaussianSpec{Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})}, Means: dense(t, [][]float64{{3}}), Variances: dense(t, [][]float64{{1}})}
	r, e := hmm.FitGaussian(context.Background(), []matrix.Matrix{dense(t, [][]float64{{3}, {3}, {3}})}, initial, hmm.FitOptions{VarianceFloor: .125})
	if e != nil && !errors.Is(e, hmm.ErrNotConverged) {
		t.Fatal(e)
	}
	step, e := r.Model.FilterStep(context.Background(), nil, []float64{3})
	if e != nil || !finite(step.LogLikelihood) {
		t.Fatalf("%v %v", step, e)
	}
}
func TestEM_PenalizedObjectiveMonotonic(t *testing.T) {
	initial := hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{.5, .5}, Transition: dense(t, [][]float64{{.8, .2}, {.2, .8}})}, Emission: dense(t, [][]float64{{.8, .2}, {.2, .8}})}
	r, e := hmm.FitCategorical(context.Background(), [][]uint32{{0, 0, 1, 1}, {1, 0, 1, 0}}, initial, hmm.FitOptions{Pseudocount: .1, MaxIterations: 8})
	if e != nil && !errors.Is(e, hmm.ErrNotConverged) {
		t.Fatal(e)
	}
	for i := 1; i < len(r.Diagnostics.Objective); i++ {
		if r.Diagnostics.Objective[i] < r.Diagnostics.Objective[i-1] {
			t.Fatalf("decrease: %v", r.Diagnostics.Objective)
		}
	}
}
func TestFit_MultipleSequencesDoNotJoin(t *testing.T) {
	initial := hmm.CategoricalSpec{
		Common:   hmm.Common{Prior: []float64{.5, .5}, Transition: dense(t, [][]float64{{.9, .1}, {.3, .7}})},
		Emission: dense(t, [][]float64{{1, 0}, {0, 1}}),
	}
	// Each length-one sequence has no within-sequence transition.
	// Any synthesized (0 -> 1) boundary would increase A[0,1].
	r, err := hmm.FitCategorical(context.Background(), [][]uint32{{0}, {1}}, initial, hmm.FitOptions{MaxIterations: 4})
	if err != nil && !errors.Is(err, hmm.ErrNotConverged) {
		t.Fatal(err)
	}
	if r == nil {
		t.Fatal("no diagnostic model")
	}
	result, err := r.Model.Filter(context.Background(), []uint32{0, 1}, hmm.FilterOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// Balanced initial gamma and deterministic emissions are unchanged.
	approx(t, math.Exp(result.LogLikelihood), .5*.1)
}

// TestResult_IsNilInterface verifies every HMM result against the actual
// core.Nilable interface, including typed nils stored behind interfaces.
// A present zero-valued result must never be confused with a nil pointer:
// IsNil is a presence predicate, not a validation or success predicate.
func TestResult_IsNilInterface(t *testing.T) {
	var _ core.Nilable = (*hmm.FilterResult)(nil)
	var _ core.Nilable = (*hmm.SmoothResult)(nil)
	var _ core.Nilable = (*hmm.DecodeResult)(nil)
	var _ core.Nilable = (*hmm.CategoricalFitResult)(nil)
	var _ core.Nilable = (*hmm.GaussianFitResult)(nil)

	cases := []struct {
		name    string
		absent  core.Nilable
		present core.Nilable
	}{
		{"causal filter", (*hmm.FilterResult)(nil), &hmm.FilterResult{}},
		{"retrospective smoother", (*hmm.SmoothResult)(nil), &hmm.SmoothResult{}},
		{"joint Viterbi path", (*hmm.DecodeResult)(nil), &hmm.DecodeResult{}},
		{"categorical fit", (*hmm.CategoricalFitResult)(nil), &hmm.CategoricalFitResult{}},
		{"Gaussian fit", (*hmm.GaussianFitResult)(nil), &hmm.GaussianFitResult{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.absent.IsNil() {
				t.Fatal("typed-nil result must satisfy core.Nilable.IsNil")
			}
			if tc.present.IsNil() {
				t.Fatal("a non-nil result pointer is present even when its fields are empty")
			}
		})
	}
}
func TestModel_ConcurrentInferenceRace(t *testing.T) {
	m := categorical(t)
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, e := m.Filter(context.Background(), []uint32{0, 1, 0}, hmm.FilterOptions{}); e != nil {
				t.Error(e)
			}
			if _, e := m.Smooth(context.Background(), []uint32{0, 1, 0}); e != nil {
				t.Error(e)
			}
			if _, e := m.Decode(context.Background(), []uint32{0, 1, 0}); e != nil {
				t.Error(e)
			}
		}()
	}
	wg.Wait()
}
