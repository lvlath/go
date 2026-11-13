// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test verifies model ownership, public wrappers, and numerical edge cases.
// Tests use only exported functions and interfaces, without privileged package access.
package hmm_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

// TestFacade_CanonicalOptions ensures convenience methods delegate to the canonical kernel.
func TestFacade_CanonicalOptions(t *testing.T) {
	m := categorical(t)
	symbols := []uint32{0, 1, 1, 0, 1}
	defaultResult, err := m.FilterDefault(context.Background(), symbols)
	if err != nil {
		t.Fatal(err)
	}
	historyResult, err := m.FilterWithHistory(context.Background(), symbols)
	if err != nil {
		t.Fatal(err)
	}
	if defaultResult.Posterior != nil || historyResult.Posterior == nil {
		t.Fatal("history policy changed")
	}
	if math.Float64bits(defaultResult.LogLikelihood) != math.Float64bits(historyResult.LogLikelihood) {
		t.Fatal("evidence changed")
	}
	for i := range defaultResult.Last.LogPosterior {
		if math.Float64bits(defaultResult.Last.LogPosterior[i]) != math.Float64bits(historyResult.Last.LogPosterior[i]) {
			t.Fatal("posterior changed")
		}
	}
	if defaultResult.Last.ModelID != m.ModelID() || m.ModelID() == "" {
		t.Fatal("model ID not exposed consistently")
	}
	var absent *hmm.CategoricalModel
	if absent.ModelID() != "" {
		t.Fatal("nil model identity")
	}
	if _, err = absent.FilterDefault(context.Background(), symbols); !errors.Is(err, hmm.ErrNilInput) {
		t.Fatal(err)
	}
}

// TestFacade_GaussianHistoryAndID compares model identity and both Gaussian history policies.
func TestFacade_GaussianHistoryAndID(t *testing.T) {
	m, err := hmm.NewGaussian(hmm.GaussianSpec{
		Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})},
		Means:  dense(t, [][]float64{{0}}), Variances: dense(t, [][]float64{{1}}),
	})
	if err != nil {
		t.Fatal(err)
	}
	observed := dense(t, [][]float64{{0}, {1}, {2}})
	defaultResult, err := m.FilterDefault(context.Background(), observed)
	if err != nil {
		t.Fatal(err)
	}
	historyResult, err := m.FilterWithHistory(context.Background(), observed)
	if err != nil {
		t.Fatal(err)
	}
	if defaultResult.Posterior != nil || historyResult.Posterior == nil || historyResult.Posterior.Rows() != 3 {
		t.Fatal("history policy")
	}
	if math.Float64bits(defaultResult.LogLikelihood) != math.Float64bits(historyResult.LogLikelihood) {
		t.Fatal("Gaussian likelihood changed")
	}
	if defaultResult.Last.ModelID != m.ModelID() {
		t.Fatal("Gaussian model identity changed")
	}
	var absent *hmm.GaussianModel
	if absent.ModelID() != "" {
		t.Fatal("nil identity")
	}
}

// TestGaussian_ProjectedVarianceFloor validates the actual fitted density rather than
// inferring success merely from a finite result at the mean.
func TestGaussian_ProjectedVarianceFloor(t *testing.T) {
	initial := hmm.GaussianSpec{
		Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})},
		Means:  dense(t, [][]float64{{3}}), Variances: dense(t, [][]float64{{1}}),
	}
	fitted, err := hmm.FitGaussian(context.Background(), []matrix.Matrix{dense(t, [][]float64{{3}, {3}, {3}})}, initial, hmm.FitOptions{VarianceFloor: .125})
	if err != nil && !errors.Is(err, hmm.ErrNotConverged) {
		t.Fatal(err)
	}
	if fitted == nil || fitted.Model == nil {
		t.Fatal("missing accepted model")
	}
	atMean, err := fitted.Model.FilterStep(context.Background(), nil, []float64{3})
	if err != nil {
		t.Fatal(err)
	}
	expected := -.5 * math.Log(2*math.Pi*.125)
	approx(t, atMean.LogLikelihood, expected)
}

// TestConstructor_DetachedParameters verifies all stored inputs are deep-copied
// and output history mutation cannot affect subsequent inference.
func TestConstructor_DetachedParameters(t *testing.T) {
	prior := []float64{.5, .5}
	transition := dense(t, [][]float64{{.95, .05}, {.1, .9}})
	emission := dense(t, [][]float64{{.8, .2}, {.3, .7}})
	model, err := hmm.NewCategorical(hmm.CategoricalSpec{
		Common: hmm.Common{Prior: prior, Transition: transition}, Emission: emission,
	})
	if err != nil {
		t.Fatal(err)
	}
	reference, err := model.FilterWithHistory(context.Background(), []uint32{0, 1, 0})
	if err != nil {
		t.Fatal(err)
	}
	prior[0] = 0
	if err = transition.Set(0, 0, 0); err != nil {
		t.Fatal(err)
	}
	if err = emission.Set(0, 0, 0); err != nil {
		t.Fatal(err)
	}
	again, err := model.FilterWithHistory(context.Background(), []uint32{0, 1, 0})
	if err != nil {
		t.Fatal(err)
	}
	if math.Float64bits(again.LogLikelihood) != math.Float64bits(reference.LogLikelihood) {
		t.Fatal("model aliases caller input")
	}
	if err = reference.Posterior.Set(0, 0, 0); err != nil {
		t.Fatal(err)
	}
	reference.Last.LogPosterior[0] = 999
	third, err := model.FilterDefault(context.Background(), []uint32{0, 1, 0})
	if err != nil {
		t.Fatal(err)
	}
	if math.Float64bits(third.LogLikelihood) != math.Float64bits(again.LogLikelihood) {
		t.Fatal("model aliases returned data")
	}
}

// TestLongSession_SmoothingAvoidsQuadraticTransitionHistory exercises long,
// moderately wide smoothing without requiring a T*K*K xi allocation.
func TestLongSession_SmoothingAvoidsQuadraticTransitionHistory(t *testing.T) {
	const states = 8
	prior := make([]float64, states)
	transition := make([][]float64, states)
	emission := make([][]float64, states)
	for i := 0; i < states; i++ {
		prior[i] = 1.0 / states
		transition[i] = make([]float64, states)
		emission[i] = []float64{.4, .6}
		for j := 0; j < states; j++ {
			transition[i][j] = 1.0 / states
		}
	}
	m, err := hmm.NewCategorical(hmm.CategoricalSpec{
		Common: hmm.Common{Prior: prior, Transition: dense(t, transition)}, Emission: dense(t, emission),
	})
	if err != nil {
		t.Fatal(err)
	}
	observations := make([]uint32, 2000)
	for i := range observations {
		observations[i] = uint32(i % 2)
	}
	smooth, err := m.Smooth(context.Background(), observations)
	if err != nil {
		t.Fatal(err)
	}
	if smooth.Posterior.Rows() != len(observations) || smooth.Posterior.Cols() != states {
		t.Fatal("posterior shape")
	}
	for i := 0; i < len(observations); i += 199 {
		var sum float64
		for j := 0; j < states; j++ {
			v, atErr := smooth.Posterior.At(i, j)
			if atErr != nil {
				t.Fatal(atErr)
			}
			sum += v
		}
		approx(t, sum, 1)
	}
}

// TestLogSumExp_ExtendedDomain verifies that valid infinite log-domain
// sentinels do not trigger an avoidable +Inf-+Inf indeterminate expression.
func TestLogSumExp_ExtendedDomain(t *testing.T) {
	negativeInf := math.Inf(-1)
	positiveInf := math.Inf(1)
	if !math.IsInf(hmm.LogSumExp(negativeInf, negativeInf), -1) {
		t.Fatal("impossible pair")
	}
	if !math.IsInf(hmm.LogSumExp(positiveInf, positiveInf), 1) {
		t.Fatal("positive infinity pair")
	}
	if !math.IsInf(hmm.LogSumExp(negativeInf, positiveInf), 1) {
		t.Fatal("positive infinity must dominate")
	}
	if !math.IsInf(hmm.LogSumExpSlice([]float64{positiveInf, 0}), 1) {
		t.Fatal("vector positive infinity")
	}
}
