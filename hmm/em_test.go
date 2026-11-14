// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test verifies checked allocation, weighted central moments and
// regularized simplex probabilities without inspecting any private HMM state.
package hmm_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

// auditVirtualMatrix advertises huge dimensions without allocating large storage.
// Any At invocation is counted, allowing early allocation checks to be verified.
type auditVirtualMatrix struct {
	rows, cols int
	reads      int
}

func (v *auditVirtualMatrix) Rows() int   { return v.rows }
func (v *auditVirtualMatrix) Cols() int   { return v.cols }
func (v *auditVirtualMatrix) IsNil() bool { return v == nil }
func (v *auditVirtualMatrix) At(int, int) (float64, error) {
	v.reads++
	return 0, errors.New("unexpected At")
}
func (v *auditVirtualMatrix) Set(int, int, float64) error { return errors.New("read-only") }
func (v *auditVirtualMatrix) Clone() matrix.Matrix {
	return &auditVirtualMatrix{rows: v.rows, cols: v.cols}
}

// TestAuditCheckedAllocationBytes requires ErrOverflow before touching an
// oversized virtual input, not an unrecoverable makeslice panic.
func TestAuditCheckedAllocationBytes(t *testing.T) {
	const float64Bytes = 8
	huge := int(^uint(0)>>1)/float64Bytes + 1
	v := &auditVirtualMatrix{rows: 1, cols: huge}
	_, err := hmm.NewCategorical(hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})}, Emission: v})
	if !errors.Is(err, hmm.ErrOverflow) || v.reads != 0 {
		t.Fatalf("categorical err=%v reads=%d", err, v.reads)
	}
	model := auditGaussian(t, []float64{1}, [][]float64{{1}}, [][]float64{{0}}, [][]float64{{1}})
	for _, tc := range []struct {
		name string
		run  func(matrix.Matrix) error
	}{
		{"history", func(input matrix.Matrix) error {
			_, e := model.Filter(context.Background(), input, hmm.FilterOptions{ReturnHistory: true})
			return e
		}},
		{"smooth", func(input matrix.Matrix) error { _, e := model.Smooth(context.Background(), input); return e }},
		{"decode", func(input matrix.Matrix) error { _, e := model.Decode(context.Background(), input); return e }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			matrixInput := &auditVirtualMatrix{rows: huge, cols: 1}
			e := tc.run(matrixInput)
			if !errors.Is(e, hmm.ErrOverflow) || matrixInput.reads != 0 {
				t.Fatalf("err=%v reads=%d", e, matrixInput.reads)
			}
		})
	}
}

// TestAuditGaussianEMCancellation checks that an offset of 1e8 does not
// cancel a physical population variance of exactly one.
func TestAuditGaussianEMCancellation(t *testing.T) {
	initial := hmm.GaussianSpec{Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})}, Means: dense(t, [][]float64{{0}}), Variances: dense(t, [][]float64{{1e16}})}
	result, err := hmm.FitGaussian(context.Background(), []matrix.Matrix{dense(t, [][]float64{{1e8 - 1}, {1e8 + 1}})}, initial, hmm.FitOptions{MaxIterations: 1})
	if err != nil && !errors.Is(err, hmm.ErrNotConverged) {
		t.Fatal(err)
	}
	if result == nil || result.Model == nil || result.Diagnostics.Iterations != 1 {
		t.Fatalf("result=%v,err=%v", result, err)
	}
	state, e := result.Model.FilterStep(context.Background(), nil, []float64{1e8})
	if e != nil {
		t.Fatal(e)
	}
	want := -.5 * math.Log(2*math.Pi)
	if math.Abs(state.LogLikelihood-want) > 2e-12 {
		t.Fatalf("fitted log density=%g want=%g", state.LogLikelihood, want)
	}
	if result.Diagnostics.Objective[1] < result.Diagnostics.Objective[0] {
		t.Fatalf("nonmonotone=%v", result.Diagnostics.Objective)
	}
}

// TestAuditGaussianEMRepresentableVariance demands normalization before
// materializing squared residuals on a symmetric 1e154-scale dataset.
func TestAuditGaussianEMRepresentableVariance(t *testing.T) {
	initial := hmm.GaussianSpec{Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})}, Means: dense(t, [][]float64{{0}}), Variances: dense(t, [][]float64{{1e308}})}
	result, err := hmm.FitGaussian(context.Background(), []matrix.Matrix{dense(t, [][]float64{{-1e154}, {0}, {1e154}})}, initial, hmm.FitOptions{MaxIterations: 1})
	if err != nil && !errors.Is(err, hmm.ErrNotConverged) {
		t.Fatal(err)
	}
	if result == nil || result.Model == nil || result.Diagnostics.Iterations != 1 {
		t.Fatalf("result=%v err=%v", result, err)
	}
	state, e := result.Model.FilterStep(context.Background(), nil, []float64{0})
	if e != nil {
		t.Fatal(e)
	}
	// Compute an equivalent finite log-variance without raw 2e308 overflow.
	want := -.5 * (math.Log(2*math.Pi) + math.Log(1e308) + math.Log(2./3.))
	if math.Abs(state.LogLikelihood-want) > 2e-11 {
		t.Fatalf("fit density=%g want=%g", state.LogLikelihood, want)
	}
}

// TestAuditCategoricalEMLargePseudocount verifies a finite huge Dirichlet
// prior without creating an infinite simplex denominator.
func TestAuditCategoricalEMLargePseudocount(t *testing.T) {
	initial := hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})}, Emission: dense(t, [][]float64{{.5, .5}})}
	result, err := hmm.FitCategorical(context.Background(), [][]uint32{{0, 1}}, initial, hmm.FitOptions{Pseudocount: 1e308, MaxIterations: 1})
	if err != nil && !errors.Is(err, hmm.ErrNotConverged) {
		t.Fatal(err)
	}
	if result == nil || result.Model == nil {
		t.Fatalf("result=%v err=%v", result, err)
	}
	for _, symbol := range []uint32{0, 1} {
		filtered, e := result.Model.Filter(context.Background(), []uint32{symbol}, hmm.FilterOptions{})
		if e != nil {
			t.Fatal(e)
		}
		if math.Abs(math.Exp(filtered.LogLikelihood)-.5) > 1e-12 {
			t.Fatalf("symbol=%d likelihood=%g", symbol, filtered.LogLikelihood)
		}
	}
	for _, score := range result.Diagnostics.Objective {
		if math.IsInf(score, 0) || math.IsNaN(score) {
			t.Fatalf("nonfinite objective=%v", result.Diagnostics.Objective)
		}
	}
}

// TestAuditGaussianEMIndependentWeightedResponsibilities independently computes
// the first MAP-free M-step for three singleton machines with unequal Gaussian
// responsibilities. No transition may cross their independent boundaries.
func TestAuditGaussianEMIndependentWeightedResponsibilities(t *testing.T) {
	prior := []float64{.4, .6}
	observations := []float64{1e8 - 1, 1e8, 1e8 + 1}
	initial := hmm.GaussianSpec{
		Common:    hmm.Common{Prior: prior, Transition: dense(t, [][]float64{{1, 0}, {0, 1}})},
		Means:     dense(t, [][]float64{{1e8 - 2}, {1e8 + 2}}),
		Variances: dense(t, [][]float64{{100}, {100}}),
	}
	runs := make([]matrix.Matrix, len(observations))
	for i, value := range observations {
		runs[i] = dense(t, [][]float64{{value}})
	}
	fit, err := hmm.FitGaussian(context.Background(), runs, initial, hmm.FitOptions{MaxIterations: 1})
	if err != nil && !errors.Is(err, hmm.ErrNotConverged) {
		t.Fatal(err)
	}
	if fit == nil || fit.Model == nil || fit.Diagnostics.Iterations != 1 {
		t.Fatalf("fit=%v err=%v", fit, err)
	}

	// Analytic singleton responsibilities use only the initial prior and
	// Gaussian likelihood; they are not extracted from production E-step code.
	weights := make([][]float64, len(observations))
	occupancy := make([]float64, 2)
	expectedMeans := make([]float64, 2)
	for i, x := range observations {
		logits := [2]float64{
			math.Log(prior[0]) - .5*math.Pow(x-(1e8-2), 2)/100,
			math.Log(prior[1]) - .5*math.Pow(x-(1e8+2), 2)/100,
		}
		maxLog := math.Max(logits[0], logits[1])
		a, b := math.Exp(logits[0]-maxLog), math.Exp(logits[1]-maxLog)
		weights[i] = []float64{a / (a + b), b / (a + b)}
		for j, w := range weights[i] {
			occupancy[j] += w
			expectedMeans[j] += w * (x - 1e8)
		}
	}
	expectedVariances := make([]float64, 2)
	expectedPrior := make([]float64, 2)
	for state := 0; state < 2; state++ {
		expectedMeans[state] = 1e8 + expectedMeans[state]/occupancy[state]
		expectedPrior[state] = occupancy[state] / float64(len(observations))
	}
	for i, x := range observations {
		for j, w := range weights[i] {
			d := x - expectedMeans[j]
			expectedVariances[j] += w * d * d / occupancy[j]
		}
	}
	oracle := auditGaussian(t, expectedPrior, [][]float64{{1, 0}, {0, 1}},
		[][]float64{{expectedMeans[0]}, {expectedMeans[1]}},
		[][]float64{{expectedVariances[0]}, {expectedVariances[1]}})
	for _, x := range []float64{1e8 - 2, 1e8, 1e8 + 2} {
		got, e := fit.Model.FilterStep(context.Background(), nil, []float64{x})
		if e != nil {
			t.Fatal(e)
		}
		want, e := oracle.FilterStep(context.Background(), nil, []float64{x})
		if e != nil {
			t.Fatal(e)
		}
		if math.Abs(got.LogLikelihood-want.LogLikelihood) > 3e-12 {
			t.Fatalf("x=%g observed=%g analytic=%g", x, got.LogLikelihood, want.LogLikelihood)
		}
		for j, v := range got.LogPosterior {
			if math.Abs(v-want.LogPosterior[j]) > 3e-12 {
				t.Fatalf("x=%g state=%d got=%g want=%g", x, j, v, want.LogPosterior[j])
			}
		}
	}
}

// TestAuditCategoricalEMLargePseudocountAsymmetricCounts checks that large
// pseudocount dominates safely without rounding a live emission row to zero.
func TestAuditCategoricalEMLargePseudocountAsymmetricCounts(t *testing.T) {
	initial := hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})}, Emission: dense(t, [][]float64{{.8, .2}})}
	sequences := [][]uint32{{0, 0, 0, 1}}
	fit, err := hmm.FitCategorical(context.Background(), sequences, initial, hmm.FitOptions{Pseudocount: 1e308, MaxIterations: 1})
	if err != nil && !errors.Is(err, hmm.ErrNotConverged) {
		t.Fatal(err)
	}
	if fit == nil || fit.Model == nil {
		t.Fatalf("fit=%v err=%v", fit, err)
	}
	for _, symbol := range []uint32{0, 1} {
		posterior, e := fit.Model.Filter(context.Background(), []uint32{symbol}, hmm.FilterOptions{})
		if e != nil {
			t.Fatal(e)
		}
		if math.Abs(math.Exp(posterior.LogLikelihood)-.5) > 1e-12 {
			t.Fatalf("symbol=%d probability=%g", symbol, math.Exp(posterior.LogLikelihood))
		}
	}
}
