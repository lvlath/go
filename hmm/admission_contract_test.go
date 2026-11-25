// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test verifies HMM model admission and error propagation.
// Invalid logical shapes and source failures must be rejected without mutation or publication.
package hmm_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

// TestNewCategorical_EmptyAlphabetAndFitAdmission rejects an empty emission domain.
// The model must have at least one symbol even when the Markov chain itself is valid.
//
// Implementation:
//   - Stage 1: create a valid one-state chain and a logical 1-by-0 emission matrix.
//   - Stage 2: call the constructor, then the public fitting facade with the same specification.
//   - Stage 3: require ErrShape, nil results, and no numerical emission reads or writes.
//
// Behavior highlights:
//   - This is an empty alphabet, not an empty observation sequence.
//
// Inputs:
//   - t: the active test; all model parameters are deterministic fixtures.
//
// Returns:
//   - None; the test asserts the public admission contract.
//
// Errors:
//   - ErrShape before emission cell access or training input validation.
//
// Determinism:
//   - Fixed one-state, zero-alphabet configuration.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The matrix has a logical shape but does not allocate a backing array.
//
// AI-Hints:
//   - Do not confuse A=0 with a valid A>=1 model whose emissions contain structural zeros.
func TestNewCategorical_EmptyAlphabetAndFitAdmission(t *testing.T) {
	ctx := context.Background()
	emission := &admissionProbeMatrix{rows: 1, cols: 0}
	spec := hmm.CategoricalSpec{
		Common:   hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})},
		Emission: emission,
	}

	model, err := hmm.NewCategorical(spec)
	if model != nil || !errors.Is(err, hmm.ErrShape) {
		t.Fatalf("constructor: model=%v err=%v, want nil/ErrShape", model, err)
	}
	// Invalid initialization precedes inspection of the absent training collection.
	fit, err := hmm.FitCategorical(ctx, nil, spec, hmm.FitOptions{})
	requireHMMFailure(t, fit, err, hmm.ErrShape)
	requireAdmissionAccess(t, emission, 0)
}

// TestNewGaussian_ParameterAdmissionStages isolates each constructor rejection stage.
// Valid preceding fields ensure each failure comes from the intended public input boundary.
//
// Implementation:
//   - Stage 1: prepare a valid one-state Gaussian specification for each subtest.
//   - Stage 2: replace exactly one field with a malformed or inaccessible input.
//   - Stage 3: verify NewGaussian and FitGaussian preserve error identity and reject publication.
//
// Behavior highlights:
//   - Typed nil, stochastic-domain errors, shape errors, source errors, and nonfinite values are distinct.
//   - Unreadable matrices are not copied or modified after a failure.
//
// Inputs:
//   - t: active test; fixture means and variances use the same physical coordinate.
//
// Returns:
//   - None; every subtest requires the expected public error.
//
// Errors:
//   - ErrNilInput, ErrProbability, ErrShape, ErrNonFinite, or the original source error.
//
// Determinism:
//   - Each subtest constructs independent probe state and a stable model specification.
//
// Complexity:
//   - O(number of cases) time and O(1) additional fixture space per case.
//
// Notes:
//   - Fit is invoked with no sequences to verify invalid initialization takes precedence.
//
// AI-Hints:
//   - Test an At failure with a correctly shaped matrix; a bad shape would mask that path.
//   - Use a read-only fixture to inject NaN because ordinary Dense writes reject it.
func TestNewGaussian_ParameterAdmissionStages(t *testing.T) {
	var typedNil *matrix.Dense
	cases := []struct {
		name   string
		change func(*hmm.GaussianSpec) *admissionProbeMatrix
		want   error
		reads  int // Per call, including the failed At attempt.
	}{
		{
			name: "typed_nil_mean", want: hmm.ErrNilInput,
			change: func(spec *hmm.GaussianSpec) *admissionProbeMatrix {
				spec.Means = typedNil
				return nil
			},
		},
		{
			name: "typed_nil_variance", want: hmm.ErrNilInput,
			change: func(spec *hmm.GaussianSpec) *admissionProbeMatrix {
				spec.Variances = typedNil
				return nil
			},
		},
		{
			name: "invalid_common_prior", want: hmm.ErrProbability,
			change: func(spec *hmm.GaussianSpec) *admissionProbeMatrix {
				// A correctly shaped Gaussian model still requires a normalized prior.
				spec.Common.Prior = []float64{.5}
				probe := &admissionProbeMatrix{rows: 1, cols: 1}
				spec.Means = probe
				return probe
			},
		},
		{
			name: "empty_feature_domain", want: hmm.ErrShape,
			change: func(spec *hmm.GaussianSpec) *admissionProbeMatrix {
				probe := &admissionProbeMatrix{rows: 1, cols: 0}
				spec.Means = probe
				return probe
			},
		},
		{
			name: "mean_state_count", want: hmm.ErrShape,
			change: func(spec *hmm.GaussianSpec) *admissionProbeMatrix {
				probe := &admissionProbeMatrix{rows: 2, cols: 1}
				spec.Means = probe
				return probe
			},
		},
		{
			name: "mean_source_failure", want: errAdmissionRead, reads: 1,
			change: func(spec *hmm.GaussianSpec) *admissionProbeMatrix {
				probe := &admissionProbeMatrix{rows: 1, cols: 1, readErr: errAdmissionRead}
				spec.Means = probe
				return probe
			},
		},
		{
			name: "nonfinite_mean", want: hmm.ErrNonFinite, reads: 1,
			change: func(spec *hmm.GaussianSpec) *admissionProbeMatrix {
				probe := &admissionProbeMatrix{rows: 1, cols: 1, value: math.NaN()}
				spec.Means = probe
				return probe
			},
		},
		{
			name: "variance_feature_count", want: hmm.ErrShape,
			change: func(spec *hmm.GaussianSpec) *admissionProbeMatrix {
				probe := &admissionProbeMatrix{rows: 1, cols: 2, value: 1}
				spec.Variances = probe
				return probe
			},
		},
		{
			name: "variance_source_failure", want: errAdmissionRead, reads: 1,
			change: func(spec *hmm.GaussianSpec) *admissionProbeMatrix {
				probe := &admissionProbeMatrix{rows: 1, cols: 1, readErr: errAdmissionRead}
				spec.Variances = probe
				return probe
			},
		},
		{
			name: "nonfinite_variance", want: hmm.ErrNonFinite, reads: 1,
			change: func(spec *hmm.GaussianSpec) *admissionProbeMatrix {
				probe := &admissionProbeMatrix{rows: 1, cols: 1, value: math.Inf(1)}
				spec.Variances = probe
				return probe
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spec := singleStateGaussianFixture(t, []float64{0}, []float64{1})
			probe := tc.change(&spec)
			model, err := hmm.NewGaussian(spec)
			if model != nil || !errors.Is(err, tc.want) {
				t.Fatalf("constructor: model=%v err=%v, want nil/%v", model, err, tc.want)
			}
			if probe != nil {
				requireAdmissionAccess(t, probe, tc.reads)
			}
			fit, err := hmm.FitGaussian(context.Background(), nil, spec, hmm.FitOptions{})
			requireHMMFailure(t, fit, err, tc.want)
			if probe != nil {
				// The second call must stop at the identical validation boundary.
				requireAdmissionAccess(t, probe, 2*tc.reads)
			}
		})
	}
}

// TestGaussianSmooth_ContextAndMatrixAdmission verifies rejection before data access.
// It covers invalid context precedence, negative logical rows, and dependency errors.
//
// Implementation:
//   - Stage 1: construct a valid one-state Gaussian model and bounded probe matrices.
//   - Stage 2: exercise the public Smooth method with each independent fault.
//   - Stage 3: require the correct sentinel, nil result, and exact access count.
//
// Behavior highlights:
//   - The test does not allocate a virtual matrix's claimed extent.
//
// Inputs:
//   - t: active test; all observations are caller-owned.
//
// Returns:
//   - None; each fault is classified independently.
//
// Errors:
//   - ErrInvalidOption, context.Canceled, ErrOverflow, or errAdmissionRead.
//
// Determinism:
//   - Cancellation is complete before the call; probe shapes never change during reads.
//
// Complexity:
//   - Time O(1), Space O(1) per case.
//
// Notes:
//   - Negative rows exercise checkedAllocation without asking the runtime for a huge allocation.
//
// AI-Hints:
//   - Never use an enormous but representable allocation as a unit-test fault injection mechanism.
func TestGaussianSmooth_ContextAndMatrixAdmission(t *testing.T) {
	model, err := hmm.NewGaussian(singleStateGaussianFixture(t, []float64{0}, []float64{1}))
	if err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	cases := []struct {
		name  string
		ctx   context.Context
		probe *admissionProbeMatrix
		want  error
		reads int
	}{
		{"nil_context", nil, &admissionProbeMatrix{rows: 1, cols: 1, readErr: errAdmissionRead}, hmm.ErrInvalidOption, 0},
		{"canceled_context", canceled, &admissionProbeMatrix{rows: 1, cols: 1, readErr: errAdmissionRead}, context.Canceled, 0},
		{"negative_rows", context.Background(), &admissionProbeMatrix{rows: -1, cols: 1}, hmm.ErrOverflow, 0},
		{"source_failure", context.Background(), &admissionProbeMatrix{rows: 1, cols: 1, readErr: errAdmissionRead}, errAdmissionRead, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := model.Smooth(tc.ctx, tc.probe)
			requireHMMFailure(t, result, err, tc.want)
			requireAdmissionAccess(t, tc.probe, tc.reads)
		})
	}
}

// TestModelConstruction_GenericMatrixMatchesDense checks the public fallback Matrix contract.
// The model accepts a valid Matrix implementation that does not expose the optional Nilable method.
//
// Implementation:
//   - Stage 1: create a Dense-backed Gaussian specification with stable finite values.
//   - Stage 2: construct an equivalent model from Matrix-only wrappers and smooth identical observations.
//   - Stage 3: compare canonical identity, log evidence, and every published posterior cell.
//
// Behavior highlights:
//   - The wrapper does not introduce reflection or access private Dense storage.
//
// Inputs:
//   - t: active test; two Gaussian states, one feature, three finite observations.
//
// Returns:
//   - None; both public inference paths must publish equivalent artifacts.
//
// Errors:
//   - None on either valid representation.
//
// Determinism:
//   - Identical row-major values and fixed model traversal order.
//
// Complexity:
//   - O(T*K*K+T*K*D) inference time, O(T*K) posterior workspace.
//
// Notes:
//   - This checks interface compatibility, not a private backing-slice implementation detail.
//
// AI-Hints:
//   - Do not require a concrete Dense or Nilable type for all valid non-nil Matrix inputs.
func TestModelConstruction_GenericMatrixMatchesDense(t *testing.T) {
	ctx := context.Background()
	spec := hmm.GaussianSpec{
		Common: hmm.Common{Prior: []float64{.75, .25}, Transition: dense(t, [][]float64{{.8, .2}, {.1, .9}})},
		Means:  dense(t, [][]float64{{-1}, {2}}), Variances: dense(t, [][]float64{{1}, {4}}),
	}
	reference, err := hmm.NewGaussian(spec)
	if err != nil {
		t.Fatal(err)
	}
	wrapped := spec
	wrapped.Common.Transition = plainMatrixFixture{Matrix: spec.Common.Transition}
	wrapped.Means = plainMatrixFixture{Matrix: spec.Means}
	wrapped.Variances = plainMatrixFixture{Matrix: spec.Variances}
	model, err := hmm.NewGaussian(wrapped)
	if err != nil || model == nil || model.ModelID() != reference.ModelID() {
		t.Fatalf("generic Matrix model=%v err=%v", model, err)
	}
	observations := dense(t, [][]float64{{-1}, {0}, {2}})
	want, err := reference.Smooth(ctx, observations)
	if err != nil || want == nil {
		t.Fatalf("Dense result=%v err=%v", want, err)
	}
	got, err := model.Smooth(ctx, plainMatrixFixture{Matrix: observations})
	if err != nil || got == nil || math.Float64bits(got.LogLikelihood) != math.Float64bits(want.LogLikelihood) {
		t.Fatalf("generic Matrix result=%v err=%v", got, err)
	}
	if got.Posterior.Rows() != observations.Rows() || got.Posterior.Cols() != 2 {
		t.Fatalf("unexpected posterior shape: %dx%d", got.Posterior.Rows(), got.Posterior.Cols())
	}
	for row := 0; row < got.Posterior.Rows(); row++ {
		for col := 0; col < got.Posterior.Cols(); col++ {
			x, xErr := got.Posterior.At(row, col)
			y, yErr := want.Posterior.At(row, col)
			if xErr != nil || yErr != nil || !finite(x) || math.Float64bits(x) != math.Float64bits(y) {
				t.Fatalf("posterior(%d,%d)=%g want=%g errors=(%v,%v)", row, col, x, y, xErr, yErr)
			}
		}
	}
}
