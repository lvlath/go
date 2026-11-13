// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test validates public error precedence and ownership without
// privileged access to hmm's private numerical kernels.
package hmm_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

// boundaryMatrix implements the public Matrix interface with deterministic
// injected numeric values/errors. It lets tests exercise the read boundary
// without bypassing the package through internal or production test helpers.
type boundaryMatrix struct {
	rows, cols int
	value      float64
	readErr    error
}

// Rows returns the advertised public matrix height.
func (m *boundaryMatrix) Rows() int { return m.rows }

// Cols returns the advertised public matrix width.
func (m *boundaryMatrix) Cols() int { return m.cols }

// At returns a controlled cell value or a controlled underlying matrix error.
func (m *boundaryMatrix) At(_, _ int) (float64, error) { return m.value, m.readErr }

// Set changes the testing matrix's scalar value, as required by Matrix.
func (m *boundaryMatrix) Set(_, _ int, v float64) error { m.value = v; return nil }

// Clone returns a detached shallow value containing only scalar test fields.
func (m *boundaryMatrix) Clone() matrix.Matrix {
	if m == nil {
		return (*boundaryMatrix)(nil)
	}
	copyOf := *m
	return &copyOf
}

// IsNil implements core.Nilable to demonstrate reflect-free typed-nil checks.
func (m *boundaryMatrix) IsNil() bool { return m == nil }

// TestInference_ErrorPrecedence verifies the public context/presence/shape
// contract across all categorical facades, not only the primary Filter call.
func TestInference_ErrorPrecedence(t *testing.T) {
	model := categorical(t)
	valid := []uint32{0, 1}
	var absent *hmm.CategoricalModel
	type check struct {
		name string
		run  func() error
		want error
	}
	cases := []check{
		{"filter context before nil model", func() error { _, err := absent.Filter(nil, valid, hmm.FilterOptions{}); return err }, hmm.ErrInvalidOption},
		{"filter nil model", func() error { _, err := absent.Filter(context.Background(), valid, hmm.FilterOptions{}); return err }, hmm.ErrNilInput},
		{"filter nil symbols", func() error { _, err := model.FilterDefault(context.Background(), nil); return err }, hmm.ErrNilInput},
		{"filter empty symbols", func() error { _, err := model.FilterDefault(context.Background(), []uint32{}); return err }, hmm.ErrEmptyInput},
		{"filter invalid symbol", func() error { _, err := model.FilterWithHistory(context.Background(), []uint32{0, 2}); return err }, hmm.ErrShape},
		{"smooth context before nil model", func() error { _, err := absent.Smooth(nil, valid); return err }, hmm.ErrInvalidOption},
		{"smooth nil model", func() error { _, err := absent.Smooth(context.Background(), valid); return err }, hmm.ErrNilInput},
		{"smooth nil symbols", func() error { _, err := model.Smooth(context.Background(), nil); return err }, hmm.ErrNilInput},
		{"smooth empty symbols", func() error { _, err := model.Smooth(context.Background(), []uint32{}); return err }, hmm.ErrEmptyInput},
		{"smooth invalid symbol", func() error { _, err := model.Smooth(context.Background(), []uint32{0, 2}); return err }, hmm.ErrShape},
		{"decode context before nil model", func() error { _, err := absent.Decode(nil, valid); return err }, hmm.ErrInvalidOption},
		{"decode nil model", func() error { _, err := absent.Decode(context.Background(), valid); return err }, hmm.ErrNilInput},
		{"decode nil symbols", func() error { _, err := model.Decode(context.Background(), nil); return err }, hmm.ErrNilInput},
		{"decode empty symbols", func() error { _, err := model.Decode(context.Background(), []uint32{}); return err }, hmm.ErrEmptyInput},
		{"decode invalid symbol", func() error { _, err := model.Decode(context.Background(), []uint32{2}); return err }, hmm.ErrShape},
		{"step nil model", func() error { _, err := absent.FilterStep(context.Background(), nil, 0); return err }, hmm.ErrNilInput},
		{"step invalid symbol", func() error { _, err := model.FilterStep(context.Background(), nil, 2); return err }, hmm.ErrShape},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); !errors.Is(err, tc.want) {
				t.Fatalf("got %v, want errors.Is(...,%v)", err, tc.want)
			}
		})
	}
}

// TestGaussianInference_BoundaryErrors exercises both nilable typed pointers
// and a foreign Matrix implementer that reports bad numeric or accessor data.
func TestGaussianInference_BoundaryErrors(t *testing.T) {
	model, err := hmm.NewGaussian(hmm.GaussianSpec{
		Common:    hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})},
		Means:     dense(t, [][]float64{{0}}),
		Variances: dense(t, [][]float64{{1}}),
	})
	if err != nil {
		t.Fatal(err)
	}
	var absent *hmm.GaussianModel
	var typedNil *boundaryMatrix
	var typedDenseNil *matrix.Dense
	forced := errors.New("deterministic matrix accessor failure")
	cases := []struct {
		name string
		call func() error
		want error
	}{
		{"filter nil context", func() error { _, err = model.Filter(nil, dense(t, [][]float64{{0}}), hmm.FilterOptions{}); return err }, hmm.ErrInvalidOption},
		{"filter nil receiver", func() error {
			_, err = absent.Filter(context.Background(), typedDenseNil, hmm.FilterOptions{})
			return err
		}, hmm.ErrNilInput},
		{"filter typed nil matrix", func() error { _, err = model.Filter(context.Background(), typedNil, hmm.FilterOptions{}); return err }, hmm.ErrNilInput},
		{"filter typed nil Dense", func() error {
			_, err = model.Filter(context.Background(), typedDenseNil, hmm.FilterOptions{})
			return err
		}, hmm.ErrNilInput},
		{"filter no rows", func() error {
			_, err = model.FilterDefault(context.Background(), &boundaryMatrix{rows: 0, cols: 1})
			return err
		}, hmm.ErrEmptyInput},
		{"filter wrong columns", func() error {
			_, err = model.FilterDefault(context.Background(), &boundaryMatrix{rows: 1, cols: 2})
			return err
		}, hmm.ErrShape},
		{"filter nonfinite At", func() error {
			_, err = model.FilterDefault(context.Background(), &boundaryMatrix{rows: 1, cols: 1, value: math.NaN()})
			return err
		}, hmm.ErrNonFinite},
		{"filter source error", func() error {
			_, err = model.FilterWithHistory(context.Background(), &boundaryMatrix{rows: 1, cols: 1, readErr: forced})
			return err
		}, forced},
		{"smooth nil receiver", func() error { _, err = absent.Smooth(context.Background(), dense(t, [][]float64{{0}})); return err }, hmm.ErrNilInput},
		{"smooth nil matrix", func() error { _, err = model.Smooth(context.Background(), typedDenseNil); return err }, hmm.ErrNilInput},
		{"smooth no rows", func() error {
			_, err = model.Smooth(context.Background(), &boundaryMatrix{rows: 0, cols: 1})
			return err
		}, hmm.ErrEmptyInput},
		{"smooth wrong columns", func() error {
			_, err = model.Smooth(context.Background(), &boundaryMatrix{rows: 1, cols: 2})
			return err
		}, hmm.ErrShape},
		{"smooth nonfinite", func() error {
			_, err = model.Smooth(context.Background(), &boundaryMatrix{rows: 1, cols: 1, value: math.Inf(-1)})
			return err
		}, hmm.ErrNonFinite},
		{"decode nil receiver", func() error { _, err = absent.Decode(context.Background(), dense(t, [][]float64{{0}})); return err }, hmm.ErrNilInput},
		{"decode nil matrix", func() error { _, err = model.Decode(context.Background(), typedDenseNil); return err }, hmm.ErrNilInput},
		{"decode no rows", func() error {
			_, err = model.Decode(context.Background(), &boundaryMatrix{rows: 0, cols: 1})
			return err
		}, hmm.ErrEmptyInput},
		{"decode wrong columns", func() error {
			_, err = model.Decode(context.Background(), &boundaryMatrix{rows: 1, cols: 2})
			return err
		}, hmm.ErrShape},
		{"decode nonfinite", func() error {
			_, err = model.Decode(context.Background(), &boundaryMatrix{rows: 1, cols: 1, value: math.Inf(1)})
			return err
		}, hmm.ErrNonFinite},
		{"step nil receiver", func() error { _, err = absent.FilterStep(context.Background(), nil, []float64{0}); return err }, hmm.ErrNilInput},
		{"step nil features", func() error { _, err = model.FilterStep(context.Background(), nil, nil); return err }, hmm.ErrNilInput},
		{"step shape", func() error { _, err = model.FilterStep(context.Background(), nil, []float64{}); return err }, hmm.ErrShape},
		{"step nonfinite", func() error { _, err = model.FilterStep(context.Background(), nil, []float64{math.NaN()}); return err }, hmm.ErrNonFinite},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err = tc.call(); !errors.Is(err, tc.want) {
				t.Fatalf("got %v, want %v", err, tc.want)
			}
		})
	}
}

// TestModelConstruction_MatrixAccessErrorPreserved verifies dependency failures
// are wrapped with their sentinel identity rather than converted to HMM messages.
func TestModelConstruction_MatrixAccessErrorPreserved(t *testing.T) {
	readError := errors.New("matrix read unavailable")
	invalid := &boundaryMatrix{rows: 1, cols: 1, readErr: readError}
	model, err := hmm.NewCategorical(hmm.CategoricalSpec{
		Common:   hmm.Common{Prior: []float64{1}, Transition: invalid},
		Emission: dense(t, [][]float64{{1}}),
	})
	if model != nil || !errors.Is(err, readError) {
		t.Fatalf("model=%v error=%v", model, err)
	}
	model, err = hmm.NewCategorical(hmm.CategoricalSpec{
		Common:   hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})},
		Emission: &boundaryMatrix{rows: 1, cols: 1, value: math.NaN()},
	})
	if model != nil || !errors.Is(err, hmm.ErrNonFinite) {
		t.Fatalf("model=%v error=%v", model, err)
	}
}

// TestGaussianFilterStep_RestoreMatchesBatch verifies that the Gaussian
// streaming API exactly reproduces the batch kernel under the same observation order.
func TestGaussianFilterStep_RestoreMatchesBatch(t *testing.T) {
	model, err := hmm.NewGaussian(hmm.GaussianSpec{
		Common: hmm.Common{
			Prior:      []float64{.8, .2},
			Transition: dense(t, [][]float64{{.92, .08}, {.13, .87}}),
		},
		Means:     dense(t, [][]float64{{.0, .01}, {-.04, .08}}),
		Variances: dense(t, [][]float64{{.0001, .0001}, {.0004, .0009}}),
	})
	if err != nil {
		t.Fatal(err)
	}
	observations := [][]float64{{.0, .012}, {-.006, .018}, {-.04, .08}, {-.03, .07}, {.002, .015}}
	batch, err := model.FilterDefault(context.Background(), dense(t, observations))
	if err != nil {
		t.Fatal(err)
	}
	var state *hmm.FilterState
	for _, observation := range observations {
		state, err = model.FilterStep(context.Background(), state, observation)
		if err != nil {
			t.Fatal(err)
		}
	}
	if state == nil || state.ModelID != batch.Last.ModelID || state.Steps != batch.Last.Steps ||
		math.Float64bits(state.LogLikelihood) != math.Float64bits(batch.LogLikelihood) {
		t.Fatalf("batch=%+v streaming=%+v", batch.Last, state)
	}
	for k, want := range batch.Last.LogPosterior {
		if math.Float64bits(state.LogPosterior[k]) != math.Float64bits(want) {
			t.Fatalf("state %d differs: %.17g vs %.17g", k, state.LogPosterior[k], want)
		}
	}
}

// TestViterbi_LowestIndexTie verifies deterministic predecessor and final-state
// rules independently of any floating tolerance or model-dependent randomness.
func TestViterbi_LowestIndexTie(t *testing.T) {
	m, err := hmm.NewCategorical(hmm.CategoricalSpec{
		Common:   hmm.Common{Prior: []float64{.5, .5}, Transition: dense(t, [][]float64{{.5, .5}, {.5, .5}})},
		Emission: dense(t, [][]float64{{.5, .5}, {.5, .5}}),
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := m.Decode(context.Background(), []uint32{0, 1, 1, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	for i, state := range result.States {
		if state != 0 {
			t.Fatalf("nonminimal deterministic tie at time %d: %v", i, result.States)
		}
	}
}

// TestFitGaussian_IndependentValidation checks Fit-specific option precedence,
// session shape rules, and the no-partial-result contract on invalid inputs.
func TestFitGaussian_IndependentValidation(t *testing.T) {
	initial := hmm.GaussianSpec{
		Common:    hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})},
		Means:     dense(t, [][]float64{{0}}),
		Variances: dense(t, [][]float64{{.5}}),
	}
	type fixture struct {
		name      string
		ctx       context.Context
		sequences []matrix.Matrix
		opts      hmm.FitOptions
		want      error
	}
	cases := []fixture{
		{"nil context", nil, []matrix.Matrix{dense(t, [][]float64{{0}})}, hmm.FitOptions{}, hmm.ErrInvalidOption},
		{"nil sequences", context.Background(), nil, hmm.FitOptions{}, hmm.ErrNilInput},
		{"empty sequences", context.Background(), []matrix.Matrix{}, hmm.FitOptions{}, hmm.ErrEmptyInput},
		{"nil member", context.Background(), []matrix.Matrix{nil}, hmm.FitOptions{}, hmm.ErrNilInput},
		{"typed nil member", context.Background(), []matrix.Matrix{(*boundaryMatrix)(nil)}, hmm.FitOptions{}, hmm.ErrNilInput},
		{"no rows", context.Background(), []matrix.Matrix{&boundaryMatrix{rows: 0, cols: 1}}, hmm.FitOptions{}, hmm.ErrEmptyInput},
		{"wrong feature count", context.Background(), []matrix.Matrix{dense(t, [][]float64{{0, 0}})}, hmm.FitOptions{}, hmm.ErrShape},
		{"nonfinite feature", context.Background(), []matrix.Matrix{&boundaryMatrix{rows: 1, cols: 1, value: math.Inf(-1)}}, hmm.FitOptions{}, hmm.ErrNonFinite},
		{"invalid option before data", context.Background(), nil, hmm.FitOptions{Tolerance: -1}, hmm.ErrInvalidOption},
		{"below floor before sequences", context.Background(), nil, hmm.FitOptions{VarianceFloor: 1}, hmm.ErrInvalidOption},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := hmm.FitGaussian(tc.ctx, tc.sequences, initial, tc.opts)
			if result != nil || !errors.Is(err, tc.want) {
				t.Fatalf("result=%v error=%v, want=%v", result, err, tc.want)
			}
		})
	}
}
