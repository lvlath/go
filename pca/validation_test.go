// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Public-surface invalid inputs and sentinel precedence tests.
package pca_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/pca"
)

type adversarialMatrix struct {
	matrix.Matrix
	badRow int
	badCol int
	value  float64
	err    error
}

func (m *adversarialMatrix) At(i, j int) (float64, error) {
	if i == m.badRow && j == m.badCol {
		return m.value, m.err
	}
	return m.Matrix.At(i, j)
}
func (m *adversarialMatrix) Clone() matrix.Matrix { return m.Matrix.Clone() }

type absentMatrix struct{ matrix.Matrix }

func (m *absentMatrix) IsNil() bool { return m == nil }

func TestPCA_FitValidationGrid(t *testing.T) {
	ctx := context.Background()
	base := fixture(t, [][]float64{{1, 2}, {3, 4}, {5, 6}})
	var denseNil *matrix.Dense
	var nilable *absentMatrix
	cases := []struct {
		name string
		x    matrix.Matrix
		opts pca.Options
		want error
	}{
		{"missing", nil, pca.Options{}, pca.ErrNilInput},
		{"typed Dense nil", denseNil, pca.Options{}, pca.ErrNilInput},
		{"typed Nilable nil", nilable, pca.Options{}, pca.ErrNilInput},
		{"unsupported", nil, pca.Options{Solver: pca.ThinSVD}, pca.ErrUnsupportedSolver},
		{"unknown solver", nil, pca.Options{Solver: pca.Solver(201)}, pca.ErrInvalidOption},
		{"negative components", nil, pca.Options{Components: -1}, pca.ErrInvalidOption},
		{"negative rotation budget", nil, pca.Options{MaxRotations: -1}, pca.ErrInvalidOption},
		{"negative rank tolerance", nil, pca.Options{RankTolerance: -1}, pca.ErrInvalidOption},
		{"nan rank tolerance", nil, pca.Options{RankTolerance: math.NaN()}, pca.ErrInvalidOption},
		{"inf eigen tolerance", nil, pca.Options{EigenTolerance: math.Inf(1)}, pca.ErrInvalidOption},
		{"zero rows", mustBlank(t, 0, 2), pca.Options{}, pca.ErrEmptyInput},
		{"zero columns", mustBlank(t, 2, 0), pca.Options{}, pca.ErrEmptyInput},
		{"single observation", mustBlank(t, 1, 2), pca.Options{}, pca.ErrShape},
		{"unattainable components", base, pca.Options{Components: 3}, pca.ErrRankDeficient},
		{"NaN cell", &adversarialMatrix{Matrix: base, badRow: 1, badCol: 1, value: math.NaN()}, pca.Options{}, pca.ErrNonFinite},
		{"positive infinity", &adversarialMatrix{Matrix: base, badRow: 1, badCol: 1, value: math.Inf(1)}, pca.Options{}, pca.ErrNonFinite},
		{"negative infinity", &adversarialMatrix{Matrix: base, badRow: 1, badCol: 1, value: math.Inf(-1)}, pca.Options{}, pca.ErrNonFinite},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := pca.Fit(ctx, tc.x, tc.opts)
			mustErrorIs(t, err, tc.want)
			if res != nil {
				t.Fatal("partial result")
			}
		})
	}
}
func mustBlank(t *testing.T, r, c int) *matrix.Dense {
	t.Helper()
	x, err := matrix.NewDense(r, c)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

func TestPCA_TransformValidationGrid(t *testing.T) {
	train := fixture(t, [][]float64{{-1, 0}, {0, 0}, {1, 0}})
	fit, err := pca.Fit(context.Background(), train, pca.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var absent *pca.Model
	if _, err = absent.Transform(context.Background(), train); !errors.Is(err, pca.ErrNilInput) {
		t.Fatal(err)
	}
	if _, err = (&pca.Model{}).Transform(context.Background(), train); !errors.Is(err, pca.ErrNilInput) {
		t.Fatal(err)
	}
	if _, err = fit.Model.Transform(nil, train); !errors.Is(err, pca.ErrInvalidOption) {
		t.Fatal(err)
	}
	wrong := fixture(t, [][]float64{{1, 2, 3}})
	if _, err = fit.Model.Transform(context.Background(), wrong); !errors.Is(err, pca.ErrShape) {
		t.Fatal(err)
	}
	raw := &adversarialMatrix{Matrix: train, badRow: 0, badCol: 0, value: math.NaN()}
	if _, err = fit.Model.Transform(context.Background(), raw); !errors.Is(err, pca.ErrNonFinite) {
		t.Fatal(err)
	}
	if _, err = fit.Model.Transform(context.Background(), nil); !errors.Is(err, pca.ErrNilInput) {
		t.Fatal(err)
	}
}
func TestPCA_InverseValidationGrid(t *testing.T) {
	train := fixture(t, [][]float64{{1, 3}, {2, 6}, {4, 1}, {5, 8}})
	fit, err := pca.Fit(context.Background(), train, pca.Options{Components: 1})
	if err != nil {
		t.Fatal(err)
	}
	var absent *pca.Model
	if _, err = absent.InverseTransform(context.Background(), train); !errors.Is(err, pca.ErrNilInput) {
		t.Fatal(err)
	}
	if _, err = (&pca.Model{}).InverseTransform(context.Background(), train); !errors.Is(err, pca.ErrNilInput) {
		t.Fatal(err)
	}
	if _, err = fit.Model.InverseTransform(nil, train); !errors.Is(err, pca.ErrInvalidOption) {
		t.Fatal(err)
	}
	if _, err = fit.Model.InverseTransform(context.Background(), train); !errors.Is(err, pca.ErrShape) {
		t.Fatal(err)
	}
	scores := fixture(t, [][]float64{{3}, {4}})
	raw := &adversarialMatrix{Matrix: scores, badRow: 0, badCol: 0, value: math.Inf(1)}
	if _, err = fit.Model.InverseTransform(context.Background(), raw); !errors.Is(err, pca.ErrNonFinite) {
		t.Fatal(err)
	}
	if _, err = fit.Model.InverseTransform(context.Background(), nil); !errors.Is(err, pca.ErrNilInput) {
		t.Fatal(err)
	}
}
func TestPCA_DetachedFitDiagnostics(t *testing.T) {
	x := fixture(t, [][]float64{{-2, -1}, {-2, 1}, {2, -1}, {2, 1}})
	result, err := pca.Fit(context.Background(), x, pca.Options{Components: 1})
	if err != nil {
		t.Fatal(err)
	}
	result.Eigenvalues[0] = -100
	result.ExplainedVarianceRatio[0] = 2
	loading := result.Model.Components()
	if err = loading.Set(0, 0, 0); err != nil {
		t.Fatal(err)
	}
	transformed, err := result.Model.Transform(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	if transformed.Rows() != 4 || transformed.Cols() != 1 {
		t.Fatal("fit diagnostics modified the model")
	}
}
func TestPCA_StandardizePopulationScale(t *testing.T) {
	x := fixture(t, [][]float64{{1, 10}, {3, 20}, {5, 30}})
	r, err := pca.Fit(context.Background(), x, pca.Options{Standardize: true, Components: 1})
	if err != nil {
		t.Fatal(err)
	}
	if r.Rank != 1 {
		t.Fatal("population standardized rank")
	}
	s, err := r.Model.Transform(context.Background(), fixture(t, [][]float64{{3, 20}}))
	if err != nil {
		t.Fatal(err)
	}
	mustNear(t, mustCell(t, s, 0, 0), 0, 1e-12)
}
func TestPCA_NilComponents(t *testing.T) {
	var m *pca.Model
	if m.Components() != nil {
		t.Fatal("nil model components")
	}
	if (&pca.Model{}).Components() != nil {
		t.Fatal("unfitted model components")
	}
}

// phantomMatrix advertises dimensions without retaining unbounded storage.
// It allows a safe external probe of integer overflow checks before matrix.At.
type phantomMatrix struct{ rows, cols int }

func (m phantomMatrix) Rows() int                    { return m.rows }
func (m phantomMatrix) Cols() int                    { return m.cols }
func (m phantomMatrix) At(int, int) (float64, error) { return 0, errors.New("unreachable At") }
func (m phantomMatrix) Set(int, int, float64) error  { return errors.New("unreachable Set") }
func (m phantomMatrix) Clone() matrix.Matrix         { return m }
func TestPCA_CheckedAllocationAndRotationArithmetic(t *testing.T) {
	if ^uint(0)>>63 == 0 {
		t.Skip("64-bit-only gigantic-shape probe")
	}
	ctx := context.Background()
	cases := []struct {
		name       string
		rows, cols int
		want       error
	}{
		{"sample bytes", 1000000000000, 2000000, pca.ErrOverflow},
		{"covariance bytes", 2, 2000000000, pca.ErrOverflow},
		{"default rotations", 2, 500000000, pca.ErrOverflow},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := pca.Fit(ctx, phantomMatrix{tc.rows, tc.cols}, pca.Options{})
			mustErrorIs(t, err, tc.want)
			if result != nil {
				t.Fatal("partial model on checked allocation failure")
			}
		})
	}
}
