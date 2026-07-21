// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test contains validation tests for lvlath/tsp input options and
// matrix preconditions. The focus is on strict sentinel errors, determinism,
// and clean table-driven structure per lvlath Coding Standard (Rules 1–99).
package tsp_test

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// -----------------------------------------------------------------------------
// Minimal test-side matrix implementations.
// We intentionally keep them tiny and purpose-built to hit validator paths,
// per our testing plan. They satisfy matrix.Matrix without leaking internals.
// -----------------------------------------------------------------------------

// sliceMatrix is a square dense matrix backed by [][]float64.
type sliceMatrix struct {
	a [][]float64
}

var _ matrix.Matrix = sliceMatrix{}

func (m sliceMatrix) Rows() int { return len(m.a) }
func (m sliceMatrix) Cols() int {
	if len(m.a) == 0 {
		return 0
	}

	return len(m.a[0])
}
func (m sliceMatrix) At(i, j int) (float64, error) { return m.a[i][j], nil }
func (m sliceMatrix) Set(i, j int, v float64) error {
	m.a[i][j] = v

	return nil
}
func (m sliceMatrix) Clone() matrix.Matrix { return nil }

// nonSquareMatrix returns mismatched dims (e.g., 2x3) to trigger ErrNonSquare.
type nonSquareMatrix struct {
	a [][]float64
}

var _ matrix.Matrix = nonSquareMatrix{}

func (m nonSquareMatrix) Rows() int { return len(m.a) }
func (m nonSquareMatrix) Cols() int {
	if len(m.a) == 0 {
		return 0
	}
	return len(m.a[0])
}
func (m nonSquareMatrix) At(i, j int) (float64, error) { return m.a[i][j], nil }
func (m nonSquareMatrix) Set(i, j int, v float64) error {
	m.a[i][j] = v
	return nil
}
func (m nonSquareMatrix) Clone() matrix.Matrix { return nil }

// Helpers to build matrices concisely.
// All helpers are O(n^2) and allocation-minimal (single [][]float64).

// mkDense builds a square symmetric matrix with zero diagonal.
// Caller must provide a square [][]float64 with len(a[i])==len(a).
func mkDense(a [][]float64) matrix.Matrix { return sliceMatrix{a: a} }

// mkAsym builds an intentionally asymmetric square matrix (ATSP-style).
func mkAsym(a [][]float64) matrix.Matrix { return sliceMatrix{a: a} }

// mkNonSquare builds a non-square matrix to trigger ErrNonSquare.
func mkNonSquare(a [][]float64) matrix.Matrix { return nonSquareMatrix{a: a} }

// mkValid3 returns a canonical, tiny, symmetric 3×3 metric instance with 0 diagonal.
func mkValid3() matrix.Matrix {
	return mkDense([][]float64{
		{0, 1, 1.5},
		{1, 0, 2},
		{1.5, 2, 0},
	})
}

// runSolve is a thin wrapper to execute SolveWithMatrix and return only error.
// We use TwoOptOnly as a benign base algo where validation is the gatekeeper.
func runSolve(m matrix.Matrix, ids []string, opt tsp.Options) error {
	_, err := tsp.SolveMatrix(m, ids, opt)
	return err
}

// defaultOpts returns a clean Options with deterministic settings.
func defaultOpts() tsp.Options {
	o := tsp.DefaultOptions()
	o.Algo = tsp.TwoOptOnly
	o.Symmetric = true
	o.EnableLocalSearch = false
	o.Eps = epsTiny
	o.Seed = 0
	o.TimeLimit = 0
	return o
}

// -----------------------------------------------------------------------------
// 1) Validation: negatives in options (Eps<0, TimeLimit<0, TwoOptMaxIters<0)
// -----------------------------------------------------------------------------

func TestValidate_NegativeOptions_StrictSentinels(t *testing.T) {
	m := mkValid3()

	t.Run("Eps<0 → ErrInvalidOptions", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			opt := defaultOpts()
			opt.Eps = -1e-9
			err := runSolve(m, nil, opt)
			if !errors.Is(err, tsp.ErrInvalidOptions) {
				t.Fatalf("want ErrInvalidOptions, got %v", err)
			}
		})
	})

	t.Run("TimeLimit<0 → ErrInvalidOptions", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			opt := defaultOpts()
			opt.TimeLimit = -1 * time.Millisecond
			err := runSolve(m, nil, opt)
			if !errors.Is(err, tsp.ErrInvalidOptions) {
				t.Fatalf("want ErrInvalidOptions, got %v", err)
			}
		})
	})

	t.Run("TwoOptMaxIters<0 → ErrInvalidOptions", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			opt := defaultOpts()
			opt.TwoOptMaxIters = -1
			err := runSolve(m, nil, opt)
			if !errors.Is(err, tsp.ErrInvalidOptions) {
				t.Fatalf("want ErrInvalidOptions, got %v", err)
			}
		})
	})
}

// -----------------------------------------------------------------------------
// 2) Validation: Algo ↔ Symmetry (Christofides, OneTreeBound not for ATSP)
// -----------------------------------------------------------------------------

func TestValidate_AlgoSymmetry_Mismatches(t *testing.T) {
	// Asymmetric 3×3 to force ATSP nature.
	asym := mkAsym([][]float64{
		{0, 1, 2},
		{3, 0, 4},
		{5, 6, 0},
	})

	t.Run("Christofides with Symmetric=false → ErrATSPNotSupportedByAlgo", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			opt := defaultOpts()
			opt.Symmetric = false
			opt.Algo = tsp.Christofides
			err := runSolve(asym, nil, opt)
			if !errors.Is(err, tsp.ErrATSPNotSupportedByAlgo) {
				t.Fatalf("want ErrATSPNotSupportedByAlgo, got %v", err)
			}
		})
	})

	t.Run("BB with OneTreeBound and Symmetric=false → ErrATSPNotSupportedByAlgo", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			opt := defaultOpts()
			opt.Symmetric = false
			opt.Algo = tsp.BranchAndBound
			opt.BoundAlgo = tsp.OneTreeBound
			err := runSolve(asym, nil, opt)
			if !errors.Is(err, tsp.ErrATSPNotSupportedByAlgo) {
				t.Fatalf("want ErrATSPNotSupportedByAlgo, got %v", err)
			}
		})
	})
}

// -----------------------------------------------------------------------------
// 3) Validation: matrix shape & values (nil/non-square/diag/NaN/neg/+Inf)
// -----------------------------------------------------------------------------

func TestValidate_Matrix_ShapeAndValues(t *testing.T) {
	base := mkValid3()

	t.Run("nil matrix → ErrDimensionMismatch", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			var m matrix.Matrix // nil interface
			opt := defaultOpts()
			err := runSolve(m, nil, opt)
			if !errors.Is(err, tsp.ErrNilDistanceMatrix) {
				t.Fatalf("want ErrNilDistanceMatrix, got %v", err)
			}
		})
	})

	t.Run("non-square dims → ErrNonSquare", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			m := mkNonSquare([][]float64{
				{0, 1, 2}, // 2×3
				{1, 0, 2},
			})
			opt := defaultOpts()
			err := runSolve(m, nil, opt)
			if !errors.Is(err, tsp.ErrNonSquare) {
				t.Fatalf("want ErrNonSquare, got %v", err)
			}
		})
	})

	t.Run("diagonal |a_ii| > symTol → ErrNonZeroDiagonal", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			m := mkDense([][]float64{
				{1e-9, 1, 1.5}, // deliberately too large vs 1e-12 tol
				{1, 0, 2},
				{1.5, 2, 0},
			})
			opt := defaultOpts()
			err := runSolve(m, nil, opt)
			if !errors.Is(err, tsp.ErrNonZeroDiagonal) {
				t.Fatalf("want ErrNonZeroDiagonal, got %v", err)
			}
		})
	})

	t.Run("NaN entry → ErrNaNInf", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			m := mkDense([][]float64{
				{0, math.NaN(), 1},
				{1, 0, 2},
				{1, 2, 0},
			})
			opt := defaultOpts()
			err := runSolve(m, nil, opt)
			if !errors.Is(err, tsp.ErrNaNInf) {
				t.Fatalf("want ErrNaNInf, got %v", err)
			}
		})
	})

	t.Run("negative entry → ErrNegativeWeight", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			m := mkDense([][]float64{
				{0, -1, 1},
				{-1, 0, 2},
				{1, 2, 0},
			})
			opt := defaultOpts()
			err := runSolve(m, nil, opt)
			if !errors.Is(err, tsp.ErrNegativeWeight) {
				t.Fatalf("want ErrNegativeWeight, got %v", err)
			}
		})
	})

	t.Run("+Inf off-diagonal with RunMetricClosure=false → ErrIncompleteGraph", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			m := mkDense([][]float64{
				{0, math.Inf(1), 1},
				{math.Inf(1), 0, 2},
				{1, 2, 0},
			})
			opt := defaultOpts()
			opt.RunMetricClosure = false
			err := runSolve(m, nil, opt)
			if !errors.Is(err, tsp.ErrIncompleteGraph) {
				t.Fatalf("want ErrIncompleteGraph, got %v", err)
			}
		})
	})

	// Control: valid baseline should pass with default options.
	t.Run("baseline valid symmetric matrix passes", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			opt := defaultOpts()
			err := runSolve(base, nil, opt)
			if err != nil {
				t.Fatalf("unexpected error on valid baseline: %v", err)
			}
		})
	})
}

// -----------------------------------------------------------------------------
// 4) Medium: symmetry tolerance - near-equal neighbors vs hard asymmetry
// -----------------------------------------------------------------------------

func TestValidate_SymmetryTolerance(t *testing.T) {
	// Start with symmetric and perturb one mirrored pair by δ.
	makeNearSym := func(delta float64) matrix.Matrix {
		return mkAsym([][]float64{
			{0, 1, 1.5},
			{1 + delta, 0, 2},
			{1.5, 2, 0},
		})
	}

	t.Run("|a_ij-a_ji| = 1e-13 → allowed under symTol", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			m := makeNearSym(1e-13) // within 1e-12 tolerance
			opt := defaultOpts()
			opt.Symmetric = true // explicitly require symmetry check
			err := runSolve(m, nil, opt)
			if err != nil {
				t.Fatalf("unexpected error for near-symmetric matrix: %v", err)
			}
		})
	})

	t.Run("|a_ij-a_ji| = 1e-11 → ErrAsymmetry", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			m := makeNearSym(1e-11) // exceeds 1e-12 tolerance
			opt := defaultOpts()
			opt.Symmetric = true
			err := runSolve(m, nil, opt)
			if !errors.Is(err, tsp.ErrAsymmetry) {
				t.Fatalf("want ErrAsymmetry, got %v", err)
			}
		})
	})
}

// -----------------------------------------------------------------------------
// 5) Medium: IDs validation - wrong length and duplicates → ErrDimensionMismatch
// -----------------------------------------------------------------------------

func TestValidate_IDs_LengthAndDuplicates(t *testing.T) {
	m := mkValid3()

	t.Run("ids length != n → ErrDimensionMismatch", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			opt := defaultOpts()
			ids := []string{"v0", "v1"} // len=2 while n=3
			err := runSolve(m, ids, opt)
			if !errors.Is(err, tsp.ErrDimensionMismatch) {
				t.Fatalf("want ErrDimensionMismatch, got %v", err)
			}
		})
	})

	t.Run("ids contain empty string → ErrInvalidIDs", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			ids := []string{"v0", "", "v2"}
			opt := defaultOpts()
			err := runSolve(m, ids, opt)
			if !errors.Is(err, tsp.ErrInvalidIDs) {
				t.Fatalf("want ErrInvalidIDs, got %v", err)
			}
		})
	})

	t.Run("ids with duplicates → ErrInvalidIDs", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			opt := defaultOpts()
			ids := []string{"v0", "v1", "v1"} // duplicate "v1"
			err := runSolve(m, ids, opt)
			if !errors.Is(err, tsp.ErrInvalidIDs) {
				t.Fatalf("want ErrInvalidIDs, got %v", err)
			}
		})
	})
}

// -----------------------------------------------------------------------------
// 6) Special: time limit == 0 is allowed; StartVertex bounds (ok and OOR)
// -----------------------------------------------------------------------------

func TestValidate_TimeZero_And_StartBounds(t *testing.T) {
	m := mkValid3()

	t.Run("TimeLimit == 0 is permitted", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			opt := defaultOpts()
			opt.TimeLimit = 0
			err := runSolve(m, nil, opt)
			if err != nil {
				t.Fatalf("unexpected error with TimeLimit=0: %v", err)
			}
		})
	})

	t.Run("StartVertex in [0, n-1] is accepted", func(t *testing.T) {
		for _, sv := range []int{0, 2} {
			t.Run((func() string {
				if sv == 0 {
					return "start=0 ok"
				}
				return "start=n-1 ok"
			})(), func(t *testing.T) {
				Repeat(t, 3, func(t *testing.T) {
					opt := defaultOpts()
					opt.StartVertex = sv
					err := runSolve(m, nil, opt)
					if err != nil {
						t.Fatalf("unexpected error with StartVertex=%d: %v", sv, err)
					}
				})
			})
		}
	})

	t.Run("StartVertex == n → ErrStartOutOfRange", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			opt := defaultOpts()
			opt.StartVertex = 3 // n==3 → OOR
			err := runSolve(m, nil, opt)
			if !errors.Is(err, tsp.ErrStartOutOfRange) {
				t.Fatalf("want ErrStartOutOfRange, got %v", err)
			}
		})
	})
}
