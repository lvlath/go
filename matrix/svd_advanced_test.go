// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test protects scale separation, convergence, ownership and UES errors.
// Tests exercise only the exported package boundary and independent math invariants.
package matrix_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/matrix"
)

// svdAssertOrthonormal verifies both economy bases against independent dot products.
// A converged decomposition must have unit diagonal and near-zero cross products.
func svdAssertOrthonormal(t *testing.T, result *matrix.SVDResult, m, n int) {
	t.Helper()
	k := min(m, n)
	for p := 0; p < k; p++ {
		for q := 0; q < k; q++ {
			target := 0.0
			if p == q {
				target = 1
			}
			left, right := 0.0, 0.0
			for row := 0; row < m; row++ {
				left += svdEntry(t, result.U, row, p) * svdEntry(t, result.U, row, q)
			}
			for col := 0; col < n; col++ {
				right += svdEntry(t, result.VT, p, col) * svdEntry(t, result.VT, q, col)
			}
			limit := 64 * float64(max(m, n)) * (0x1p-52)
			if math.Abs(left-target) > limit || math.Abs(right-target) > limit {
				t.Fatalf("orthonormal(%d,%d): U=%g, VT=%g, expected=%g", p, q, left, right, target)
			}
		}
	}
}

// svdAssertReconstruction checks every entry in a scale-aware coordinate system.
// Unlike a raw Frobenius sum, the oracle cannot overflow on finite large inputs.
func svdAssertReconstruction(t *testing.T, result *matrix.SVDResult, m, n int, original []float64, tolerance float64) {
	t.Helper()
	scale := 0.0
	for _, value := range original {
		scale = math.Max(scale, math.Abs(value))
	}
	if scale == 0 {
		scale = 1
	}
	for row := 0; row < m; row++ {
		for col := 0; col < n; col++ {
			got := 0.0
			for j, singular := range result.SingularValues {
				got += (singular / scale) * svdEntry(t, result.U, row, j) * svdEntry(t, result.VT, j, col)
			}
			want := original[row*n+col] / scale
			if math.Abs(got-want) > tolerance {
				t.Fatalf("reconstruction(%d,%d): got %.17g, want %.17g", row, col, got, want)
			}
		}
	}
}

// TestSVD_ExtremeColumnScales proves that globally scaling by the largest matrix
// entry cannot silently erase a much smaller, independently representable mode.
func TestSVD_ExtremeColumnScales(t *testing.T) {
	cases := []struct {
		name  string
		large float64
		small float64
	}{
		{"huge-and-tiny", 1e308, 1e-300},
		{"normal-and-subnormal", 1, math.SmallestNonzeroFloat64},
		{"two-subnormals", 2 * math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64},
		{"large-orthogonal-pair", 1e300, 1e-100},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			input := svdFixture(t, 2, 2, []float64{tc.large, 0, 0, tc.small})
			result, err := matrix.SVD(context.Background(), input, matrix.SVDOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if result.SingularValues[0] != tc.large || result.SingularValues[1] != tc.small {
				t.Fatalf("singular values %v; expected [%g,%g]", result.SingularValues, tc.large, tc.small)
			}
			if !result.Converged || result.Iterations != 0 {
				t.Fatalf("orthogonal fixture needed sweeps: %+v", result)
			}
			svdAssertOrthonormal(t, result, 2, 2)
		})
	}
}

// TestSVD_NumericRankRetainsTinyMode tests that the default rank cutoff cannot
// alter decomposition rank-k shape or zero a legitimate nonzero singular value.
func TestSVD_NumericRankRetainsTinyMode(t *testing.T) {
	input := svdFixture(t, 2, 2, []float64{1, 0, 0, 1e-16})
	result, err := matrix.SVDDefault(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if result.Rank != 1 || result.SingularValues[1] != 1e-16 {
		t.Fatalf("rank=%d singular=%v; numeric cutoff must not truncate", result.Rank, result.SingularValues)
	}
	svdAssertOrthonormal(t, result, 2, 2)
}

// TestSVD_BasisCompletionDistributedNullspace catches an erroneous fixed 0.5
// residual cutoff. Every standard candidate has residual norm only 1/sqrt(8).
func TestSVD_BasisCompletionDistributedNullspace(t *testing.T) {
	const dimension = 8
	data := make([]float64, dimension*dimension)
	for row := 0; row < dimension; row++ {
		for column := 0; column < dimension-1; column++ {
			// The seven Walsh/Hadamard columns are pairwise orthogonal and
			// perpendicular to a remaining direction spread over all rows.
			parity := 0
			for bits := row & (column + 1); bits != 0; bits >>= 1 {
				parity ^= bits & 1
			}
			value := 1.0
			if parity != 0 {
				value = -1
			}
			data[row*dimension+column] = value
		}
	}
	result, err := matrix.SVDDefault(context.Background(), svdFixture(t, dimension, dimension, data))
	if err != nil {
		t.Fatal(err)
	}
	if result.Rank != dimension-1 || result.SingularValues[dimension-1] != 0 {
		t.Fatalf("rank=%d values=%v", result.Rank, result.SingularValues)
	}
	svdAssertOrthonormal(t, result, dimension, dimension)
	svdAssertReconstruction(t, result, dimension, dimension, data, 1e-12)
}

// TestSVD_NonConvergenceReturnsDiagnostic verifies that one sweep is counted
// honestly and never reported as a successful usable orthogonal factor.
func TestSVD_NonConvergenceReturnsDiagnostic(t *testing.T) {
	input := []float64{1, 1, 1, 0, 1, 1, 0, 0, 1}
	result, err := matrix.SVD(context.Background(), svdFixture(t, 3, 3, input), matrix.SVDOptions{MaxIterations: 1})
	if !errors.Is(err, matrix.ErrNotConverged) || result == nil || result.Converged || result.Iterations != 1 {
		t.Fatalf("expected one-sweep diagnostic result; result=%+v, err=%v", result, err)
	}
	svdAssertReconstruction(t, result, 3, 3, input, 1e-12)
	converged, err := matrix.SVDDefault(context.Background(), svdFixture(t, 3, 3, input))
	if err != nil || !converged.Converged {
		t.Fatalf("default budget failed: err=%v result=%+v", err, converged)
	}
	svdAssertOrthonormal(t, converged, 3, 3)
}

// TestSVD_InvalidInputsAndOverflow defends the public error precedence and
// distinguishes a finite input with unrepresentable singular value from NaN data.
func TestSVD_InvalidInputsAndOverflow(t *testing.T) {
	valid := svdFixture(t, 1, 1, []float64{1})
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	cases := []struct {
		name  string
		ctx   context.Context
		input matrix.Matrix
		opts  matrix.SVDOptions
		want  error
	}{
		{"context-precedes-nil", nil, nil, matrix.SVDOptions{MaxIterations: -1}, matrix.ErrInvalidOption},
		{"canceled-precedes-nil", canceled, nil, matrix.SVDOptions{}, context.Canceled},
		{"nil-precedes-options", context.Background(), nil, matrix.SVDOptions{MaxIterations: -1}, matrix.ErrNilMatrix},
		{"negative-tolerance", context.Background(), valid, matrix.SVDOptions{Tolerance: -1}, matrix.ErrInvalidOption},
		{"tolerance-one", context.Background(), valid, matrix.SVDOptions{Tolerance: 1}, matrix.ErrInvalidOption},
		{"positive-infinity", context.Background(), valid, matrix.SVDOptions{Tolerance: math.Inf(1)}, matrix.ErrInvalidOption},
		{"negative-infinity", context.Background(), valid, matrix.SVDOptions{Tolerance: math.Inf(-1)}, matrix.ErrInvalidOption},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := matrix.SVD(tc.ctx, tc.input, tc.opts)
			if !errors.Is(err, tc.want) || result != nil {
				t.Fatalf("got result=%v err=%v, want nil result and %v", result, err, tc.want)
			}
		})
	}
	large := svdFixture(t, 2, 1, []float64{math.MaxFloat64, math.MaxFloat64})
	if result, err := matrix.SVDDefault(context.Background(), large); result != nil || !errors.Is(err, matrix.ErrOverflow) {
		t.Fatalf("unrepresentable norm: result=%v err=%v", result, err)
	}
}

// TestSVD_MutationAndConcurrentReaders confirms all outputs are detached and
// simultaneous calls on one immutable input do not share working state.
func TestSVD_MutationAndConcurrentReaders(t *testing.T) {
	const rows, cols = 24, 6
	data := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[i*cols+j] = math.Sin(float64(i+j)) + float64((i*j)%7)
		}
	}
	input := svdFixture(t, rows, cols, data)
	first, err := matrix.SVDDefault(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if err = first.U.Set(0, 0, 0); err != nil {
		t.Fatal(err)
	}
	first.SingularValues[0] = -1 // Caller-owned result mutation must not reach input.
	if got := svdEntry(t, input, 0, 0); got != data[0] {
		t.Fatalf("input was mutated: got %.17g want %.17g", got, data[0])
	}
	second, err := matrix.SVDDefault(context.Background(), input)
	if err != nil || second.SingularValues[0] <= 0 {
		t.Fatalf("result mutation leaked: result=%+v err=%v", second, err)
	}

	const callers = 64
	var group sync.WaitGroup
	errorsFromWorkers := make(chan error, callers)
	for i := 0; i < callers; i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			computed, solveErr := matrix.SVDDefault(context.Background(), input)
			if solveErr == nil && (!computed.Converged || computed.Rank != cols) {
				solveErr = errors.New("unexpected SVD result in concurrent worker")
			}
			errorsFromWorkers <- solveErr
		}()
	}
	group.Wait()
	close(errorsFromWorkers)
	for workerErr := range errorsFromWorkers {
		if workerErr != nil {
			t.Fatal(workerErr)
		}
	}
}

// TestSVD_CanonicalSigns preserves reconstruction while enforcing the earliest
// maximum-magnitude left coordinate as the public orientation convention.
func TestSVD_CanonicalSigns(t *testing.T) {
	data := []float64{-4, 1, 0, -2, 0, -3}
	result, err := matrix.SVDDefault(context.Background(), svdFixture(t, 3, 2, data))
	if err != nil {
		t.Fatal(err)
	}
	for mode := range result.SingularValues {
		maximum, signed := 0.0, 0.0
		for row := 0; row < 3; row++ {
			value := svdEntry(t, result.U, row, mode)
			if math.Abs(value) > maximum {
				maximum, signed = math.Abs(value), value
			}
		}
		if signed < 0 {
			t.Fatalf("mode %d canonical pivot is negative: %.17g", mode, signed)
		}
	}
	svdAssertOrthonormal(t, result, 3, 2)
	svdAssertReconstruction(t, result, 3, 2, data, 1e-12)
}

// TestSVD_UnrepresentablePairRotationMustNotReportConvergence protects the
// mathematical distinction between two representable input columns and a
// right singular-vector correction smaller than any representable binary64.
// An extremely small physical tangent rounds to zero, so it cannot satisfy
// the requested correlation criterion and must return a diagnostic failure.
func TestSVD_UnrepresentablePairRotationMustNotReportConvergence(t *testing.T) {
	data := []float64{1e300, 1e-100, 0, 1e-100}
	result, err := matrix.SVDDefault(context.Background(), svdFixture(t, 2, 2, data))
	if !errors.Is(err, matrix.ErrNotConverged) || result == nil || result.Converged {
		t.Fatalf("unrepresentable tangent must report a diagnostic error, result=%+v err=%v", result, err)
	}
	if result.Iterations != 1 {
		t.Fatalf("a stagnant completed sweep must count once, got %d", result.Iterations)
	}
}

// svdCancelOnLastRead cancels an external context immediately after the last
// input cell has been read, testing the boundary between validation and sweeps.
type svdCancelOnLastRead struct {
	rows, cols int
	reads      int
	cancel     context.CancelFunc
}

func (s *svdCancelOnLastRead) Rows() int { return s.rows }
func (s *svdCancelOnLastRead) Cols() int { return s.cols }
func (s *svdCancelOnLastRead) At(i, j int) (float64, error) {
	s.reads++
	if s.reads == s.rows*s.cols {
		s.cancel()
	}
	if i == j {
		return 2, nil
	}
	return 1, nil
}
func (s *svdCancelOnLastRead) Set(int, int, float64) error { return nil }
func (s *svdCancelOnLastRead) Clone() matrix.Matrix {
	// This is a read-only test fixture; a copy preserves the matrix shape.
	return &svdCancelOnLastRead{rows: s.rows, cols: s.cols, cancel: s.cancel}
}

// TestSVD_CancellationAtJacobiBoundary ensures a cancellation arriving exactly
// between the finite-input scan and the first pair cannot publish a result.
func TestSVD_CancellationAtJacobiBoundary(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	input := &svdCancelOnLastRead{rows: 3, cols: 3, cancel: cancel}
	result, err := matrix.SVDDefault(ctx, input)
	if result != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation with nil result, got result=%+v err=%v", result, err)
	}
}
