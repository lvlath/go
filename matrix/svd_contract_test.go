// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package matrix_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

func svdFixture(t *testing.T, rows, cols int, values []float64) matrix.Matrix {
	t.Helper()
	a, err := matrix.NewDense(rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if err = a.Set(i, j, values[i*cols+j]); err != nil {
				t.Fatal(err)
			}
		}
	}
	return a
}

func svdEntry(t *testing.T, a matrix.Matrix, i, j int) float64 {
	t.Helper()
	v, err := a.At(i, j)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestSVD_ExactFixtures(t *testing.T) {
	cases := []struct {
		name         string
		m, n         int
		data, values []float64
		rank         int
	}{
		{"SVD01", 2, 2, []float64{3, 0, 0, 2}, []float64{3, 2}, 2},
		{"SVD02", 3, 2, []float64{3, 0, 0, 4, 0, 0}, []float64{4, 3}, 2},
		{"SVD03", 2, 3, []float64{3, 0, 0, 0, 4, 0}, []float64{4, 3}, 2},
		{"SVD04", 2, 2, []float64{1, 2, 2, 4}, []float64{5, 0}, 1},
		{"SVD05", 3, 2, make([]float64, 6), []float64{0, 0}, 0},
		{"SVD06", 3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}, []float64{1, 1, 1}, 3},
		{"SVD07", 2, 2, []float64{1, 0, 0, 1e-12}, []float64{1, 1e-12}, 2},
		{"SVD08", 3, 3, []float64{math.MaxFloat64 / 4, 0, 0, 0, math.MaxFloat64 / 4, 0, 0, 0, math.MaxFloat64 / 4}, []float64{math.MaxFloat64 / 4, math.MaxFloat64 / 4, math.MaxFloat64 / 4}, 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := svdFixture(t, tc.m, tc.n, tc.data)
			result, err := matrix.SVD(context.Background(), a, matrix.SVDOptions{})
			if err != nil {
				t.Fatal(err)
			}
			k := tc.m
			if tc.n < k {
				k = tc.n
			}
			if result.U.Rows() != tc.m || result.U.Cols() != k || result.VT.Rows() != k || result.VT.Cols() != tc.n {
				t.Fatalf("wrong economy shape")
			}
			if result.Rank != tc.rank {
				t.Fatalf("rank=%d want=%d", result.Rank, tc.rank)
			}
			for i, want := range tc.values {
				got := result.SingularValues[i]
				if math.Abs((got-want)/math.Max(math.Abs(want), math.SmallestNonzeroFloat64)) > 5e-12 {
					t.Fatalf("singular[%d]=%.17g want %.17g", i, got, want)
				}
			}
			// Use scaled per-coordinate reconstruction to avoid overflow of the test oracle.
			scale := 0.0
			for _, v := range tc.data {
				scale = math.Max(scale, math.Abs(v))
			}
			if scale == 0 {
				scale = 1
			}
			for i := 0; i < tc.m; i++ {
				for j := 0; j < tc.n; j++ {
					sum := 0.0
					for h := 0; h < k; h++ {
						sum += (result.SingularValues[h] / scale) * svdEntry(t, result.U, i, h) * svdEntry(t, result.VT, h, j)
					}
					if math.Abs(sum-tc.data[i*tc.n+j]/scale) > 2e-12 {
						t.Fatalf("reconstruction[%d,%d]=%.17g", i, j, sum)
					}
				}
			}
			for i := 0; i < k; i++ {
				for j := 0; j < k; j++ {
					target := 0.0
					if i == j {
						target = 1
					}
					left, right := 0.0, 0.0
					for row := 0; row < tc.m; row++ {
						left += svdEntry(t, result.U, row, i) * svdEntry(t, result.U, row, j)
					}
					for col := 0; col < tc.n; col++ {
						right += svdEntry(t, result.VT, i, col) * svdEntry(t, result.VT, j, col)
					}
					if math.Abs(left-target) > 2e-12 || math.Abs(right-target) > 2e-12 {
						t.Fatalf("orthogonality(%d,%d): U=%.17g VT=%.17g", i, j, left, right)
					}
				}
			}
		})
	}
}

func TestSVDResult_IsNilInterface(t *testing.T) {
	var nilResult *matrix.SVDResult
	if !nilResult.IsNil() {
		t.Fatal("nil result must be nil-like")
	}
	result, err := matrix.SVD(context.Background(), svdFixture(t, 1, 1, []float64{2}), matrix.SVDOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if result.IsNil() {
		t.Fatal("non-nil result must not be nil-like")
	}
}

func TestSVD_Validation(t *testing.T) {
	a := svdFixture(t, 1, 1, []float64{1})
	for _, tc := range []struct {
		name    string
		ctx     context.Context
		input   matrix.Matrix
		options matrix.SVDOptions
		want    error
	}{
		{"nil-context", nil, a, matrix.SVDOptions{}, matrix.ErrInvalidOption},
		{"nil-matrix", context.Background(), nil, matrix.SVDOptions{}, matrix.ErrNilMatrix},
		{"negative-sweeps", context.Background(), a, matrix.SVDOptions{MaxIterations: -1}, matrix.ErrInvalidOption},
		{"nan-tolerance", context.Background(), a, matrix.SVDOptions{Tolerance: math.NaN()}, matrix.ErrInvalidOption},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := matrix.SVD(tc.ctx, tc.input, tc.options)
			if !errors.Is(err, tc.want) || result != nil {
				t.Fatalf("result=%v err=%v", result, err)
			}
		})
	}
}
