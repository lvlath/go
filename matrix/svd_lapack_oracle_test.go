// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test checks economy SVD against frozen independent LAPACK fixtures.
// Reference: SciPy 1.17.0 scipy.linalg.lapack.dgejsv (LAPACK DGEJSV), info=0.
// These fixtures are stored as values so Go tests do not need SciPy at runtime.
package matrix_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

// TestSVD_LAPACKDGEJSVIndependentOracle compares singular values from a
// production Jacobi SVD reference rather than another path through this kernel.
func TestSVD_LAPACKDGEJSVIndependentOracle(t *testing.T) {
	cases := []struct {
		name           string
		rows, cols     int
		data, singular []float64
	}{
		{
			name: "industrial-sensor-loading-6x3", rows: 6, cols: 3,
			data: []float64{
				2.7, -0.4, 1.2, 2.6, -0.2, 1.5, 3.4, 0.1, 1.1,
				3.1, 0.3, 1.7, 4.2, 0.4, 0.9, 3.8, 0.8, 1.6,
			},
			singular: []float64{8.8191450218175174, 1.0613101322586853, 0.81012461221488019},
		},
		{
			name: "industrial-sensor-loading-transposed-3x6", rows: 3, cols: 6,
			data: []float64{
				2.7, 2.6, 3.4, 3.1, 4.2, 3.8,
				-0.4, -0.2, 0.1, 0.3, 0.4, 0.8,
				1.2, 1.5, 1.1, 1.7, 0.9, 1.6,
			},
			singular: []float64{8.8191450218175174, 1.0613101322586853, 0.81012461221488019},
		},
		{
			name: "coupled-stiffness-8x4", rows: 8, cols: 4,
			data: []float64{
				12, 2, 0, 0, 2, 11, 3, 0, 0, 3, 9, 2, 0, 0, 2, 8,
				1.5, -0.4, 2, 0, -2, 1.5, 0.3, 1, -0.8, -0.2, 2.5, 0.9,
				1, -2, -1.5, 1.1,
			},
			singular: []float64{14.625463904563366, 11.958392867816109, 8.9528823468466392, 6.073593934302874},
		},
		{
			name: "near-zero-column-4x3", rows: 4, cols: 3,
			data: []float64{
				1, -0.125, 2e-9, 0.5, 0.375, -4e-9,
				-0.25, 0.125, 6e-9, 0.75, -0.5, 8e-9,
			},
			singular: []float64{1.3972166226018692, 0.5870781119451024, 7.7863339027981303e-9},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := matrix.SVDDefault(context.Background(), svdFixture(t, tc.rows, tc.cols, tc.data))
			if err != nil {
				t.Fatal(err)
			}
			if result.Rank != len(tc.singular) {
				t.Fatalf("rank=%d, expected %d independent modes", result.Rank, len(tc.singular))
			}
			for mode, want := range tc.singular {
				got := result.SingularValues[mode]
				if math.Abs(got/want-1) > 1e-11 {
					t.Fatalf("DGEJSV mode %d: %.17g vs %.17g", mode, got, want)
				}
			}
			svdAssertOrthonormal(t, result, tc.rows, tc.cols)
			svdAssertReconstruction(t, result, tc.rows, tc.cols, tc.data, 1e-11)
		})
	}
}
