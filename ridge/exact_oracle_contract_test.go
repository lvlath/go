// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test checks QR outputs against independent multivariate exact oracles.
// These fixtures protect mathematics, not a particular QR pivot implementation.
package ridge_test

import (
	"context"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/ridge"
)

// assertRationalClose compares an accepted output to a mathematical exact-binary
// oracle with scale-aware absolute and relative floating-point tolerance.
func assertRationalClose(t *testing.T, got float64, expected *big.Rat, where string) {
	t.Helper()
	want, _ := expected.Float64()
	const relativeTolerance = 3e-9
	const absoluteTolerance = 3e-12
	delta := math.Abs(got - want)
	if math.IsNaN(got) || math.IsInf(got, 0) || delta > absoluteTolerance+relativeTolerance*math.Abs(want) {
		t.Fatalf("%s: actual=%.17g exact-round=%.17g error=%.4g", where, got, want, delta)
	}
}

// TestRidge_MultivariateExactRationalOracles compares intercept, SSE, penalty,
// objective, duplicate slopes, and wide designs without calling production QR.
func TestRidge_MultivariateExactRationalOracles(t *testing.T) {
	fixtures := []struct {
		name        string
		x           [][]int64
		y           []int64
		lambdaNum   int64
		lambdaDen   int64
		noIntercept bool
	}{
		{"two independent factors", [][]int64{{-2, 1}, {-1, 0}, {0, 3}, {1, 1}, {2, 4}, {3, -1}}, []int64{-2, 0, 3, 4, 7, 5}, 3, 4, false},
		{"duplicate indices", [][]int64{{1, 1}, {2, 2}, {4, 4}, {5, 5}, {7, 7}}, []int64{4, 7, 10, 15, 20}, 1, 2, false},
		{"wide spectroscopy", [][]int64{{1, 0, 3, 2, 0}, {0, 2, 1, 0, 5}}, []int64{3, 7}, 1, 1, false},
		{"no intercept", [][]int64{{1, 0}, {0, 1}, {2, 1}, {1, 2}}, []int64{4, -3, 5, -2}, 2, 1, true},
		{"one observation wide", [][]int64{{2, 3, 4, 5}}, []int64{17}, 1, 1, false},
	}
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			input := make([][]float64, len(fixture.x))
			for i, row := range fixture.x {
				input[i] = make([]float64, len(row))
				for j, value := range row {
					input[i][j] = float64(value)
				}
			}
			response := make([]float64, len(fixture.y))
			for i, value := range fixture.y {
				response[i] = float64(value)
			}
			lambda := float64(fixture.lambdaNum) / float64(fixture.lambdaDen)
			result, err := ridge.Fit(context.Background(), dense(t, input), response, ridge.Options{Lambda: lambda, NoIntercept: fixture.noIntercept})
			if err != nil {
				t.Fatal(err)
			}
			parameters, sse, penalty, objective := exactRidgeOracle(t, fixture.x, fixture.y, fixture.lambdaNum, fixture.lambdaDen, fixture.noIntercept)
			beta, intercept := result.Model.Coefficients()
			offset := 0
			if !fixture.noIntercept {
				assertRationalClose(t, intercept, parameters[0], "intercept")
				offset = 1
			} else if math.Float64bits(intercept) != math.Float64bits(0) {
				t.Fatalf("origin intercept must be +0, got %.17g", intercept)
			}
			for j := range beta {
				assertRationalClose(t, beta[j], parameters[j+offset], "physical slope")
			}
			assertRationalClose(t, result.ResidualSumSquares, sse, "SSE")
			assertRationalClose(t, result.Penalty, penalty, "Penalty")
			assertRationalClose(t, result.Objective, objective, "Objective")
		})
	}
}
