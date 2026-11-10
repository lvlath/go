// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test exercises small pivoted and SPD systems generated from
// bounded exact integer arithmetic. The oracle constructs B=A*X independently
// of the library, so all successful solvers must recover known coefficients.
package matrix_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

// FuzzFactorizationIndependentIntegerSystems checks reusable LU, pivoted QR
// and Cholesky solves on independent matrices sharing a known physical state.
// Its integer fixtures guarantee invertibility by strict diagonal dominance,
// tall full column rank, or a positive-definite Gram-plus-identity construction.
// Implementation:
//   - Stage 1: map bounded fuzz bytes into deterministic finite coefficients.
//   - Stage 2: form three mathematically admissible systems with exact integer RHS.
//   - Stage 3: factor, solve, and compare detached coefficients with the known state.
//
// Behavior highlights: every checked system is full-rank by construction;
// errors cannot be dismissed as accidental rank-deficient random fixtures.
// Inputs: arbitrary byte sequence folded into 21 fixed coefficient bytes.
// Returns: test failure on an unexpected factor or solve error, or wrong solution.
// Errors: runtime numeric failures on valid bounded fixtures are regressions.
// Determinism: fixed row-major construction and integer coefficient mapping.
// Complexity: O(3^3) per factor plus bounded exact residual computation.
// Notes: an exact integer coefficient oracle is independent of factor storage.
// AI-Hints: do not use another matrix solver as the expected-value generator;
// generating B from a known X avoids circular reference implementations.
func FuzzFactorizationIndependentIntegerSystems(f *testing.F) {
	for _, seed := range [][]byte{
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0},
		{9, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0},
		{255, 0, 254, 1, 253, 2, 252, 3, 251, 4, 250, 5, 249, 6, 248, 7, 247, 8, 246, 9, 0},
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, input []byte) {
		// Fuzz mutation can change the input length. Fold all inputs into a
		// fixed-size fixture, including an empty input, so each execution
		// performs a meaningful mathematical test with bounded allocation.
		var coefficients [21]byte
		for i := range coefficients {
			if len(input) > 0 {
				coefficients[i] = input[i%len(input)]
			}
		}
		// All generated values are small exact integers. Using a known
		// solution avoids an independent floating-point reference solver.
		source := [3]float64{float64(coefficients[0]%7) - 3, float64(coefficients[1]%7) - 3, float64(coefficients[2]%7) - 3}
		square := make([][]float64, 3)
		tall := make([][]float64, 6)
		for i := 0; i < 3; i++ {
			square[i] = make([]float64, 3)
			for j := 0; j < 3; j++ {
				square[i][j] = float64(coefficients[3+i*3+j]%9) - 4
				if i == j {
					square[i][j] += 16
				}
			}
			tall[i] = append([]float64(nil), square[i]...)
		}
		for i := 3; i < 6; i++ {
			tall[i] = make([]float64, 3)
			for j := 0; j < 3; j++ {
				tall[i][j] = float64(coefficients[12+(i-3)*3+j]%9) - 4
			}
		}
		// Construct an SPD matrix C = A^T*A + 3I in exact integer arithmetic.
		positive := make([][]float64, 3)
		for i := 0; i < 3; i++ {
			positive[i] = make([]float64, 3)
			for j := 0; j < 3; j++ {
				for k := 0; k < 3; k++ {
					positive[i][j] += square[k][i] * square[k][j]
				}
				if i == j {
					positive[i][j] += 3
				}
			}
		}
		testCases := []struct {
			name         string
			coefficients [][]float64
			factorize    func(matrix.Matrix) (func(context.Context, matrix.Matrix) (*matrix.SolveResult, error), error)
		}{
			{"LU", square, func(a matrix.Matrix) (func(context.Context, matrix.Matrix) (*matrix.SolveResult, error), error) {
				factor, err := matrix.FactorLU(context.Background(), a, matrix.FactorOptions{})
				if err != nil {
					return nil, err
				}
				return factor.Solve, nil
			}},
			{"QR", tall, func(a matrix.Matrix) (func(context.Context, matrix.Matrix) (*matrix.SolveResult, error), error) {
				factor, err := matrix.FactorQR(context.Background(), a, matrix.FactorOptions{})
				if err != nil {
					return nil, err
				}
				return factor.Solve, nil
			}},
			{"Cholesky", positive, func(a matrix.Matrix) (func(context.Context, matrix.Matrix) (*matrix.SolveResult, error), error) {
				factor, err := matrix.FactorCholesky(context.Background(), a, 0, 0)
				if err != nil {
					return nil, err
				}
				return factor.Solve, nil
			}},
		}
		for _, testCase := range testCases {
			t.Run(testCase.name, func(t *testing.T) {
				rhs := make([][]float64, len(testCase.coefficients))
				for i, row := range testCase.coefficients {
					result := 0.0
					for j, value := range row {
						result += value * source[j]
					}
					rhs[i] = []float64{result}
				}
				solve, err := testCase.factorize(factorFixture(t, testCase.coefficients))
				if err != nil {
					t.Fatalf("valid factor rejected: %v", err)
				}
				actual, err := solve(context.Background(), factorFixture(t, rhs))
				if err != nil {
					t.Fatalf("valid RHS rejected: %v", err)
				}
				if actual.Rank != 3 || len(actual.ResidualNorms) != 1 || math.IsNaN(actual.ConditionEstimate) || actual.ConditionEstimate < 1 {
					t.Fatalf("invalid published diagnostics: rank=%d residuals=%v condition=%g", actual.Rank, actual.ResidualNorms, actual.ConditionEstimate)
				}
				for i, expected := range source {
					factorNear(t, factorCell(t, actual.Solution, i, 0), expected, 1e-11, 1e-11)
				}
				if actual.ResidualNorms[0] > 1e-9 {
					t.Fatalf("unexpected original-coordinate residual: %.17g", actual.ResidualNorms[0])
				}
			})
		}
	})
}

// --- Statistics Fuzz ---

// Checks binary64 exponent boundaries with independent
// rational oracles, including cancellation that survives a huge scale gap.
// Fuzz seeds and their extensions are deterministic mathematical fixtures;
// they are not permission to weaken covariance or correlation tolerances.

// FuzzStatisticsCancellationAndCorrelationOracle stresses the exact rescue
// contract over independently scaled binary64 observations and two interfaces.
// The three-row construction cancels two dominant cross-products, leaving
// only the contribution of a smaller third observation.
// Implementation:
//   - Stage 1: map arbitrary int16 inputs to bounded binary exponents.
//   - Stage 2: build original binary64 values and exact big.Rat moments.
//   - Stage 3: compare covariance and Pearson output on Dense and fallback.
//
// Behavior highlights: covariance's final values stay finite for the chosen
// exponent bounds; no mathematical operation relies on floating Gram products.
// Inputs: fuzzed integer exponent coordinates, not arbitrary decimal strings.
// Returns: regression failures if public estimates diverge from the oracle.
// Errors: any matrix error is unexpected for these finite representable cases.
// Determinism: fixed 3x2 row order and binary exponent mapping.
// Complexity: O(n*d*d) kernel work plus bounded-size rational oracle work.
// Notes: the oracle treats each source float64 as its exact binary rational.
// AI-Hints: never use Covariance as its own Pearson reference; each comparison
// uses an independently computed high-precision rational moment.
func FuzzStatisticsCancellationAndCorrelationOracle(f *testing.F) {
	for _, seed := range [][3]int16{
		{480, -480, 0}, {480, -420, -480}, {100, -300, 300},
		{-480, -479, -480}, {0, -480, 480}, {250, -250, 0},
	} {
		f.Add(seed[0], seed[1], seed[2])
	}

	f.Fuzz(func(t *testing.T, dominantKey, survivorKey, counterpartKey int16) {
		// The exponent reduction bounds every squared magnitude by 2^960.
		// This prevents valid fixture generation from requiring overflowed
		// result-domain covariance even though intermediate Gram terms are huge.
		const exponentSpread = 481
		dominantExponent := int(dominantKey) % exponentSpread
		survivorExponent := int(survivorKey) % exponentSpread
		counterpartExponent := int(counterpartKey) % exponentSpread
		dominant := math.Ldexp(1, dominantExponent)
		survivor := math.Ldexp(1, survivorExponent)
		counterpart := math.Ldexp(1, counterpartExponent)

		observations := [][]float64{
			{dominant, counterpart},
			{-dominant, counterpart},
			{survivor, -2 * counterpart},
		}
		exactMeans, exactCovariances := statisticsOracleMoments(observations)
		input := statsTestDense(t, observations)
		for _, backend := range []struct {
			name  string
			input matrix.Matrix
		}{
			{"dense", input},
			{"generic", hide{input}},
		} {
			t.Run(backend.name, func(t *testing.T) {
				covariance, means, err := matrix.Covariance(backend.input)
				if err != nil {
					t.Fatalf("representable covariance failed: %v", err)
				}
				correlation, _, _, err := matrix.Correlation(backend.input)
				if err != nil {
					t.Fatalf("representable Pearson failed: %v", err)
				}
				for j := 0; j < 2; j++ {
					expectedMean := statisticsOracleFloat(exactMeans[j])
					statsTestClose(t, means[j], expectedMean,
						2*math.SmallestNonzeroFloat64, 2e-12)
					for k := 0; k < 2; k++ {
						expectedCov := statisticsOracleFloat(exactCovariances[j][k])
						statsTestClose(t, statsTestValue(t, covariance, j, k), expectedCov,
							2*math.SmallestNonzeroFloat64, 2e-12)
						expectedPearson := statisticsOracleCorrelation(exactCovariances, j, k)
						statsTestClose(t, statsTestValue(t, correlation, j, k), expectedPearson,
							2*math.SmallestNonzeroFloat64, 2e-12)
					}
				}
			})
		}
	})
}
