// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test checks binary64 exponent boundaries with independent
// rational oracles, including cancellation that survives a huge scale gap.
// Fuzz seeds and their extensions are deterministic mathematical fixtures;
// they are not permission to weaken covariance or correlation tolerances.
package matrix_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

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
