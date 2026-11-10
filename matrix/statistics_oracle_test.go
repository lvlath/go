// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test compares statistical reductions with independent exact
// binary-rational arithmetic, not another float64 implementation of the kernel.
package matrix_test

import (
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/lvlath/go/matrix"
)

const (
	statisticsOraclePrecision       = 512
	statisticsOracleSeed      int64 = 20261109
)

// TestStatisticsIndependentRationalOracle runs seeded integer-rational fixtures
// at many binary scales. All comparisons use the original float64 input bits
// through big.Rat, so decimal-printing drift cannot affect the expected value.
func TestStatisticsIndependentRationalOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(statisticsOracleSeed))
	for trial := 0; trial < 160; trial++ {
		rows, cols := 3+rng.Intn(5), 1+rng.Intn(4)
		values := make([][]float64, rows)
		exponents := make([]int, cols)
		for j := range exponents {
			exponents[j] = -400 + rng.Intn(801)
		}
		for i := range values {
			values[i] = make([]float64, cols)
			for j := 0; j < cols; j++ {
				// Deliberately use finite exact binary rationals, with very
				// different magnitude regimes between independent columns.
				values[i][j] = math.Ldexp(float64(rng.Intn(511)-255)/16, exponents[j])
			}
		}
		input := statsTestDense(t, values)
		exactMeans, exactCovariance := statisticsOracleMoments(values)
		gotCovariance, gotMeans, err := matrix.Covariance(input)
		if err != nil {
			t.Fatalf("trial %d covariance: %v", trial, err)
		}
		gotCorrelation, corrMeans, gotStd, err := matrix.Correlation(input)
		if err != nil {
			t.Fatalf("trial %d correlation: %v", trial, err)
		}
		for j := 0; j < cols; j++ {
			wantMean := statisticsOracleFloat(exactMeans[j])
			// An absolute budget scales with the original data, not with
			// an accidentally rounded-to-zero expected mean.
			meanBudget := math.Ldexp(1e-12, exponents[j])
			statsTestClose(t, gotMeans[j], wantMean, meanBudget, 1e-12)
			statsTestClose(t, corrMeans[j], wantMean, meanBudget, 1e-12)

			wantStd := new(big.Float).SetPrec(statisticsOraclePrecision).Sqrt(new(big.Float).SetPrec(statisticsOraclePrecision).SetRat(exactCovariance[j][j]))
			std64, _ := wantStd.Float64()
			statsTestClose(t, gotStd[j], std64, math.Ldexp(1e-12, exponents[j]), 1e-12)
			for k := 0; k < cols; k++ {
				wantCov := statisticsOracleFloat(exactCovariance[j][k])
				gotCov := statsTestValue(t, gotCovariance, j, k)
				budget := math.Ldexp(1e-11, exponents[j]+exponents[k])
				statsTestClose(t, gotCov, wantCov, budget, 1e-12)

				wantCorr := statisticsOracleCorrelation(exactCovariance, j, k)
				gotCorr := statsTestValue(t, gotCorrelation, j, k)
				statsTestClose(t, gotCorr, wantCorr, 1e-12, 1e-12)
			}
		}
	}
}

// TestStatisticsOracleSubnormalAndHugeOffset checks extreme cases for which
// an ordinary Gram computation is numerically undefined, but the final
// covariance or L2 normalization remains representable.
func TestStatisticsOracleSubnormalAndHugeOffset(t *testing.T) {
	for _, observations := range [][][]float64{
		{{-1e154}, {0}, {1e154}},
		{{math.SmallestNonzeroFloat64}, {0}, {math.SmallestNonzeroFloat64}},
		{{math.Ldexp(1, 450)}, {math.Nextafter(math.Ldexp(1, 450), math.Inf(1))}},
	} {
		exactMeans, exactCov := statisticsOracleMoments(observations)
		gotCov, gotMeans, err := matrix.Covariance(statsTestDense(t, observations))
		if err != nil {
			t.Fatal(err)
		}
		statsTestClose(t, gotMeans[0], statisticsOracleFloat(exactMeans[0]), math.Abs(statisticsOracleFloat(exactMeans[0]))*1e-13, 1e-13)
		want := statisticsOracleFloat(exactCov[0][0])
		if want == 0 {
			statsTestClose(t, statsTestValue(t, gotCov, 0, 0), want, 0, 0)
		} else {
			statsTestClose(t, statsTestValue(t, gotCov, 0, 0), want, 0, 2e-13)
		}
	}
}

// TestStatisticsCancellationAcrossExponentRange stresses a cancellation law:
// two opposite huge terms cancel in a cross-moment, leaving only a small third
// observation. The expected covariance is computed from exact binary fractions.
// A zero result is not accepted merely because the terms were normalized away.
func TestStatisticsCancellationAcrossExponentRange(t *testing.T) {
	rng := rand.New(rand.NewSource(statisticsOracleSeed + 1))
	for trial := 0; trial < 120; trial++ {
		largeExponent := 100 + rng.Intn(399)
		smallExponent := -650 + rng.Intn(1100)
		otherExponent := -350 + rng.Intn(750)
		large := math.Ldexp(1, largeExponent)
		small := math.Ldexp(1, smallExponent)
		other := math.Ldexp(1, otherExponent)
		if trial%2 != 0 {
			small = -small
		}
		observations := [][]float64{{large, other}, {-large, other}, {small, -2 * other}}
		_, exactCov := statisticsOracleMoments(observations)
		want := statisticsOracleFloat(exactCov[0][1])
		got, _, err := matrix.Covariance(statsTestDense(t, observations))
		if err != nil {
			t.Fatalf("trial %d covariance: %v", trial, err)
		}
		value := statsTestValue(t, got, 0, 1)
		if want == 0 {
			if value != 0 {
				t.Fatalf("trial %d spurious cross moment %g", trial, value)
			}
			continue
		}
		statsTestClose(t, value, want, 2*math.SmallestNonzeroFloat64, 1e-12)
	}
}

// TestStatisticsMixedExponentRationalOracle verifies finite results when
// different observations in one column occupy widely separated binary
// exponent bands. Such inputs invalidate assumptions that every column has
// one representative floating-point magnitude. A fixed seed makes any
// numerical counterexample reproducible without a timing-based fuzz harness.
func TestStatisticsMixedExponentRationalOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(statisticsOracleSeed + 2))
	verifiedCases := 0
	for trial := 0; trial < 900; trial++ {
		rows := 3 + rng.Intn(4)
		observations := make([][]float64, rows)
		baseExponent := rng.Intn(1901) - 950
		for i := range observations {
			observations[i] = make([]float64, 2)
			for j := range observations[i] {
				exponent := baseExponent
				if j == 1 || rng.Intn(4) == 0 {
					exponent = rng.Intn(1901) - 950
				}
				value := math.Ldexp(float64(1+rng.Intn(9)), exponent)
				if rng.Intn(2) == 0 {
					value = -value
				}
				observations[i][j] = value
			}
		}
		exactMeans, exactCovariance := statisticsOracleMoments(observations)
		// The finite-output contract cannot demand success when the exact
		// mathematical covariance is outside the float64 result domain.
		representable := true
		for j := 0; j < 2; j++ {
			if math.IsInf(statisticsOracleFloat(exactMeans[j]), 0) {
				representable = false
			}
			for k := 0; k < 2; k++ {
				if math.IsInf(statisticsOracleFloat(exactCovariance[j][k]), 0) {
					representable = false
				}
			}
		}
		if !representable {
			continue
		}
		verifiedCases++
		cov, means, err := matrix.Covariance(statsTestDense(t, observations))
		if err != nil {
			t.Fatalf("trial %d representable covariance returned error: %v", trial, err)
		}
		for j := 0; j < 2; j++ {
			wantMean := statisticsOracleFloat(exactMeans[j])
			if wantMean != 0 {
				statsTestClose(t, means[j], wantMean, 0, 1e-9)
			}
			for k := 0; k < 2; k++ {
				wantCovariance := statisticsOracleFloat(exactCovariance[j][k])
				if wantCovariance != 0 {
					statsTestClose(t, statsTestValue(t, cov, j, k), wantCovariance, 0, 1e-9)
				}
			}
		}
	}
	if verifiedCases < 100 {
		t.Fatalf("only %d representable fixtures were exercised", verifiedCases)
	}
}
