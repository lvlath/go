// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File oracle_test.go independently evaluates the mathematical definition with big.Rat.
// AI-Hints: do not reuse production peak tracking, fixed-width products or rounding.
package drawdown_test

import (
	"context"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/drawdown"
)

// TestDrawdown_RationalOracle compares all output bits and indices to an independently rescanned exact oracle.
func TestDrawdown_RationalOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(3608))
	for trial := 0; trial < 800; trial++ {
		n := 2 + rng.Intn(23)
		equity := make([]float64, n)
		equity[0] = finitePositive(rng)
		for i := 1; i < n; i++ {
			switch rng.Intn(5) {
			case 0:
				equity[i] = 0
			case 1:
				equity[i] = equity[rng.Intn(i)]
			case 2:
				equity[i] = math.Nextafter(equity[i-1], 0)
			default:
				equity[i] = finitePositive(rng)
			}
		}
		mustResult(t, mustCompute(t, equity), rationalOracle(equity))
	}
	// Cross-peak exact comparisons with close exponent products, large exponent
	// gaps in both directions, and equal ratios across different significands.
	for _, equity := range [][]float64{
		{3, 2, 5, 3}, {3, 1, 5, 2}, {3, 1, 9, 3}, {3, 2, 9, 5},
		{1, 0.125, 16, 15}, {1, .99, 16, .001}, {3, 2, 7, 5},
		{math.SmallestNonzeroFloat64, 0, math.MaxFloat64, 1},
		{math.SmallestNonzeroFloat64 * 3, math.SmallestNonzeroFloat64, math.MaxFloat64, math.SmallestNonzeroFloat64},
	} {
		mustResult(t, mustCompute(t, equity), rationalOracle(equity))
	}
}

// TestDrawdown_PrefixInvariance ensures adding future observations never changes already computed depths.
func TestDrawdown_PrefixInvariance(t *testing.T) {
	equity := []float64{100, 120, 60, 80, 120, 240, 30, 250, 240, 250}
	full := mustCompute(t, equity)
	for n := 1; n <= len(equity); n++ {
		prefix := mustCompute(t, equity[:n])
		mustEqualSlice(t, prefix.Depth, full.Depth[:n])
		mustResult(t, prefix, rationalOracle(equity[:n]))
	}
}

// TestDrawdown_ExactScaleInvariance checks exact binary scaling without changing residual ratios.
func TestDrawdown_ExactScaleInvariance(t *testing.T) {
	equity := []float64{1, 3, 2, 3, 6, 1, 6, 0, 8}
	baseline := mustCompute(t, equity)
	for _, exponent := range []int{-1074, -1000, -20, 20, 1000, 1020} {
		scaled := make([]float64, len(equity))
		for i, v := range equity {
			scaled[i] = math.Ldexp(v, exponent)
		}
		mustResult(t, mustCompute(t, scaled), baseline)
	}
}

// FuzzDrawdown_PublicBoundary exercises raw binary64 patterns, chronological
// validation, cancellation precedence and exact rational parity of valid inputs.
func FuzzDrawdown_PublicBoundary(f *testing.F) {
	for _, values := range [][]float64{
		nil, {0}, {1}, {1, -1}, {1, math.NaN()}, {math.Inf(-1)}, {1, math.Inf(1)},
		{1, math.Nextafter(1, 0)}, {3, 1, 6, 2}, {1, 0x1p-55, 2, 0x1p-55, 2},
		{math.MaxFloat64, math.SmallestNonzeroFloat64, 0},
	} {
		f.Add(encodeEquity(values...), false)
	}
	f.Add(encodeEquity(0), true)
	f.Fuzz(func(t *testing.T, data []byte, canceled bool) {
		// Limit the deliberately quadratic oracle, not the public API.
		n := min(len(data)/8, 32)
		equity := make([]float64, n)
		for i := range equity {
			equity[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[8*i:]))
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		var wantErr error
		if canceled {
			cancel()
			wantErr = context.Canceled
		} else {
			wantErr = oracleError(equity)
		}
		got, err := drawdown.Compute(ctx, equity)
		if wantErr != nil {
			mustErrorIs(t, err, wantErr)
			mustNilState(t, got, true, "fuzz error result")
			return
		}
		if err != nil {
			t.Fatalf("valid input: %v", err)
		}
		mustResult(t, got, rationalOracle(equity))
	})
}
