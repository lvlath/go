// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External fuzz testing targets signed-int64 order/overflow/cadence boundaries.
// The reference calculation uses math/big.Int instead of production arithmetic.
package series_test

import (
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/series"
)

// FuzzValidate_AdjacentIntegerOracle compares errors against exact integer math.
// It exercises arbitrary timestamp pairs and cadence values, including extremes,
// without inferring calendars or modifying the public input policy.
func FuzzValidate_AdjacentIntegerOracle(f *testing.F) {
	// These seeds pin zero crossings, duplicate order, negative cadence, and gaps.
	for _, sample := range [][3]int64{
		{math.MinInt64, 0, 0},
		{math.MinInt64, -1, 1},
		{-1, math.MaxInt64, 10},
		{0, 10, 10},
		{0, 30, 10},
		{10, 10, 0},
		{10, 0, 1},
		{0, 3, 2},
		{0, 1, -1},
	} {
		f.Add(sample[0], sample[1], sample[2])
	}

	f.Fuzz(func(t *testing.T, previous, current, cadence int64) {
		// Compute the contract classification with arbitrary-precision integers.
		want := error(nil)
		switch {
		case cadence < 0:
			want = series.ErrInvalidOption
		case current <= previous:
			want = series.ErrOrder
		default:
			span := new(big.Int).Sub(big.NewInt(current), big.NewInt(previous))
			if !span.IsInt64() {
				want = series.ErrOverflow
			} else if cadence > 0 {
				cadenceBig := big.NewInt(cadence)
				if new(big.Int).Mod(span, cadenceBig).Sign() != 0 {
					want = series.ErrGap
				}
			}
		}

		result, err := series.Validate(
			[]int64{previous, current},
			[]float64{1, 2},
			series.GridOptions{CadenceNS: cadence, AllowGaps: true},
		)
		mustErrorIs(t, err, want)
		if want != nil {
			if result != nil {
				t.Fatalf("failed grid published result %+v", result)
			}
			return
		}
		if result == nil || result.Samples != 2 || result.Gaps == nil {
			t.Fatalf("valid grid returned invalid artifact %+v", result)
		}
		if cadence > 0 {
			span := new(big.Int).Sub(big.NewInt(current), big.NewInt(previous))
			missing := span.Int64()/cadence - 1
			if missing > 0 {
				assertGaps(t, result.Gaps, []series.Gap{{AfterIndex: 0, MissingCount: missing}})
			} else {
				assertGaps(t, result.Gaps, []series.Gap{})
			}
		} else {
			assertGaps(t, result.Gaps, []series.Gap{})
		}
	})
}
