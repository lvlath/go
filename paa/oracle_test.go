// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Independent quadratic exact rational test oracle only.
// AI-Hints: This intentionally does not replicate the production two-pointer scan.

package paa_test

import (
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/paa"
)

// oraclePAA enumerates all sample/segment pairs with rational arithmetic.
// It is deliberately O(n*w), suitable only for small test fixtures. SetFloat64
// reads the exact IEEE-754 binary rational, not an approximate decimal literal.
func oraclePAA(t *testing.T, values []float64, w int) []*big.Rat {
	t.Helper()
	n := len(values)
	result := make([]*big.Rat, w)
	for j := 0; j < w; j++ {
		sum := new(big.Rat)
		left, right := int64(j*n), int64((j+1)*n)
		for i, x := range values {
			lo, hi := int64(i*w), int64((i+1)*w)
			if lo < left {
				lo = left
			}
			if hi > right {
				hi = right
			}
			if hi <= lo {
				continue
			}
			weighted := new(big.Rat).SetFloat64(x)
			weighted.Mul(weighted, new(big.Rat).SetFrac64(hi-lo, int64(n)))
			sum.Add(sum, weighted)
		}
		result[j] = sum
	}
	return result
}

// assertOracle compares differences in high-precision rational units so that
// neither subtraction of extreme float64 values nor test-side sums can overflow.
func assertOracle(t *testing.T, got float64, want *big.Rat) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("nonfinite result %v", got)
	}
	g := new(big.Rat).SetFloat64(got)
	diff := new(big.Rat).Sub(g, want)
	diff.Abs(diff)
	bound := new(big.Rat).Abs(new(big.Rat).Set(want))
	if bound.Cmp(big.NewRat(1, 1)) < 0 {
		bound.SetInt64(1)
	}
	bound.Mul(bound, new(big.Rat).SetFloat64(3e-14))
	if diff.Cmp(bound) > 0 {
		t.Fatalf("got %.17g differs from rational oracle %s by %s > %s", got, want.RatString(), diff.RatString(), bound.RatString())
	}
}

// assertExactSegments independently checks both numerators and denominator.
func assertExactSegments(t *testing.T, result *paa.Result, n, w int) {
	t.Helper()
	if result == nil {
		t.Fatal("nil result")
	}
	if result.OriginalLength != n || len(result.Values) != w || len(result.Segments) != w {
		t.Fatalf("shape: %+v", result)
	}
	for j, s := range result.Segments {
		want := paa.Segment{LeftNumerator: int64(j * n), RightNumerator: int64((j + 1) * n), Denominator: int64(w)}
		if s != want {
			t.Fatalf("segment %d: got %+v want %+v", j, s, want)
		}
	}
}
