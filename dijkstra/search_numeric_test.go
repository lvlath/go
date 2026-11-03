// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra_test tests internal exact-sum predicates against rational arithmetic.
// AI-Hints:
//   - Rational oracle is test-only and must not call the production sum certificate.
//   - Do not replace non-representability with an epsilon-based success assertion.
package dijkstra_test

import (
	"errors"
	"github.com/lvlath/go/dijkstra"
	"math"
	"math/big"
	"math/rand"
	"testing"
)

// TestSplitSearchSum_RationalIdentity checks FastTwoSum's exact residual property.
// Binary64 inputs are interpreted as exact rationals without decimal re-parsing.
func TestSplitSearchSum_RationalIdentity(t *testing.T) {
	rng := rand.New(rand.NewSource(20261030))
	pairs := [][2]float64{
		{0.9, 0.1},
		{0.1, 0.2},
		{math.Ldexp(1, 53), 1},
		{math.MaxFloat64, math.MaxFloat64},
		{math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64},
		{math.Ldexp(1, -1000), math.Ldexp(1, -1001)},
	}
	for index := 0; index < 500; index++ {
		leftExponent := rng.Intn(1800) - 900
		rightExponent := rng.Intn(1800) - 900
		pairs = append(pairs, [2]float64{
			math.Ldexp(float64(1+rng.Intn(32)), leftExponent),
			math.Ldexp(float64(1+rng.Intn(32)), rightExponent),
		})
	}
	for _, pair := range pairs {
		left, right := pair[0], pair[1]
		exact := new(big.Rat).Add(new(big.Rat).SetFloat64(left), new(big.Rat).SetFloat64(right))
		hi, lo, overflow := dijkstra.SearchTestSplitSum(left, right)
		if overflow {
			if !math.IsInf(hi, 1) || exact.Cmp(new(big.Rat).SetFloat64(math.MaxFloat64)) <= 0 {
				t.Fatalf("false/non-positive overflow: a=%g b=%g sum=%g", left, right, hi)
			}
			continue
		}
		got := new(big.Rat).Add(new(big.Rat).SetFloat64(hi), new(big.Rat).SetFloat64(lo))
		if got.Cmp(exact) != 0 {
			t.Fatalf("FastTwoSum lost exact residual: a=%g b=%g hi=%g lo=%g", left, right, hi, lo)
		}
		if lo == 0 && new(big.Rat).SetFloat64(hi).Cmp(exact) != 0 {
			t.Fatalf("false exact-storage certificate: a=%g b=%g", left, right)
		}
	}
}

// TestAddSearchCost_RationalCutoff verifies exact admission and storage refusal.
// The predicate is independent of floating-point grouping or test tolerances.
func TestAddSearchCost_RationalCutoff(t *testing.T) {
	cases := []struct {
		a, b, cutoff float64
	}{
		{0.9, 0.1, 1},
		{0.9, 0.1, math.Nextafter(1, math.Inf(1))},
		{0.1, 0.2, math.Inf(1)},
		{math.Ldexp(1, 53), 1, math.Inf(1)},
		{math.Ldexp(1, 53), 1, math.Ldexp(1, 53)},
		{math.MaxFloat64, math.MaxFloat64, math.MaxFloat64},
		{math.MaxFloat64, math.MaxFloat64, math.Inf(1)},
		{math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64, 0},
		{math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64, math.Inf(1)},
		{0, 0, 0},
	}
	for _, tc := range cases {
		exact := new(big.Rat).Add(new(big.Rat).SetFloat64(tc.a), new(big.Rat).SetFloat64(tc.b))
		cost, allowed, err := dijkstra.SearchTestAddCost(tc.a, tc.b, tc.cutoff)
		if !math.IsInf(tc.cutoff, 1) && exact.Cmp(new(big.Rat).SetFloat64(tc.cutoff)) > 0 {
			if allowed || err != nil {
				t.Fatalf("out-of-cutoff sum a=%g b=%g cutoff=%g: allowed=%t err=%v", tc.a, tc.b, tc.cutoff, allowed, err)
			}
			continue
		}
		if exact.Cmp(new(big.Rat).SetFloat64(math.MaxFloat64)) > 0 {
			if allowed || !errors.Is(err, dijkstra.ErrDistanceOverflow) {
				t.Fatalf("unbounded overflow a=%g b=%g: allowed=%t err=%v", tc.a, tc.b, allowed, err)
			}
			continue
		}
		if err == nil {
			if !allowed || new(big.Rat).SetFloat64(cost).Cmp(exact) != 0 {
				t.Fatalf("inexact successful cost a=%g b=%g cost=%g", tc.a, tc.b, cost)
			}
		} else if !errors.Is(err, dijkstra.ErrInexactArithmetic) || allowed {
			t.Fatalf("incorrect arithmetic classification a=%g b=%g: %v", tc.a, tc.b, err)
		}
	}
}
