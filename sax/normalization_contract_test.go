// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External normalization oracles protect fitted population metadata and
// distinguish exactly constant windows from unrepresentable nonzero variance.
package sax_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/sax"
)

func TestNormalizeWindow_NonconstantSubnormalSigmaIsOverflow(t *testing.T) {
	// sigma([0, minSubnormal]) = minSubnormal/2, below binary64.
	// It is mathematically nonzero; publishing StdDev=0 without Degenerate
	// would be a false constant classification and inconsistent metadata.
	values := []float64{0, math.SmallestNonzeroFloat64}
	for _, policy := range []sax.ConstantPolicy{sax.RejectConstant, sax.ZeroWord} {
		opts := sax.DefaultOptions(2)
		opts.ConstantPolicy = policy
		got, err := sax.Encode(values, opts)
		mustErrorIs(t, err, sax.ErrOverflow)
		mustNilState(t, got, true, "unrepresentable positive sigma")
	}
}

func TestNormalizeWindow_RepresentableSubnormalVariation(t *testing.T) {
	unit := math.SmallestNonzeroFloat64
	values := []float64{4 * unit, 6 * unit}
	opts := sax.DefaultOptions(2)
	opts.ReturnPAA = true
	result, err := sax.Encode(values, opts)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualBool(t, result.Degenerate, false, "representable subnormal is not constant")
	mustNear(t, result.Mean, 5*unit, 0, 0, "exact subnormal mean")
	mustNear(t, result.StdDev, unit, 0, 0, "exact subnormal sigma")
	mustNear(t, result.PAA[0], -1, 2e-15, 0, "first Z score")
	mustNear(t, result.PAA[1], 1, 2e-15, 0, "second Z score")
}

func TestNormalizeWindow_ExtremeFiniteSigns(t *testing.T) {
	values := []float64{-math.MaxFloat64, math.MaxFloat64}
	opts := sax.DefaultOptions(2)
	opts.ReturnPAA = true
	result, err := sax.Encode(values, opts)
	if err != nil {
		t.Fatal(err)
	}
	mustNear(t, result.Mean, 0, 0, 0, "symmetric extreme mean")
	mustNear(t, result.StdDev, math.MaxFloat64, 0, 1e-15, "extreme population sigma")
	mustNear(t, result.PAA[0], -1, 1e-14, 0, "extreme first z")
	mustNear(t, result.PAA[1], 1, 1e-14, 0, "extreme second z")
}

func TestNormalizeWindow_ScaledOffsetAndPopulationDivisor(t *testing.T) {
	values := []float64{1.5e150, 2.5e150, 3.5e150, 4.5e150}
	opts := sax.DefaultOptions(4)
	opts.ReturnPAA = true
	result, err := sax.Encode(values, opts)
	if err != nil {
		t.Fatal(err)
	}
	mustNear(t, result.Mean, 3e150, 1e135, 0, "nonzero extreme-offset mean")
	mustNear(t, result.StdDev, math.Sqrt(1.25)*1e150, 1e135, 0, "population sigma divisor n")
	for _, normalized := range result.PAA {
		if !(!math.IsNaN(normalized) && !math.IsInf(normalized, 0)) {
			t.Fatalf("non-finite normalized value %g", normalized)
		}
	}
}

func TestNormalizeWindow_ExactConstantsAndNegativeZero(t *testing.T) {
	for _, values := range [][]float64{
		{42, 42, 42, 42},
		{math.Copysign(0, -1), 0, math.Copysign(0, -1), 0},
	} {
		opts := sax.DefaultOptions(4)
		got, err := sax.Encode(values, opts)
		mustErrorIs(t, err, sax.ErrZeroVariance)
		mustNilState(t, got, true, "constant reject")
		opts.ConstantPolicy = sax.ZeroWord
		opts.ReturnPAA = true
		got, err = sax.Encode(values, opts)
		if err != nil {
			t.Fatal(err)
		}
		mustEqualBool(t, got.Degenerate, true, "ZeroWord degeneracy")
		mustNear(t, got.StdDev, 0, 0, 0, "ZeroWord sigma")
		for _, val := range got.PAA {
			mustNear(t, val, 0, 0, 0, "ZeroWord PAA coordinate")
		}
	}
}
