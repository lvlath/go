// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Regression tests use only public SAX facades and an independent rational
// population oracle. Exact translations are proved before invariance checks.
package sax_test

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/sax"
)

func TestFIXSAX_LargeOffsetSmallSpread(t *testing.T) {
	base := []float64{0, 2, 6}
	opts := sax.DefaultOptions(len(base))
	opts.ReturnPAA = true
	for _, offset := range []float64{0, 1e16} {
		values := exactAffineFixture(t, base, offset, 0)
		before := slices.Clone(values)
		result := mustEncode(t, values, opts)
		mustEqualSymbols(t, result.Symbols, []uint16{1, 3, 7}, "Gaussian-8 word")
		assertPopulationOracle(t, result, values)
		if !slices.Equal(before, values) {
			t.Fatal("Encode mutated input")
		}
	}
}

func TestFIXSAX_SignedOffsetsAndPowerOfTwoScale(t *testing.T) {
	base := []float64{0, 2, 6}
	opts := sax.DefaultOptions(len(base))
	opts.ReturnPAA = true
	for _, offset := range []float64{-1e16, 0, 1e16} {
		for _, exponent := range []int{-400, 0, 400} {
			t.Run(fmt.Sprintf("offset=%g/exponent=%d", offset, exponent), func(t *testing.T) {
				values := exactAffineFixture(t, base, offset, exponent)
				before := slices.Clone(values)
				result := mustEncode(t, values, opts)
				assertPopulationOracle(t, result, values)
				mustEqualSymbols(t, result.Symbols, []uint16{1, 3, 7}, "exact affine word")
				if !slices.Equal(before, values) {
					t.Fatal("input changed")
				}
			})
		}
	}
}

func TestSAXLargeOffset(t *testing.T) {
	// Five additional distinct fixtures exercise offset sign, local ULP spacing,
	// repeated observations, sample order and a nonuniform seven-point window.
	cases := [][]float64{
		{1e12 + 1, 1e12 + 2, 1e12 + 3},
		{1e16 + 6, 1e16, 1e16 + 2},
		{-1e16, -1e16 + 2, -1e16 + 6},
		{math.Nextafter(math.MaxFloat64, 0), math.MaxFloat64},
		{1e16, 1e16 + 8, 1e16, 1e16 + 2, 1e16 + 4, 1e16 + 6, 1e16 + 2},
	}
	for index, values := range cases {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			opts := sax.DefaultOptions(len(values))
			opts.ReturnPAA = true
			assertPopulationOracle(t, mustEncode(t, values, opts), values)
		})
	}
}

func TestSAX_ExtremeOppositeSigns(t *testing.T) {
	for _, values := range [][]float64{
		{-math.MaxFloat64, math.MaxFloat64},
		{math.MaxFloat64, 0, -math.MaxFloat64},
		{-math.MaxFloat64, 1e-200, math.MaxFloat64},
		{-math.MaxFloat64, math.Ldexp(math.SmallestNonzeroFloat64, 1024), math.MaxFloat64},
		{-1e16, 1, 1e16},
	} {
		opts := sax.DefaultOptions(len(values))
		opts.ReturnPAA = true
		assertPopulationOracle(t, mustEncode(t, values, opts), values)
	}
}

func TestFIXSAX_SubnormalPublicationAndFractionalWindow(t *testing.T) {
	unit := math.SmallestNonzeroFloat64
	for _, values := range [][]float64{{4 * unit, 6 * unit}, {0, unit, 2 * unit}} {
		opts := sax.DefaultOptions(len(values))
		opts.ReturnPAA = true
		assertPopulationOracle(t, mustEncode(t, values, opts), values)
	}
	for _, policy := range []sax.ConstantPolicy{sax.RejectConstant, sax.ZeroWord} {
		opts := sax.DefaultOptions(2)
		opts.ConstantPolicy = policy
		result, err := sax.Encode([]float64{0, unit}, opts)
		mustErrorIs(t, err, sax.ErrOverflow)
		mustNilState(t, result, true, "positive sigma rounded to zero")
	}
	// A five-sample window compressed into three segments must retain its
	// fractional geometry; compare independent rational overlap averages of Z.
	values := exactAffineFixture(t, []float64{0, 2, 6, 4, 8}, 1e16, 0)
	opts := sax.DefaultOptions(3)
	opts.ReturnPAA = true
	result := mustEncode(t, values, opts)
	z, _, _ := exactPopulationZ(values)
	want := rationalPAA(z, opts.Segments)
	for index, segment := range want {
		average, _ := segment.Float64()
		mustNear(t, result.PAA[index], average, 1e-12, 0, "fractional normalized PAA")
	}
}

func TestMindist_EncodingIDMismatch(t *testing.T) {
	// SHA-256 values independently serialized from the pinned Gaussian-8 bytes
	// and the historical v1 version; neither is derived from a current Result.
	const oldVersion = "zpopulation-scaled-twopass-v1"
	const newVersion = "zpopulation-anchor-offset-v2"
	cases := []struct {
		mode  sax.Normalization
		oldID string
		newID string
	}{
		{sax.ZPopulation, "db5b22c54ba96c43a8e73e2ab5ae4abf8d6937f89263b0056b9132311eccaa40", "b0c1c415945a548e47ad2f839de27e306067f8dc78380e82db30ecdc7bc53ad7"},
		{sax.Raw, "2c89f3851f92744c8460034a65be887ab6b5e4a97b5492c642484ea1f4b98f4f", "bb95c383c9726b85c6d18b8d3b7d737e55c0e06938d36c58fd3b5079ceb7b901"},
	}
	for _, tc := range cases {
		opts := sax.DefaultOptions(3)
		opts.Normalization = tc.mode
		opts.ReturnPAA = true
		encoder, err := sax.NewEncoder(opts)
		if err != nil {
			t.Fatal(err)
		}
		current, err := encoder.Encode([]float64{0, 2, 6})
		if err != nil {
			t.Fatal(err)
		}
		mustEqualString(t, current.NormalizationVersion, newVersion, "published version")
		mustEqualString(t, current.EncodingID, tc.newID, "independent v2 identity")
		// Persisted v1 symbols for the unshifted fixture are unchanged. Preserve
		// their actual old policy identifiers, not a fabricated arbitrary hash.
		old := *current
		old.EncodingID, old.NormalizationVersion = tc.oldID, oldVersion
		for _, pair := range [][2]*sax.Result{{&old, current}, {current, &old}, {&old, &old}} {
			distance, err := encoder.Mindist(pair[0], pair[1])
			mustErrorIs(t, err, sax.ErrIncompatible)
			mustNear(t, distance, 0, 0, 0, "no incompatible distance")
		}
		// A dishonest ID relabel alone still fails the independent version gate.
		old.EncodingID = current.EncodingID
		distance, err := encoder.Mindist(current, &old)
		mustErrorIs(t, err, sax.ErrIncompatible)
		mustNear(t, distance, 0, 0, 0, "no relabeled distance")
		if tc.mode == sax.Raw && !slices.Equal(current.PAA, []float64{0, 2, 6}) {
			t.Fatal("Raw computation changed")
		}
	}
}
