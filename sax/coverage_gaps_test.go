// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External admission and arithmetic fixtures exercise public failure contracts.
// The optional asset fixture test runs in separate builds, never mutating the
// process-wide cache of a valid Encoder or adding a production test hook.
package sax_test

import (
	"math"
	"os"
	"testing"

	"github.com/lvlath/go/sax"
)

func TestSAX_CodecovGapsCoverage(t *testing.T) {
	t.Run("Gaussian facade", func(t *testing.T) {
		for _, tc := range []struct {
			name               string
			values             []float64
			segments, alphabet int
			want               error
		}{
			{"small alphabet", nil, 2, 1, sax.ErrInvalidOption},
			{"large alphabet", nil, 2, 257, sax.ErrInvalidOption},
			{"zero segments", nil, 0, 8, sax.ErrInvalidOption},
			{"nil", nil, 2, 8, sax.ErrNilInput},
			{"empty", []float64{}, 2, 8, sax.ErrEmptyInput},
			{"w exceeds n", []float64{math.NaN()}, 2, 8, sax.ErrInvalidOption},
			{"nonfinite", []float64{0, math.Inf(1)}, 2, 8, sax.ErrNonFinite},
			{"constant", []float64{2, 2}, 2, 8, sax.ErrZeroVariance},
		} {
			t.Run(tc.name, func(t *testing.T) {
				result, err := sax.EncodeGaussian(tc.values, tc.segments, tc.alphabet)
				mustErrorIs(t, err, tc.want)
				mustNilState(t, result, true, "failed Gaussian facade")
			})
		}
		result, err := sax.EncodeGaussian([]float64{0, 2, 6}, 3, 8)
		if err != nil {
			t.Fatal(err)
		}
		mustEqualSymbols(t, result.Symbols, []uint16{1, 3, 7}, "Gaussian wrapper")
	})
	t.Run("custom raw facade", func(t *testing.T) {
		for _, breaks := range [][]float64{nil, {}, make([]float64, 256), {1, 0}, {0, 0}, {math.NaN()}, {math.Inf(-1)}} {
			result, err := sax.EncodeRawWithBreakpoints([]float64{0, 1}, 2, breaks, "custom")
			mustErrorIs(t, err, sax.ErrInvalidOption)
			mustNilState(t, result, true, "invalid custom wrapper")
		}
		result, err := sax.EncodeRawWithBreakpoints([]float64{0, 1}, 2, []float64{0, 1}, "custom")
		if err != nil {
			t.Fatal(err)
		}
		mustEqualSymbols(t, result.Symbols, []uint16{1, 2}, "right-bin wrapper")
	})
	t.Run("pair facade precedence", func(t *testing.T) {
		opts := sax.DefaultOptions(2)
		bad := opts
		bad.Alphabet = 257
		for _, tc := range []struct {
			name        string
			left, right []float64
			opts        sax.Options
			want        error
		}{
			{"options before nil", nil, nil, bad, sax.ErrInvalidOption},
			{"left nil", nil, []float64{1, 2}, opts, sax.ErrNilInput},
			{"right nil", []float64{1, 2}, nil, opts, sax.ErrNilInput},
			{"left constant before right nil", []float64{1, 1}, nil, opts, sax.ErrZeroVariance},
			{"unequal n", []float64{1, 2}, []float64{1, 2, 3}, opts, sax.ErrIncompatible},
			{"right nonfinite", []float64{1, 2}, []float64{1, math.NaN()}, opts, sax.ErrNonFinite},
		} {
			t.Run(tc.name, func(t *testing.T) {
				distance, err := sax.MindistValues(tc.left, tc.right, tc.opts)
				mustErrorIs(t, err, tc.want)
				mustNear(t, distance, 0, 0, 0, "no partial distance")
			})
		}
		distance, err := sax.MindistValues([]float64{1, 2}, []float64{1, 2}, opts)
		if err != nil {
			t.Fatal(err)
		}
		mustNear(t, distance, 0, 0, 0, "one-shot distance")
	})
}

func TestMindist_LeftRightAdmissionAndNumericBoundaries(t *testing.T) {
	opts := sax.DefaultOptions(3)
	opts.ReturnPAA = true
	encoder, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	good, err := encoder.Encode([]float64{0, 2, 6})
	if err != nil {
		t.Fatal(err)
	}
	var absent *sax.Encoder
	var uninitialized sax.Encoder
	for _, tc := range []struct {
		encoder     *sax.Encoder
		left, right *sax.Result
		want        error
	}{
		{absent, good, good, sax.ErrNilInput},
		{&uninitialized, good, good, sax.ErrInvalidOption},
		{encoder, nil, good, sax.ErrNilInput},
		{encoder, good, nil, sax.ErrNilInput},
	} {
		distance, err := tc.encoder.Mindist(tc.left, tc.right)
		mustErrorIs(t, err, tc.want)
		mustNear(t, distance, 0, 0, 0, "receiver/presence failure")
	}
	for _, tc := range []struct {
		name   string
		mutate func(*sax.Result)
		want   error
	}{
		{"word shape", func(r *sax.Result) { r.Symbols = r.Symbols[:1] }, sax.ErrShape},
		{"PAA shape", func(r *sax.Result) { r.PAA = r.PAA[:1] }, sax.ErrShape},
		{"source length", func(r *sax.Result) { r.OriginalLength++ }, sax.ErrIncompatible},
		{"encoding identity", func(r *sax.Result) { r.EncodingID = "other" }, sax.ErrIncompatible},
		{"version identity", func(r *sax.Result) { r.NormalizationVersion = "old" }, sax.ErrIncompatible},
		{"zero sigma", func(r *sax.Result) { r.StdDev = 0 }, sax.ErrShape},
		{"negative sigma", func(r *sax.Result) { r.StdDev = -1 }, sax.ErrShape},
		{"nonfinite sigma", func(r *sax.Result) { r.StdDev = math.Inf(1) }, sax.ErrNonFinite},
		{"missing ID", func(r *sax.Result) { r.EncodingID = "" }, sax.ErrIncompatible},
		{"nonfinite PAA", func(r *sax.Result) { r.PAA = []float64{0, math.NaN(), 0} }, sax.ErrNonFinite},
		{"false symbol", func(r *sax.Result) { r.Symbols = []uint16{0, 0, 0} }, sax.ErrShape},
		{"false degeneracy", func(r *sax.Result) { r.Degenerate = true; r.StdDev = 0 }, sax.ErrShape},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := *good
			tc.mutate(&bad)
			for _, pair := range [][2]*sax.Result{{&bad, good}, {good, &bad}} {
				distance, err := encoder.Mindist(pair[0], pair[1])
				mustErrorIs(t, err, tc.want)
				mustNear(t, distance, 0, 0, 0, "bad result")
			}
		})
	}
	// A finite per-bin gap can still overflow the accumulated norm, even at n=w.
	for _, tc := range []struct {
		name string
		n, w int
		gap  float64
	}{
		{"norm overflow", 2, 2, 1.4e308},
		{"final scaling overflow", 4, 1, 1e308},
		{"rounded division boundary", 9, 1, math.MaxFloat64 / 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := sax.Options{Segments: tc.w, Alphabet: 3, Normalization: sax.Raw, Breakpoints: []float64{0, tc.gap}, BreakpointID: "extreme-gap"}
			left, right := make([]float64, tc.n), make([]float64, tc.n)
			for i := range left {
				left[i] = -1
				right[i] = tc.gap
			}
			distance, err := sax.MindistValues(left, right, opts)
			mustErrorIs(t, err, sax.ErrOverflow)
			mustNear(t, distance, 0, 0, 0, "unpublishable distance")
		})
	}
}

// TestSAX_EmbeddedAssetRejected is intentionally run only in a separate
// fixture build selected by the verification script. Every case calls the
// normal constructor twice and proves that a cached admission failure cannot
// poison the custom-breakpoint path. No private production state is accessed.
func TestSAX_EmbeddedAssetRejected(t *testing.T) {
	if os.Getenv("LVLATH_SAX_CORRUPT_ASSET_FIXTURE") != "1" {
		t.Skip("run tools/sax/verify_asset_failures.py for isolated corrupt-asset builds")
	}
	for i := 0; i < 2; i++ {
		encoder, err := sax.NewEncoder(sax.DefaultOptions(2))
		mustErrorIs(t, err, sax.ErrInvalidOption)
		if encoder != nil {
			t.Fatal("corrupt embedded asset published an Encoder")
		}
	}
	result, err := sax.EncodeRawWithBreakpoints([]float64{-1, 1}, 2, []float64{0}, "custom")
	if err != nil {
		t.Fatal(err)
	}
	mustEqualSymbols(t, result.Symbols, []uint16{0, 1}, "custom mode unaffected")
}

// TestMindist_ZeroWordWithoutDiagnostics checks the uniquely determined zero
// bin even in the compact persisted representation without PAA coordinates.
func TestMindist_ZeroWordWithoutDiagnostics(t *testing.T) {
	opts := sax.DefaultOptions(3)
	opts.ConstantPolicy = sax.ZeroWord
	encoder, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	constant, err := encoder.Encode([]float64{12, 12, 12})
	if err != nil {
		t.Fatal(err)
	}
	varying, err := encoder.Encode([]float64{0, 2, 6})
	if err != nil {
		t.Fatal(err)
	}
	if constant.PAA != nil {
		t.Fatal("unexpected diagnostic PAA")
	}
	if _, err := encoder.Mindist(constant, varying); err != nil {
		t.Fatal(err)
	}
	broken := *constant
	broken.Symbols = []uint16{0, 0, 0}
	for _, pair := range [][2]*sax.Result{{&broken, varying}, {varying, &broken}} {
		distance, err := encoder.Mindist(pair[0], pair[1])
		mustErrorIs(t, err, sax.ErrShape)
		mustNear(t, distance, 0, 0, 0, "forged degenerate word")
	}
	// Exercise optional PAA as a separate invariant, not as a proxy for symbols.
	withPAA := opts
	withPAA.ReturnPAA = true
	brokenPAA := mustEncode(t, []float64{12, 12, 12}, withPAA)
	brokenPAA.PAA[0] = 0.01
	distance, err := encoder.Mindist(brokenPAA, varying)
	mustErrorIs(t, err, sax.ErrShape)
	mustNear(t, distance, 0, 0, 0, "nonzero degenerate PAA")
	// Raw diagnostics are identity metadata, not fitted population moments.
	raw := opts
	raw.Normalization = sax.Raw
	rawEncoder, err := sax.NewEncoder(raw)
	if err != nil {
		t.Fatal(err)
	}
	rawWord, err := rawEncoder.Encode([]float64{0, 2, 6})
	if err != nil {
		t.Fatal(err)
	}
	badRaw := *rawWord
	badRaw.Mean = 1
	distance, err = rawEncoder.Mindist(&badRaw, rawWord)
	mustErrorIs(t, err, sax.ErrShape)
	mustNear(t, distance, 0, 0, 0, "nonidentity raw diagnostics")
}
