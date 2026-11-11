// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Identity tests treat configuration bytes as a mathematical compatibility
// protocol, including negative zero and audit-label independence.
package sax_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/sax"
)

func TestEncodingID_CanonicalParametersAndDistinctPolicies(t *testing.T) {
	base := sax.Options{Segments: 4, Alphabet: 3, Normalization: sax.Raw,
		Breakpoints: []float64{0, 1}, BreakpointID: "first", ReturnPAA: false}
	makeID := func(opts sax.Options) string {
		t.Helper()
		enc, err := sax.NewEncoder(opts)
		if err != nil {
			t.Fatal(err)
		}
		result, err := enc.Encode([]float64{-1, 0, 1, 2})
		if err != nil {
			t.Fatal(err)
		}
		return result.EncodingID
	}
	canonical := makeID(base)

	identical := base
	identical.Breakpoints = []float64{math.Copysign(0, -1), 1}
	identical.BreakpointID = "renamed label"
	identical.ReturnPAA = true
	mustEqualString(t, makeID(identical), canonical, "minus-zero and audit/output independence")

	variants := []struct {
		name string
		edit func(*sax.Options)
	}{
		{"segments", func(o *sax.Options) { o.Segments = 3 }},
		{"alphabet", func(o *sax.Options) { o.Alphabet = 4; o.Breakpoints = []float64{0, 1, 2} }},
		{"normalization", func(o *sax.Options) { o.Normalization = sax.ZPopulation }},
		{"constant policy", func(o *sax.Options) { o.ConstantPolicy = sax.ZeroWord }},
		{"boundary value", func(o *sax.Options) { o.Breakpoints = []float64{0, math.Nextafter(1, 2)} }},
	}
	for _, variant := range variants {
		t.Run(variant.name, func(t *testing.T) {
			value := base
			variant.edit(&value)
			if id := makeID(value); id == canonical {
				t.Fatal("changed mathematical configuration retained identical EncodingID")
			}
		})
	}
}

func TestOptions_NilVsEmptyBreakpoints(t *testing.T) {
	opts := sax.DefaultOptions(2)
	opts.Breakpoints = []float64{}
	opts.BreakpointID = "empty-is-custom"
	enc, err := sax.NewEncoder(opts)
	mustErrorIs(t, err, sax.ErrShape)
	if enc != nil {
		t.Fatal("invalid constructor published an encoder")
	}
}

func TestOptions_UnknownEnumerationValues(t *testing.T) {
	cases := []struct {
		name string
		edit func(*sax.Options)
	}{
		{"normalization", func(o *sax.Options) { o.Normalization = 255 }},
		{"constant-policy", func(o *sax.Options) { o.ConstantPolicy = 255 }},
		{"alphabet-small", func(o *sax.Options) { o.Alphabet = 1 }},
		{"alphabet-large", func(o *sax.Options) { o.Alphabet = 257 }},
		{"segments-zero", func(o *sax.Options) { o.Segments = 0 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			opts := sax.DefaultOptions(2)
			tc.edit(&opts)
			result, err := sax.NewEncoder(opts)
			mustErrorIs(t, err, sax.ErrInvalidOption)
			if result != nil {
				t.Fatal("invalid option must not publish encoder")
			}
		})
	}
}
