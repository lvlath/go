// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External adversarial tests mutate only caller-owned Results and assert
// independent MINDIST validation without private field access or reflect.
package sax_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/sax"
)

func TestMindist_RejectsCorruptPublicResultMetadata(t *testing.T) {
	opts := sax.Options{Segments: 4, Alphabet: 3, Normalization: sax.Raw,
		Breakpoints: []float64{0, 1}, BreakpointID: "physical-units", ReturnPAA: true}
	enc, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	source, err := enc.Encode([]float64{-1, -0.5, 0.5, 2})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name   string
		mutate func(*sax.Result)
		want   error
	}{
		{"original length zero", func(r *sax.Result) { r.OriginalLength = 0 }, sax.ErrShape},
		{"segments zero", func(r *sax.Result) { r.Segments = 0 }, sax.ErrShape},
		{"w greater than n", func(r *sax.Result) { r.OriginalLength = 3 }, sax.ErrShape},
		{"different n", func(r *sax.Result) { r.OriginalLength = 5 }, sax.ErrIncompatible},
		{"different alphabet", func(r *sax.Result) { r.Alphabet = 4 }, sax.ErrIncompatible},
		{"symbol length", func(r *sax.Result) { r.Symbols = r.Symbols[:2] }, sax.ErrShape},
		{"symbol outside alphabet", func(r *sax.Result) { r.Symbols[0] = 3 }, sax.ErrShape},
		{"PAA length", func(r *sax.Result) { r.PAA = r.PAA[:2] }, sax.ErrShape},
		{"NaN PAA", func(r *sax.Result) { r.PAA[0] = math.NaN() }, sax.ErrNonFinite},
		{"positive infinity PAA", func(r *sax.Result) { r.PAA[0] = math.Inf(1) }, sax.ErrNonFinite},
		{"NaN mean", func(r *sax.Result) { r.Mean = math.NaN() }, sax.ErrNonFinite},
		{"negative sigma", func(r *sax.Result) { r.StdDev = -1 }, sax.ErrShape},
		{"infinity sigma", func(r *sax.Result) { r.StdDev = math.Inf(1) }, sax.ErrNonFinite},
		{"missing encoding ID", func(r *sax.Result) { r.EncodingID = "" }, sax.ErrIncompatible},
		{"fake encoding ID", func(r *sax.Result) { r.EncodingID = "fake" }, sax.ErrIncompatible},
		{"missing version", func(r *sax.Result) { r.NormalizationVersion = "" }, sax.ErrIncompatible},
		{"unknown version", func(r *sax.Result) { r.NormalizationVersion = "future-version" }, sax.ErrIncompatible},
		{"raw mean changed", func(r *sax.Result) { r.Mean = 123 }, sax.ErrShape},
		{"raw sigma changed", func(r *sax.Result) { r.StdDev = 0 }, sax.ErrShape},
		{"raw falsely degenerate", func(r *sax.Result) { r.Degenerate = true }, sax.ErrShape},
		{"PAA no longer matches symbol", func(r *sax.Result) { r.PAA[0] = 2 }, sax.ErrShape},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			copyValue := *source
			copyValue.Symbols = append([]uint16(nil), source.Symbols...)
			copyValue.PAA = append([]float64(nil), source.PAA...)
			test.mutate(&copyValue)
			d, compareErr := enc.Mindist(source, &copyValue)
			mustErrorIs(t, compareErr, test.want)
			mustNear(t, d, 0, 0, 0, "no partial distance")
		})
	}
}

func TestMindist_ZPopulationMetadataAndPolicyValidation(t *testing.T) {
	opts := sax.DefaultOptions(4)
	opts.ReturnPAA = true
	opts.ConstantPolicy = sax.ZeroWord
	enc, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := enc.Encode([]float64{1, 2, 4, 8})
	if err != nil {
		t.Fatal(err)
	}
	constant, err := enc.Encode([]float64{3, 3, 3, 3})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = enc.Mindist(baseline, constant); err != nil {
		t.Fatalf("degenerate and non-degenerate words are compatible under ZeroWord: %v", err)
	}
	altered := *baseline
	altered.Degenerate = true
	mustErrorIs(t, func() error { _, err = enc.Mindist(baseline, &altered); return err }(), sax.ErrShape)
	altered = *constant
	altered.StdDev = 1
	mustErrorIs(t, func() error { _, err = enc.Mindist(baseline, &altered); return err }(), sax.ErrShape)

	// A nonzero PAA coordinate still fits the central Gaussian bin, but
	// contradicts the exact-zero meaning of a degenerate ZeroWord result.
	altered = *constant
	altered.PAA = append([]float64(nil), constant.PAA...)
	altered.PAA[0] = 0.01
	mustErrorIs(t, func() error { _, err = enc.Mindist(baseline, &altered); return err }(), sax.ErrShape)

	// A RejectConstant encoder has another EncodingID even for nonconstant
	// input, so forged degeneracy cannot be smuggled into its comparison.
	opts.ConstantPolicy = sax.RejectConstant
	reject, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	x, err := reject.Encode([]float64{1, 2, 4, 8})
	if err != nil {
		t.Fatal(err)
	}
	altered = *x
	altered.Degenerate = true
	altered.StdDev = 0
	mustErrorIs(t, func() error { _, err = reject.Mindist(x, &altered); return err }(), sax.ErrShape)
}

func TestMindist_RejectsMissingEncoderAndResults(t *testing.T) {
	var absent *sax.Encoder
	var zeroValue sax.Encoder
	mustErrorIs(t, func() error { _, err := absent.Mindist(nil, nil); return err }(), sax.ErrNilInput)
	mustErrorIs(t, func() error { _, err := zeroValue.Mindist(nil, nil); return err }(), sax.ErrInvalidOption)
	opts := sax.DefaultOptions(2)
	enc, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	mustErrorIs(t, func() error { _, err = enc.Mindist(nil, &sax.Result{}); return err }(), sax.ErrNilInput)
	mustErrorIs(t, func() error { _, err = enc.Mindist(&sax.Result{}, nil); return err }(), sax.ErrNilInput)
}

func TestMindist_OverflowFromFinalFractionalScale(t *testing.T) {
	// The bin gap fits float64 but multiplying by sqrt(n/w)=2 does not.
	opts := sax.Options{Segments: 1, Alphabet: 3, Normalization: sax.Raw,
		Breakpoints: []float64{-5e307, 5e307}, BreakpointID: "extreme"}
	enc, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	left, err := enc.Encode([]float64{-6e307, -6e307, -6e307, -6e307})
	if err != nil {
		t.Fatal(err)
	}
	right, err := enc.Encode([]float64{6e307, 6e307, 6e307, 6e307})
	if err != nil {
		t.Fatal(err)
	}
	got, err := enc.Mindist(left, right)
	mustErrorIs(t, err, sax.ErrOverflow)
	mustNear(t, got, 0, 0, 0, "no partial overflowed distance")
}
