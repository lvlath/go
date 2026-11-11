// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package sax_test

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/sax"
)

func TestEncode_AllBinsAndBreakpointEquality(t *testing.T) {
	opts := sax.Options{Segments: 5, Alphabet: 3, Normalization: sax.Raw, Breakpoints: []float64{0, 1}, BreakpointID: "custom", ReturnPAA: true}
	got, err := sax.Encode([]float64{-1, 0, 0.5, 1, 2}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got.Symbols, []uint16{0, 1, 1, 2, 2}) {
		t.Fatalf("symbols=%v", got.Symbols)
	}
	if !slices.Equal(got.PAA, []float64{-1, 0, 0.5, 1, 2}) {
		t.Fatalf("PAA=%v", got.PAA)
	}
}

func TestEncode_ConstantPolicy(t *testing.T) {
	opts := sax.DefaultOptions(4)
	opts.Alphabet = 2
	_, err := sax.Encode([]float64{7, 7, 7, 7}, opts)
	if !errors.Is(err, sax.ErrZeroVariance) {
		t.Fatalf("err=%v", err)
	}
	opts.ConstantPolicy = sax.ZeroWord
	got, err := sax.Encode([]float64{7, 7, 7, 7}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Degenerate || got.Mean != 7 || got.StdDev != 0 || !slices.Equal(got.Symbols, []uint16{1, 1, 1, 1}) {
		t.Fatalf("degenerate=%v mean=%v std=%v symbols=%v", got.Degenerate, got.Mean, got.StdDev, got.Symbols)
	}
	opts.Normalization = sax.Raw
	got, err = sax.Encode([]float64{7, 7, 7, 7}, opts)
	if err != nil || got.Degenerate || got.Mean != 0 || got.StdDev != 1 {
		t.Fatalf("raw=%+v err=%v", got, err)
	}
}

func TestMindist_RejectsMismatchedBreakpointContent(t *testing.T) {
	a := sax.Options{Segments: 2, Alphabet: 3, Normalization: sax.Raw, Breakpoints: []float64{0, 1}, BreakpointID: "custom"}
	b := a
	b.Breakpoints = []float64{0, 2}
	ea, err := sax.NewEncoder(a)
	if err != nil {
		t.Fatal(err)
	}
	eb, err := sax.NewEncoder(b)
	if err != nil {
		t.Fatal(err)
	}
	x, err := ea.Encode([]float64{-1, 3})
	if err != nil {
		t.Fatal(err)
	}
	y, err := eb.Encode([]float64{-1, 3})
	if err != nil {
		t.Fatal(err)
	}
	if x.EncodingID == y.EncodingID {
		t.Fatal("fake compatibility")
	}
	if _, err = ea.Mindist(x, y); !errors.Is(err, sax.ErrIncompatible) {
		t.Fatalf("err=%v", err)
	}
}

func TestEncode_ResultDetached(t *testing.T) {
	opts := sax.Options{Segments: 3, Alphabet: 3, Normalization: sax.Raw, Breakpoints: []float64{0, 1}, BreakpointID: "custom", ReturnPAA: true}
	source := []float64{-1, 0, 2}
	e, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := e.Encode(source)
	if err != nil {
		t.Fatal(err)
	}
	baseline.Symbols[0] = 77
	baseline.PAA[0] = 999
	opts.Breakpoints[0] = -999
	got, err := e.Encode(source)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got.Symbols, []uint16{0, 1, 2}) || got.PAA[0] != -1 || !slices.Equal(source, []float64{-1, 0, 2}) {
		t.Fatalf("state leaked: %+v source=%v", got, source)
	}
}

func TestSAXShortlist_IsNotExactDTWCertificate(t *testing.T) {
	opts := sax.Options{Segments: 3, Alphabet: 3, Normalization: sax.Raw, Breakpoints: []float64{0.25, 0.75}, BreakpointID: "custom"}
	enc, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	q, err := enc.Encode([]float64{0, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	c, err := enc.Encode([]float64{0, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(q.Symbols, []uint16{0, 0, 2}) || !slices.Equal(c.Symbols, []uint16{0, 2, 2}) {
		t.Fatalf("q=%v c=%v", q.Symbols, c.Symbols)
	}
	mindist, err := enc.Mindist(q, c)
	if err != nil {
		t.Fatal(err)
	}
	// Independent DP oracle: unconstrained sum of squared per-pair costs.
	x, y := []float64{0, 0, 1}, []float64{0, 1, 1}
	dp := make([][]float64, len(x)+1)
	for i := range dp {
		dp[i] = make([]float64, len(y)+1)
		for j := range dp[i] {
			dp[i][j] = math.Inf(1)
		}
	}
	dp[0][0] = 0
	for i := 1; i <= len(x); i++ {
		for j := 1; j <= len(y); j++ {
			delta := x[i-1] - y[j-1]
			dp[i][j] = delta*delta + math.Min(dp[i-1][j-1], math.Min(dp[i-1][j], dp[i][j-1]))
		}
	}
	if mindist != 0.5 || dp[len(x)][len(y)] != 0 {
		t.Fatalf("MINDIST=%v DTW=%v", mindist, dp[len(x)][len(y)])
	}
}

func TestResult_IsNilInterface(t *testing.T) {
	var p *sax.Result
	var n core.Nilable = p
	if !n.IsNil() {
		t.Fatal("typed nil not recognized")
	}
	n = &sax.Result{}
	if n.IsNil() {
		t.Fatal("zero-value pointer falsely nil")
	}
}

func TestEncoder_AllAlphabets(t *testing.T) {
	for a := 2; a <= 256; a++ {
		opts := sax.DefaultOptions(4)
		opts.Alphabet = a
		opts.Normalization = sax.Raw
		enc, err := sax.NewEncoder(opts)
		if err != nil {
			t.Fatalf("a=%d: %v", a, err)
		}
		x, err := enc.Encode([]float64{-100, -1, 1, 100})
		if err != nil {
			t.Fatalf("a=%d: %v", a, err)
		}
		if len(x.Symbols) != 4 || x.Symbols[0] != 0 || int(x.Symbols[3]) != a-1 {
			t.Fatalf("a=%d: symbols=%v", a, x.Symbols)
		}
	}
}

func TestErrors_ValidationPrecedence(t *testing.T) {
	opts := sax.DefaultOptions(2)
	opts.Alphabet = 1
	if _, err := sax.Encode(nil, opts); !errors.Is(err, sax.ErrInvalidOption) {
		t.Fatalf("options before nil: %v", err)
	}
	opts.Alphabet = 8
	if _, err := sax.Encode(nil, opts); !errors.Is(err, sax.ErrNilInput) {
		t.Fatalf("nil: %v", err)
	}
	if _, err := sax.Encode([]float64{}, opts); !errors.Is(err, sax.ErrEmptyInput) {
		t.Fatalf("empty: %v", err)
	}
	if _, err := sax.Encode([]float64{math.NaN()}, opts); !errors.Is(err, sax.ErrInvalidOption) {
		t.Fatalf("shape before NaN: %v", err)
	}
	if _, err := sax.Encode([]float64{math.NaN(), 1}, opts); !errors.Is(err, sax.ErrNonFinite) {
		t.Fatalf("non-finite: %v", err)
	}
	var e *sax.Encoder
	if _, err := e.Encode([]float64{1}); !errors.Is(err, sax.ErrNilInput) {
		t.Fatalf("typed nil encoder: %v", err)
	}
	if _, err := e.Mindist(nil, nil); !errors.Is(err, sax.ErrNilInput) {
		t.Fatalf("typed nil mindist: %v", err)
	}
}

func TestMindist_RejectsExternallyMalformed(t *testing.T) {
	opts := sax.Options{Segments: 2, Alphabet: 3, Normalization: sax.Raw, Breakpoints: []float64{0, 1}, BreakpointID: "custom"}
	enc, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	x, err := enc.Encode([]float64{-1, 2})
	if err != nil {
		t.Fatal(err)
	}
	y, err := enc.Encode([]float64{2, -1})
	if err != nil {
		t.Fatal(err)
	}
	y.Symbols = []uint16{2, 3}
	if _, err = enc.Mindist(x, y); !errors.Is(err, sax.ErrShape) {
		t.Fatalf("out of alphabet: %v", err)
	}
	y.Symbols = []uint16{2, 0}
	y.EncodingID = "fake"
	if _, err = enc.Mindist(x, y); !errors.Is(err, sax.ErrIncompatible) {
		t.Fatalf("fake ID: %v", err)
	}
}

func TestEncoder_EncodingIdentityIgnoresAuditLabelAndReturnPAA(t *testing.T) {
	a := sax.Options{Segments: 3, Alphabet: 3, Normalization: sax.Raw, Breakpoints: []float64{math.Copysign(0, -1), 1}, BreakpointID: "first"}
	b := a
	b.Breakpoints = []float64{0, 1}
	b.BreakpointID = "renamed"
	b.ReturnPAA = true
	ea, err := sax.NewEncoder(a)
	if err != nil {
		t.Fatal(err)
	}
	eb, err := sax.NewEncoder(b)
	if err != nil {
		t.Fatal(err)
	}
	x, err := ea.Encode([]float64{-1, 0.5, 3})
	if err != nil {
		t.Fatal(err)
	}
	y, err := eb.Encode([]float64{-1, 0.5, 3})
	if err != nil {
		t.Fatal(err)
	}
	if x.EncodingID != y.EncodingID {
		t.Fatalf("different mathematical identities: %s %s", x.EncodingID, y.EncodingID)
	}
	if x.PAA != nil || y.PAA == nil {
		t.Fatal("ReturnPAA artifact mismatch")
	}
	d, err := ea.Mindist(x, y)
	if err != nil || d != 0 {
		t.Fatalf("audit label falsely affects compatibility: %g %v", d, err)
	}
}

func TestMindist_OverflowFromSeparatedExtremeBins(t *testing.T) {
	opts := sax.Options{Segments: 1, Alphabet: 3, Normalization: sax.Raw, Breakpoints: []float64{-1e308, 1e308}, BreakpointID: "extreme"}
	enc, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	a, err := enc.Encode([]float64{-1.5e308})
	if err != nil {
		t.Fatal(err)
	}
	b, err := enc.Encode([]float64{1.5e308})
	if err != nil {
		t.Fatal(err)
	}
	if value, err := enc.Mindist(a, b); !errors.Is(err, sax.ErrOverflow) || value != 0 {
		t.Fatalf("overflow: %.5g %v", value, err)
	}
}

func TestEncoder_RejectMalformedCustomBreakpoints(t *testing.T) {
	base := sax.Options{Segments: 3, Alphabet: 3, Normalization: sax.Raw, Breakpoints: []float64{0, 1}, BreakpointID: "custom"}
	cases := []struct {
		label    string
		breaks   []float64
		id       string
		expected error
	}{
		{"shape", []float64{0}, "custom", sax.ErrShape},
		{"duplicate", []float64{0, 0}, "custom", sax.ErrInvalidOption},
		{"not-finite", []float64{0, math.Inf(1)}, "custom", sax.ErrInvalidOption},
		{"no-label", []float64{0, 1}, "", sax.ErrInvalidOption},
	}
	for _, tc := range cases {
		t.Run(tc.label, func(t *testing.T) {
			opts := base
			opts.Breakpoints = tc.breaks
			opts.BreakpointID = tc.id
			if _, err := sax.NewEncoder(opts); !errors.Is(err, tc.expected) {
				t.Fatalf("err=%v want=%v", err, tc.expected)
			}
		})
	}
}
