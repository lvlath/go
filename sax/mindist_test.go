// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package sax_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/sax"
)

func TestMindist_FractionalPAAEuclideanBound(t *testing.T) {
	rng := rand.New(rand.NewSource(20261111))
	for iter := 0; iter < 2500; iter++ {
		n := 5 + rng.Intn(25)
		w := 1 + rng.Intn(n)
		opts := sax.Options{Segments: w, Alphabet: 3, Normalization: sax.Raw, Breakpoints: []float64{-0.25, 0.25}, BreakpointID: "oracle", ReturnPAA: true}
		enc, err := sax.NewEncoder(opts)
		if err != nil {
			t.Fatal(err)
		}
		q, c := make([]float64, n), make([]float64, n)
		for i := range q {
			q[i] = rng.Float64()*4 - 2
			c[i] = rng.Float64()*4 - 2
		}
		lhs, err := enc.Encode(q)
		if err != nil {
			t.Fatal(err)
		}
		rhs, err := enc.Encode(c)
		if err != nil {
			t.Fatal(err)
		}
		got, err := enc.Mindist(lhs, rhs)
		if err != nil {
			t.Fatal(err)
		}
		// Independent geometric oracle: integrate each original sample overlap.
		// No paa.Transform and no sax helper participates in the PAA oracle.
		paaSq := 0.0
		for j := 0; j < w; j++ {
			left := float64(j) * float64(n) / float64(w)
			right := float64(j+1) * float64(n) / float64(w)
			integral := 0.0
			for i := 0; i < n; i++ {
				overlap := math.Max(0, math.Min(right, float64(i+1))-math.Max(left, float64(i)))
				integral += overlap * (q[i] - c[i])
			}
			delta := integral / (right - left)
			paaSq += (right - left) * delta * delta
		}
		euclideanSq := 0.0
		for i := range q {
			euclideanSq += math.Pow(q[i]-c[i], 2)
		}
		paad, euclidean := math.Sqrt(paaSq), math.Sqrt(euclideanSq)
		if got > paad+1e-11 || paad > euclidean+1e-11 {
			t.Fatalf("iter=%d n=%d w=%d: MINDIST %.17g > PAA %.17g > Euclidean %.17g", iter, n, w, got, paad, euclidean)
		}
	}
}

func TestMindist_AdjacentAndSeparatedBins(t *testing.T) {
	opts := sax.Options{Segments: 1, Alphabet: 3, Normalization: sax.Raw, Breakpoints: []float64{0, 1}, BreakpointID: "metric"}
	enc, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	x, err := enc.Encode([]float64{-1})
	if err != nil {
		t.Fatal(err)
	}
	y, err := enc.Encode([]float64{0.5})
	if err != nil {
		t.Fatal(err)
	}
	d, err := enc.Mindist(x, y)
	if err != nil || d != 0 {
		t.Fatalf("adjacent %g %v", d, err)
	}
	opts.Segments = 4
	enc, err = sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	x, err = enc.Encode([]float64{-1, -1, -1, -1})
	if err != nil {
		t.Fatal(err)
	}
	y, err = enc.Encode([]float64{2, 2, 2, 2})
	if err != nil {
		t.Fatal(err)
	}
	d, err = enc.Mindist(x, y)
	if err != nil || d != 2 {
		t.Fatalf("separated %g %v", d, err)
	}
}
