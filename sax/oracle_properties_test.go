// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External mathematical oracles use exact rational fractional geometry and
// independent two-pass population fitting. They do not call paa.Transform,
// private sax helpers, or the production metric to calculate expected values.
package sax_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/sax"
)

func TestPAA_RationalNearestEvenFractionalOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(20261111))
	for iteration := 0; iteration < 1200; iteration++ {
		length := 2 + rng.Intn(20)
		segments := 1 + rng.Intn(length)
		data := make([]float64, length)
		for i := range data {
			// Dyadic fractions ensure every input is exactly representable.
			data[i] = float64(rng.Intn(61)-30) / 8
		}
		opts := sax.Options{Segments: segments, Alphabet: 3,
			Normalization: sax.Raw, Breakpoints: []float64{-0.25, 0.25},
			BreakpointID: "rational-fixture", ReturnPAA: true}
		result, err := sax.Encode(data, opts)
		if err != nil {
			t.Fatalf("iteration=%d: %v", iteration, err)
		}
		oracle := rationalPAA(data, segments)
		for j := range oracle {
			expected, _ := oracle[j].Float64()
			if math.Float64bits(result.PAA[j]) != math.Float64bits(expected) {
				t.Fatalf("iter=%d n=%d w=%d j=%d PAA=%.17g want nearest-even %.17g",
					iteration, length, segments, j, result.PAA[j], expected)
			}
		}
	}
}

func TestMindist_ExactRationalJensenOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(20261112))
	for iteration := 0; iteration < 1200; iteration++ {
		length := 3 + rng.Intn(24)
		segments := 1 + rng.Intn(length)
		left, right := make([]float64, length), make([]float64, length)
		for i := range left {
			left[i] = float64(rng.Intn(201)-100) / 16
			right[i] = float64(rng.Intn(201)-100) / 16
		}
		opts := sax.Options{Segments: segments, Alphabet: 5,
			Normalization: sax.Raw, Breakpoints: []float64{-2, -0.5, 0.5, 2},
			BreakpointID: "dyadic", ReturnPAA: true}
		encoder, err := sax.NewEncoder(opts)
		if err != nil {
			t.Fatal(err)
		}
		leftWord, err := encoder.Encode(left)
		if err != nil {
			t.Fatal(err)
		}
		rightWord, err := encoder.Encode(right)
		if err != nil {
			t.Fatal(err)
		}
		mindist, err := encoder.Mindist(leftWord, rightWord)
		if err != nil {
			t.Fatal(err)
		}
		exactPAA := rationalSquareRoot(rationalPAASquared(rationalPAA(left, segments), rationalPAA(right, segments), length))
		exactEuclidean := rationalSquareRoot(rationalEuclideanSquared(left, right))
		budget := 3e-13 * math.Max(1, exactEuclidean)
		if mindist > exactPAA+budget || exactPAA > exactEuclidean+budget {
			t.Fatalf("iter=%d n=%d w=%d: mindist %.17g, exact PAA %.17g, exact Euclidean %.17g",
				iteration, length, segments, mindist, exactPAA, exactEuclidean)
		}
	}
}

func TestMindist_IndependentPopulationZOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(20261113))
	for iteration := 0; iteration < 900; iteration++ {
		length := 5 + rng.Intn(30)
		segments := 1 + rng.Intn(length)
		left, right := make([]float64, length), make([]float64, length)
		for i := range left {
			left[i] = float64(rng.Intn(1000)-500)/10 + 12
			right[i] = float64(rng.Intn(1000)-500)/10 - 7
		}
		opts := sax.DefaultOptions(segments)
		opts.ReturnPAA = true
		encoder, err := sax.NewEncoder(opts)
		if err != nil {
			t.Fatal(err)
		}
		leftWord, err := encoder.Encode(left)
		if err != nil {
			t.Fatal(err)
		}
		rightWord, err := encoder.Encode(right)
		if err != nil {
			t.Fatal(err)
		}
		bound, err := encoder.Mindist(leftWord, rightWord)
		if err != nil {
			t.Fatal(err)
		}
		leftZ, leftMean, leftStd := referenceZ(left)
		rightZ, rightMean, rightStd := referenceZ(right)
		mustNear(t, leftWord.Mean, leftMean, 5e-13, 5e-13, "independent mean")
		mustNear(t, leftWord.StdDev, leftStd, 5e-13, 5e-13, "independent pop sigma")
		mustNear(t, rightWord.Mean, rightMean, 5e-13, 5e-13, "independent mean right")
		mustNear(t, rightWord.StdDev, rightStd, 5e-13, 5e-13, "independent pop sigma right")
		paad := rationalSquareRoot(rationalPAASquared(rationalPAA(leftZ, segments), rationalPAA(rightZ, segments), length))
		euclidean := rationalSquareRoot(rationalEuclideanSquared(leftZ, rightZ))
		budget := 5e-11 * math.Max(1, euclidean)
		if bound > paad+budget || paad > euclidean+budget {
			t.Fatalf("Z iteration=%d n=%d w=%d: MINDIST=%g PAA=%g Euclidean=%g", iteration, length, segments, bound, paad, euclidean)
		}
	}
}
